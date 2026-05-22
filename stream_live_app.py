"""
Minimal browser-microphone chunk streaming demo for DyStream.

This is a pragmatic live experiment:
- Upload one reference image and initialize the model/cache once.
- Browser MediaRecorder records microphone chunks and uploads them repeatedly.
- The server processes chunks sequentially and returns MP4 chunks.
- The browser queues returned MP4 chunks and auto-plays them back-to-back.

It is chunk-streaming, not low-latency WebRTC.
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from PIL import Image

import app as base
from stream_app import encode_audio_window, latents_to_video_frames_batched, preprocess_reference_cached, stamp


ROOT = Path(__file__).resolve().parent
SESSION_ROOT = ROOT / "outputs" / "stream_live_sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class StreamSession:
    session_id: str
    out_dir: Path
    resized_pil: Image.Image
    masked_pil: Image.Image
    motion_latent_cpu: torch.Tensor
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    process_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_index: int = 0
    completed: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    worker_started: bool = False
    config: dict = field(default_factory=dict)
    audio_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    generated_audio_frames: int = 0
    past_motion: torch.Tensor | None = None
    vad_ema: float = 0.0
    generation: int = 0


sessions = {}
app = FastAPI()
app.mount("/sessions", StaticFiles(directory=str(SESSION_ROOT)), name="sessions")


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_once():
    base.load_dystream_model()
    base.load_visualization_model()


def get_anchor_motion(session: StreamSession):
    motion_latent = session.motion_latent_cpu.to(base.DEVICE)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.unsqueeze(0)
    if motion_latent.dim() == 2:
        motion_latent = motion_latent.unsqueeze(0)
    return motion_latent[:, 0:1, :]


def append_audio(session: StreamSession, audio_path, audio_sr):
    audio_np, _ = librosa.load(audio_path, sr=audio_sr)
    audio_np = audio_np.astype(np.float32)
    if session.audio_buffer.size == 0:
        session.audio_buffer = audio_np
    else:
        session.audio_buffer = np.concatenate([session.audio_buffer, audio_np])
    return audio_np


def audio_rms(audio_np):
    if audio_np.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_np.astype(np.float32))) + 1e-12))


def update_audio_gate(session: StreamSession, audio_np):
    rms = audio_rms(audio_np)
    rms_db = 20.0 * np.log10(max(rms, 1e-8))
    floor_db = float(session.config.get("vad_floor_db", -46.0))
    speech_db = float(session.config.get("vad_speech_db", -30.0))
    if speech_db <= floor_db:
        raw_scale = 1.0
    else:
        raw_scale = float(np.clip((rms_db - floor_db) / (speech_db - floor_db), 0.0, 1.0))
    attack = float(session.config.get("vad_attack", 0.35))
    release = float(session.config.get("vad_release", session.config.get("vad_smoothing", 0.85)))
    smoothing = attack if raw_scale > session.vad_ema else release
    session.vad_ema = smoothing * session.vad_ema + (1.0 - smoothing) * raw_scale
    min_scale = float(session.config.get("vad_min_scale", 0.0))
    scale = float(np.clip(session.vad_ema, min_scale, 1.0))
    return {
        "rms": rms,
        "rms_db": float(rms_db),
        "raw_scale": raw_scale,
        "audio_scale": scale,
        "vad_ema": float(session.vad_ema),
        "vad_smoothing": float(smoothing),
    }


async def drain_queue(queue):
    drained = 0
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        queue.task_done()
        drained += 1
    return drained


async def reset_session_stream(session: StreamSession):
    drained = await drain_queue(session.queue)
    async with session.lock:
        session.generation += 1
        session.next_index = 0
        session.completed.clear()
        session.errors.clear()
        session.audio_buffer = np.zeros(0, dtype=np.float32)
        session.generated_audio_frames = 0
        session.past_motion = None
        session.vad_ema = 0.0
    return drained


def audio_window_for_generation(audio_buffer, start_frame, n_frames, prefix_frames, hop):
    first_frame = start_frame - prefix_frames
    sample_start = first_frame * hop
    sample_end = (first_frame + n_frames) * hop
    left_pad = max(0, -sample_start)
    right_pad = max(0, sample_end - len(audio_buffer))
    clipped_start = max(0, sample_start)
    clipped_end = min(len(audio_buffer), sample_end)
    window = audio_buffer[clipped_start:clipped_end]
    if left_pad or right_pad:
        window = np.pad(window, (left_pad, right_pad))
    return window.astype(np.float32, copy=False)


def rolling_generate_motion(session: StreamSession, model, audio_sr, pose_fps, timings, audio_scale):
    hop = int(audio_sr / pose_fps)
    prefix_frames = model.inpainting_length
    chunk_frames = max(1, int(session.config["motion_chunk_frames"]))
    available_frames = len(session.audio_buffer) // hop
    remaining = available_frames - session.generated_audio_frames
    if remaining <= 0:
        return None, []

    anchor_motion = get_anchor_motion(session)
    if session.past_motion is None:
        session.past_motion = anchor_motion.repeat(1, prefix_frames, 1).contiguous()

    chunks = []
    chunk_infos = []
    while session.generated_audio_frames < available_frames:
        start_frame = session.generated_audio_frames
        gen_frames = min(chunk_frames, available_frames - start_frame)
        n_frames = prefix_frames + gen_frames + 1
        audio_window = audio_window_for_generation(
            session.audio_buffer,
            start_frame,
            n_frames,
            prefix_frames,
            hop,
        )
        audio_other_window = np.zeros_like(audio_window)

        sub_t0 = stamp()
        t0 = stamp()
        audio_tensor, audio_other_tensor, feat_self, feat_other = encode_audio_window(
            model, audio_window, audio_other_window, n_frames
        )
        feat_self = feat_self * float(audio_scale)
        feat_other = feat_other * float(audio_scale)
        audio_time = stamp() - t0

        profile = {}
        out = model.one_clip_only_inference(
            per_compute_audio_feature=feat_self,
            per_compute_audio_other_feature=feat_other,
            audio_self=audio_tensor,
            audio_other=audio_other_tensor,
            past_audio_self=None,
            past_audio_other=None,
            past_motion=session.past_motion,
            gen_frames=gen_frames,
            anchor_latent=anchor_motion,
            noise_scheduler=base._noise_scheduler,
            num_inference_steps=int(session.config["denoising_steps"]),
            profile=profile,
            guidance_mode=session.config["guidance_mode"],
        )
        if audio_scale < 0.999:
            idle = session.past_motion[:, -1:].repeat(1, out.shape[1], 1)
            out = out * float(audio_scale) + idle * (1.0 - float(audio_scale))
        session.past_motion = torch.cat([session.past_motion, out], dim=1)[:, -prefix_frames:].contiguous()
        session.generated_audio_frames += gen_frames
        chunks.append(out)

        info = {
            "start_frame": int(start_frame),
            "frames": int(gen_frames),
            "audio_encoder": float(audio_time),
            "gpt": float(profile.get("gpt", 0.0)),
            "fm": float(profile.get("fm", 0.0)),
            "total": float(stamp() - sub_t0),
            "audio_scale": float(audio_scale),
        }
        chunk_infos.append(info)
        timings["motion_audio_encoder"] = timings.get("motion_audio_encoder", 0.0) + info["audio_encoder"]
        timings["motion_gpt"] = timings.get("motion_gpt", 0.0) + info["gpt"]
        timings["motion_fm"] = timings.get("motion_fm", 0.0) + info["fm"]

    return torch.cat(chunks, dim=1), chunk_infos


def process_audio_chunk(session: StreamSession, item: dict):
    timings = {}
    t_total = stamp()

    cfg = base._dystream_cfg
    model = base._dystream_model
    audio_sr = int(OmegaConf.select(cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(OmegaConf.select(cfg.config, "model.pose_fps", default=25))
    hop = int(audio_sr / pose_fps)

    t0 = stamp()
    new_audio = append_audio(session, item["audio_path"], audio_sr)
    gate = update_audio_gate(session, new_audio)
    timings["audio_prepare"] = stamp() - t0
    timings["vad_rms"] = gate["rms"]
    timings["vad_rms_db"] = gate["rms_db"]
    timings["vad_raw_scale"] = gate["raw_scale"]
    timings["audio_scale"] = gate["audio_scale"]
    timings["vad_ema"] = gate["vad_ema"]

    model.cfg_audio = float(session.config["cfg_audio"])
    model.cfg_audio_other = float(session.config["cfg_audio_other"])
    model.cfg_anchor = float(session.config["cfg_anchor"])
    model.cfg_all = float(session.config["cfg_all"])

    if base._dystream_ema is not None:
        base._dystream_ema.to(base.DEVICE)
        ema_ctx = base._dystream_ema.average_parameters(model.parameters())
    else:
        from contextlib import nullcontext
        ema_ctx = nullcontext()

    t0 = stamp()
    subchunks = []
    with torch.no_grad(), ema_ctx:
        motion_pred, subchunks = rolling_generate_motion(
            session,
            model,
            audio_sr,
            pose_fps,
            timings,
            audio_scale=gate["audio_scale"],
        )
    if motion_pred is None:
        motion_pred = get_anchor_motion(session)[:, :0]
    timings["motion_inference"] = stamp() - t0
    timings["motion_other"] = max(
        0.0,
        timings["motion_inference"]
        - timings.get("motion_gpt", 0.0)
        - timings.get("motion_fm", 0.0)
        - timings.get("motion_audio_encoder", 0.0),
    )
    timings["uploaded_audio_sec"] = float(len(new_audio) / audio_sr)
    timings["generated_audio_frames"] = int(session.generated_audio_frames)
    timings["subchunks"] = subchunks

    t0 = stamp()
    frames = latents_to_video_frames_batched(
        motion_pred,
        session.resized_pil,
        render_batch_size=int(session.config["render_batch_size"]),
        source_motion_latent=get_anchor_motion(session),
    )
    timings["render"] = stamp() - t0

    frame_urls = []
    skip_playback = bool(session.config.get("skip_first_playback", True)) and item["index"] == 0
    if item.get("save_frames", True) and not skip_playback:
        t0 = stamp()
        frame_dir = session.out_dir / f"frames_{item['index']:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in enumerate(frames):
            frame_path = frame_dir / f"{frame_idx:04d}.jpg"
            Image.fromarray(frame).save(frame_path, quality=88)
            frame_urls.append(f"/sessions/{session.session_id}/{frame_dir.name}/{frame_path.name}")
        timings["save_frames"] = stamp() - t0
    else:
        timings["save_frames"] = 0.0

    video_url = None
    if item.get("save_mp4", session.config.get("save_mp4", False)):
        t0 = stamp()
        video_path = session.out_dir / f"chunk_{item['index']:04d}.mp4"
        base.save_video_with_audio(frames, item["audio_path"], str(video_path), fps=pose_fps)
        timings["mux"] = stamp() - t0
        video_url = f"/sessions/{session.session_id}/{video_path.name}"
    else:
        timings["mux"] = 0.0
    timings["total"] = stamp() - t_total

    return {
        "index": item["index"],
        "video_url": video_url,
        "frame_urls": frame_urls,
        "fps": pose_fps,
        "frames": int(motion_pred.shape[1]),
        "duration": float(motion_pred.shape[1] / pose_fps),
        "skipped_playback": skip_playback,
        "timings": timings,
    }


async def session_worker(session: StreamSession):
    while True:
        item = await session.queue.get()
        try:
            async with session.process_lock:
                result = await asyncio.to_thread(process_audio_chunk, session, item)
            if item.get("generation") == session.generation:
                result["generation"] = session.generation
                session.completed.append(result)
                with open(session.out_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
        except Exception as exc:
            if item.get("generation") == session.generation:
                error = {"index": item.get("index"), "generation": item.get("generation"), "error": repr(exc)}
                session.errors.append(error)
        finally:
            session.queue.task_done()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/api/init_session")
async def init_session(
    image: UploadFile = File(...),
    denoising_steps: int = Form(1),
    guidance_mode: str = Form("all_only"),
    motion_chunk_frames: int = Form(8),
    render_batch_size: int = Form(8),
    cfg_audio: float = Form(0.5),
    cfg_audio_other: float = Form(0.5),
    cfg_anchor: float = Form(0.0),
    cfg_all: float = Form(1.0),
    save_mp4: bool = Form(False),
    skip_first_playback: bool = Form(True),
    vad_floor_db: float = Form(-46.0),
    vad_speech_db: float = Form(-30.0),
    vad_attack: float = Form(0.35),
    vad_release: float = Form(0.85),
    vad_min_scale: float = Form(0.0),
):
    if guidance_mode not in {"full_5way", "uncond_all_2way", "all_only"}:
        raise HTTPException(status_code=400, detail="Invalid guidance_mode")

    session_id = uuid.uuid4().hex[:12]
    out_dir = SESSION_ROOT / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "reference_upload"
    image_path.write_bytes(await image.read())

    t0 = stamp()
    load_once()
    image_pil = Image.open(image_path).convert("RGB")
    resized_pil, masked_pil, motion_latent_cpu, ref_cache_hit = preprocess_reference_cached(image_pil)
    init_time = stamp() - t0

    resized_pil.save(out_dir / "reference_resized.png")
    masked_pil.save(out_dir / "reference_masked.png")

    session = StreamSession(
        session_id=session_id,
        out_dir=out_dir,
        resized_pil=resized_pil,
        masked_pil=masked_pil,
        motion_latent_cpu=motion_latent_cpu,
        config={
            "denoising_steps": denoising_steps,
            "guidance_mode": guidance_mode,
            "motion_chunk_frames": motion_chunk_frames,
            "render_batch_size": render_batch_size,
            "cfg_audio": cfg_audio,
            "cfg_audio_other": cfg_audio_other,
            "cfg_anchor": cfg_anchor,
            "cfg_all": cfg_all,
            "save_mp4": save_mp4,
            "skip_first_playback": skip_first_playback,
            "vad_floor_db": vad_floor_db,
            "vad_speech_db": vad_speech_db,
            "vad_attack": vad_attack,
            "vad_release": vad_release,
            "vad_min_scale": vad_min_scale,
        },
    )
    sessions[session_id] = session
    if not session.worker_started:
        asyncio.create_task(session_worker(session))
        session.worker_started = True

    return {
        "session_id": session_id,
        "init_time": init_time,
        "ref_cache_hit": ref_cache_hit,
        "resized_url": f"/sessions/{session_id}/reference_resized.png",
        "masked_url": f"/sessions/{session_id}/reference_masked.png",
    }


@app.post("/api/update_config")
async def update_config(
    session_id: str = Form(...),
    vad_floor_db: float | None = Form(None),
    vad_speech_db: float | None = Form(None),
    vad_attack: float | None = Form(None),
    vad_release: float | None = Form(None),
    vad_min_scale: float | None = Form(None),
):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    updates = {
        "vad_floor_db": vad_floor_db,
        "vad_speech_db": vad_speech_db,
        "vad_attack": vad_attack,
        "vad_release": vad_release,
        "vad_min_scale": vad_min_scale,
    }
    for key, value in updates.items():
        if value is not None:
            session.config[key] = float(value)
    return {"ok": True, "config": session.config}


@app.post("/api/reset_stream")
async def reset_stream(session_id: str = Form(...)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    async with session.process_lock:
        drained = await reset_session_stream(session)
    return {
        "ok": True,
        "generation": session.generation,
        "drained": drained,
    }


@app.post("/api/audio_chunk")
async def audio_chunk(session_id: str = Form(...), audio: UploadFile = File(...)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    async with session.lock:
        index = session.next_index
        session.next_index += 1
        generation = session.generation

    suffix = Path(audio.filename or "chunk.webm").suffix or ".webm"
    audio_path = session.out_dir / f"gen{generation:03d}_chunk_{index:04d}{suffix}"
    audio_path.write_bytes(await audio.read())
    await session.queue.put({"index": index, "generation": generation, "audio_path": str(audio_path)})
    return {"queued": True, "index": index, "generation": generation}


@app.post("/api/process_recording")
async def process_recording(session_id: str = Form(...), audio: UploadFile = File(...)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    async with session.lock:
        index = session.next_index
        session.next_index += 1

    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    audio_path = session.out_dir / f"recording_{index:04d}{suffix}"
    audio_path.write_bytes(await audio.read())
    item = {
        "index": index,
        "audio_path": str(audio_path),
        "save_mp4": True,
        "save_frames": False,
    }
    async with session.process_lock:
        result = await asyncio.to_thread(process_audio_chunk, session, item)
    return result


@app.get("/api/status/{session_id}")
async def status(session_id: str, after: int = -1):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    chunks = [item for item in session.completed if item["index"] > after]
    return JSONResponse({
        "session_id": session_id,
        "generation": session.generation,
        "queued": session.queue.qsize(),
        "next_index": session.next_index,
        "chunks": chunks,
        "errors": session.errors[-5:],
    })


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>DyStream Live Chunk Demo</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; max-width: 1120px; }
    .row { display: flex; gap: 20px; align-items: flex-start; }
    .panel { flex: 1; min-width: 320px; }
    label { display: block; margin-top: 10px; font-weight: 600; }
    input, select, button { margin-top: 4px; }
    canvas { width: 100%; background: #111; }
    img { width: 160px; margin-right: 8px; border: 1px solid #ccc; }
    pre { background: #f6f6f6; padding: 12px; overflow: auto; }
  </style>
</head>
<body>
  <h1>DyStream Live Chunk Demo</h1>
  <div class="row">
    <div class="panel">
      <label>Reference image</label>
      <input id="image" type="file" accept="image/*" />

      <label>Chunk seconds</label>
      <input id="chunkSec" type="number" min="0.2" max="10" step="0.1" value="0.6" />

      <label>Guidance mode</label>
      <select id="guidanceMode">
        <option value="all_only" selected>all_only</option>
        <option value="uncond_all_2way">uncond_all_2way</option>
        <option value="full_5way">full_5way</option>
      </select>

      <label>Motion chunk frames</label>
      <input id="motionChunkFrames" type="number" min="1" max="8" step="1" value="8" />

      <label>Render batch size</label>
      <input id="renderBatchSize" type="number" min="1" max="64" step="1" value="8" />

      <label>
        <input id="saveMp4" type="checkbox" />
        Save debug MP4 chunks
      </label>

      <label>
        <input id="skipFirstPlayback" type="checkbox" checked />
        Skip first playback chunk
      </label>

      <label>Startup buffer frames</label>
      <input id="startupBufferFrames" type="number" min="0" max="64" step="1" value="8" />

      <label>Noise gate floor dB</label>
      <input id="vadFloorDb" type="number" min="-80" max="-10" step="1" value="-46" />

      <label>Speech gate dB</label>
      <input id="vadSpeechDb" type="number" min="-80" max="-10" step="1" value="-30" />

      <label>Gate attack smoothing</label>
      <input id="vadAttack" type="number" min="0" max="0.99" step="0.05" value="0.35" />

      <label>Gate release smoothing</label>
      <input id="vadRelease" type="number" min="0" max="0.99" step="0.05" value="0.85" />

      <div style="margin-top: 16px;">
        <button id="initBtn">Init Session</button>
        <button id="startBtn" disabled>Start Mic</button>
        <button id="stopBtn" disabled>Stop</button>
        <button id="recordMp4Btn" disabled>Record Fixed MP4</button>
      </div>

      <div style="margin-top: 16px;">
        <img id="refImg" />
        <img id="maskImg" />
      </div>
    </div>

    <div class="panel">
      <canvas id="canvas" width="512" height="512"></canvas>
      <video id="mp4Player" playsinline controls style="width:100%; margin-top:12px; background:#111;"></video>
      <pre id="log"></pre>
    </div>
  </div>

<script>
let sessionId = null;
let recorder = null;
let stream = null;
let pollTimer = null;
let micRunning = false;
let lastDone = -1;
let frameQueue = [];
let frameTimer = null;
let playing = false;
let targetFps = 25;

const logEl = document.getElementById('log');
function log(msg) {
  const t = new Date().toLocaleTimeString();
  logEl.textContent = `[${t}] ${msg}\n` + logEl.textContent;
}

async function initSession() {
  const image = document.getElementById('image').files[0];
  if (!image) {
    alert('Choose a reference image first.');
    return;
  }
  const form = new FormData();
  form.append('image', image);
  form.append('denoising_steps', '1');
  form.append('guidance_mode', document.getElementById('guidanceMode').value);
  form.append('motion_chunk_frames', document.getElementById('motionChunkFrames').value);
  form.append('render_batch_size', document.getElementById('renderBatchSize').value);
  form.append('cfg_audio', '0.5');
  form.append('cfg_audio_other', '0.5');
  form.append('cfg_anchor', '0.0');
  form.append('cfg_all', '1.0');
  form.append('save_mp4', document.getElementById('saveMp4').checked ? 'true' : 'false');
  form.append('skip_first_playback', document.getElementById('skipFirstPlayback').checked ? 'true' : 'false');
  form.append('vad_floor_db', document.getElementById('vadFloorDb').value);
  form.append('vad_speech_db', document.getElementById('vadSpeechDb').value);
  form.append('vad_attack', document.getElementById('vadAttack').value);
  form.append('vad_release', document.getElementById('vadRelease').value);
  form.append('vad_min_scale', '0.0');

  log('initializing session...');
  const res = await fetch('/api/init_session', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    log('init failed: ' + JSON.stringify(data));
    return;
  }
  sessionId = data.session_id;
  document.getElementById('refImg').src = data.resized_url;
  document.getElementById('maskImg').src = data.masked_url;
  document.getElementById('startBtn').disabled = false;
  document.getElementById('recordMp4Btn').disabled = false;
  log(`session ${sessionId} ready, init=${data.init_time.toFixed(3)}s`);
}

async function updateGateConfig() {
  if (!sessionId) return;
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('vad_floor_db', document.getElementById('vadFloorDb').value);
  form.append('vad_speech_db', document.getElementById('vadSpeechDb').value);
  form.append('vad_attack', document.getElementById('vadAttack').value);
  form.append('vad_release', document.getElementById('vadRelease').value);
  form.append('vad_min_scale', '0.0');
  const res = await fetch('/api/update_config', { method: 'POST', body: form });
  if (!res.ok) {
    const data = await res.json();
    log('config update failed: ' + JSON.stringify(data));
  }
}

function clearLocalPlayback() {
  frameQueue = [];
  lastDone = -1;
  playing = false;
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
}

async function resetServerStream() {
  if (!sessionId) return;
  const form = new FormData();
  form.append('session_id', sessionId);
  const res = await fetch('/api/reset_stream', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    log('reset failed: ' + JSON.stringify(data));
    return;
  }
  log(`stream reset, generation=${data.generation}, drained=${data.drained}`);
}

async function startMic() {
  if (!sessionId) return;
  clearLocalPlayback();
  await resetServerStream();
  await updateGateConfig();
  const chunkMs = Math.max(500, Number(document.getElementById('chunkSec').value) * 1000);
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micRunning = true;
  pollTimer = setInterval(pollStatus, 200);
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  log(`microphone started, chunk=${chunkMs / 1000}s`);
  recordLoop(chunkMs);
}

function pickMimeType() {
  if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) return 'audio/webm;codecs=opus';
  if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm';
  return '';
}

async function uploadAudioBlob(blob) {
  if (!micRunning) return;
  if (!blob || blob.size === 0 || !sessionId) return;
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('audio', blob, `mic_${Date.now()}.webm`);
  const res = await fetch('/api/audio_chunk', { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    log('upload failed: ' + JSON.stringify(data));
    return;
  }
  log(`uploaded audio chunk ${data.index}`);
}

async function recordBlobFromStream(inputStream, chunkMs) {
  const mimeType = pickMimeType();
  const options = mimeType ? { mimeType } : {};
  const chunks = [];
  const localRecorder = new MediaRecorder(inputStream, options);
  recorder = localRecorder;
  localRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) chunks.push(event.data);
  };
  const stopped = new Promise(resolve => {
    localRecorder.onstop = resolve;
  });
  localRecorder.start();
  await new Promise(resolve => setTimeout(resolve, chunkMs));
  if (localRecorder.state !== 'inactive') localRecorder.stop();
  await stopped;
  return new Blob(chunks, { type: mimeType || 'audio/webm' });
}

async function recordOneChunk(chunkMs) {
  const blob = await recordBlobFromStream(stream, chunkMs);
  if (!micRunning) return;
  await uploadAudioBlob(blob);
}

async function recordLoop(chunkMs) {
  while (micRunning) {
    try {
      await recordOneChunk(chunkMs);
    } catch (err) {
      log('record chunk failed: ' + err);
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
}

function stopMic() {
  micRunning = false;
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (pollTimer) clearInterval(pollTimer);
  clearLocalPlayback();
  resetServerStream();
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  log('microphone stopped');
}

async function recordFixedMp4() {
  if (!sessionId) return;
  const chunkMs = Math.max(500, Number(document.getElementById('chunkSec').value) * 1000);
  document.getElementById('recordMp4Btn').disabled = true;
  log(`recording fixed mp4 audio, ${chunkMs / 1000}s...`);
  let localStream = null;
  try {
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const blob = await recordBlobFromStream(localStream, chunkMs);
    const form = new FormData();
    form.append('session_id', sessionId);
    form.append('audio', blob, `fixed_${Date.now()}.webm`);
    log('processing fixed recording...');
    const res = await fetch('/api/process_recording', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) {
      log('fixed recording failed: ' + JSON.stringify(data));
      return;
    }
    const player = document.getElementById('mp4Player');
    player.src = data.video_url;
    player.play().catch(() => {});
    const tt = data.timings.total.toFixed(3);
    const mi = data.timings.motion_inference.toFixed(3);
    const rd = data.timings.render.toFixed(3);
    const mx = data.timings.mux.toFixed(3);
    log(`fixed mp4 ready, total=${tt}s motion=${mi}s render=${rd}s mux=${mx}s`);
  } catch (err) {
    log('fixed mp4 error: ' + err);
  } finally {
    if (localStream) localStream.getTracks().forEach(t => t.stop());
    document.getElementById('recordMp4Btn').disabled = false;
  }
}

async function pollStatus() {
  if (!sessionId) return;
  const res = await fetch(`/api/status/${sessionId}?after=${lastDone}`);
  const data = await res.json();
  for (const chunk of data.chunks) {
    lastDone = Math.max(lastDone, chunk.index);
    targetFps = chunk.fps || 25;
    enqueueFrames(chunk.frame_urls);
    const tt = chunk.timings.total.toFixed(3);
    const mi = chunk.timings.motion_inference.toFixed(3);
    const rd = chunk.timings.render.toFixed(3);
    const sf = (chunk.timings.save_frames || 0).toFixed(3);
    const mx = (chunk.timings.mux || 0).toFixed(3);
    const sc = (chunk.timings.audio_scale ?? 1.0).toFixed(2);
    const db = (chunk.timings.vad_rms_db ?? 0.0).toFixed(1);
    const skipped = chunk.skipped_playback ? ' skipped_playback=true' : '';
    log(`ready frame chunk ${chunk.index}, frames=${chunk.frame_urls.length}/${chunk.frames} total=${tt}s motion=${mi}s render=${rd}s save_frames=${sf}s mux=${mx}s gate=${sc} rms=${db}dB queue=${data.queued}${skipped}`);
  }
  if (data.errors && data.errors.length) {
    log('errors: ' + JSON.stringify(data.errors));
  }
  startFramePlayerIfNeeded();
}

function enqueueFrames(urls) {
  for (const url of urls) {
    const img = new Image();
    img.src = url;
    frameQueue.push(img);
  }
}

function startFramePlayerIfNeeded() {
  if (playing) return;
  const startupBufferFrames = Number(document.getElementById('startupBufferFrames').value) || 0;
  if (frameQueue.length < startupBufferFrames) return;
  playing = true;
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const interval = Math.max(10, Math.round(1000 / targetFps));
  frameTimer = setInterval(() => {
    if (frameQueue.length === 0) {
      playing = false;
      clearInterval(frameTimer);
      frameTimer = null;
      return;
    }
    const img = frameQueue.shift();
    if (img.complete && img.naturalWidth > 0) {
      canvas.width = img.naturalWidth || 512;
      canvas.height = img.naturalHeight || 512;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      return;
    }
    img.onload = () => {
      canvas.width = img.naturalWidth || 512;
      canvas.height = img.naturalHeight || 512;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
  }, interval);
}

document.getElementById('initBtn').onclick = initSession;
document.getElementById('startBtn').onclick = startMic;
document.getElementById('stopBtn').onclick = stopMic;
document.getElementById('recordMp4Btn').onclick = recordFixedMp4;
for (const id of ['vadFloorDb', 'vadSpeechDb', 'vadAttack', 'vadRelease']) {
  document.getElementById(id).addEventListener('change', updateGateConfig);
}
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
