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
from stream_app import latents_to_video_frames_batched, preprocess_reference_cached, stamp


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


sessions = {}
app = FastAPI()
app.mount("/sessions", StaticFiles(directory=str(SESSION_ROOT)), name="sessions")


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_once():
    base.load_dystream_model()
    base.load_visualization_model()


def process_audio_chunk(session: StreamSession, item: dict):
    timings = {}
    t_total = stamp()

    cfg = base._dystream_cfg
    model = base._dystream_model
    audio_sr = int(OmegaConf.select(cfg.config, "model.audio_sr", default=16000))
    pose_fps = int(OmegaConf.select(cfg.config, "model.pose_fps", default=25))
    hop = int(audio_sr / pose_fps)

    t0 = stamp()
    audio_np, _ = librosa.load(item["audio_path"], sr=audio_sr)
    prefix_frames = model.inpainting_length
    audio_np = np.concatenate([np.zeros(prefix_frames * hop, dtype=np.float32), audio_np.astype(np.float32)])
    audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(base.DEVICE)
    audio_other = torch.zeros_like(audio)
    timings["audio_prepare"] = stamp() - t0

    t0 = stamp()
    motion_latent = session.motion_latent_cpu.to(base.DEVICE)
    if motion_latent.dim() == 1:
        motion_latent = motion_latent.unsqueeze(0)
    if motion_latent.dim() == 2:
        motion_latent = motion_latent.unsqueeze(0)
    total_frames = audio.shape[1] // hop
    motion_in = motion_latent[:, 0:1, :].repeat(1, total_frames, 1)

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

    profile = {}
    with torch.no_grad(), ema_ctx:
        motion_pred = model.inference(
            audio,
            audio_other=audio_other,
            init_motion=motion_in,
            cond_motion=motion_in,
            anchor_motion=motion_latent[:, 0:1, :],
            noise_scheduler=base._noise_scheduler,
            num_inference_steps=int(session.config["denoising_steps"]),
            profile=profile,
            guidance_mode=session.config["guidance_mode"],
            stream_stride=int(session.config["motion_chunk_frames"]),
        )
    motion_pred = motion_pred[:, prefix_frames:]
    timings["motion_inference"] = stamp() - t0
    timings["motion_gpt"] = profile.get("gpt", 0.0)
    timings["motion_fm"] = profile.get("fm", 0.0)
    timings["motion_audio_encoder"] = profile.get("audio_encoder", 0.0)

    t0 = stamp()
    frames = latents_to_video_frames_batched(
        motion_pred,
        session.resized_pil,
        render_batch_size=int(session.config["render_batch_size"]),
    )
    timings["render"] = stamp() - t0

    frame_urls = []
    if item.get("save_frames", True):
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
        "timings": timings,
    }


async def session_worker(session: StreamSession):
    while True:
        item = await session.queue.get()
        try:
            async with session.process_lock:
                result = await asyncio.to_thread(process_audio_chunk, session, item)
            session.completed.append(result)
            with open(session.out_dir / "manifest.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as exc:
            error = {"index": item.get("index"), "error": repr(exc)}
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


@app.post("/api/audio_chunk")
async def audio_chunk(session_id: str = Form(...), audio: UploadFile = File(...)):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session")

    async with session.lock:
        index = session.next_index
        session.next_index += 1

    suffix = Path(audio.filename or "chunk.webm").suffix or ".webm"
    audio_path = session.out_dir / f"chunk_{index:04d}{suffix}"
    audio_path.write_bytes(await audio.read())
    await session.queue.put({"index": index, "audio_path": str(audio_path)})
    return {"queued": True, "index": index}


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
      <input id="chunkSec" type="number" min="1" max="10" step="0.5" value="3" />

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

async function startMic() {
  if (!sessionId) return;
  const chunkMs = Math.max(500, Number(document.getElementById('chunkSec').value) * 1000);
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micRunning = true;
  pollTimer = setInterval(pollStatus, 1000);
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
    frameQueue.push(...chunk.frame_urls);
    const tt = chunk.timings.total.toFixed(3);
    const mi = chunk.timings.motion_inference.toFixed(3);
    const rd = chunk.timings.render.toFixed(3);
    const sf = (chunk.timings.save_frames || 0).toFixed(3);
    const mx = (chunk.timings.mux || 0).toFixed(3);
    log(`ready frame chunk ${chunk.index}, frames=${chunk.frame_urls.length} total=${tt}s motion=${mi}s render=${rd}s save_frames=${sf}s mux=${mx}s queue=${data.queued}`);
  }
  if (data.errors && data.errors.length) {
    log('errors: ' + JSON.stringify(data.errors));
  }
  startFramePlayerIfNeeded();
}

function startFramePlayerIfNeeded() {
  if (playing) return;
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
    const url = frameQueue.shift();
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth || 512;
      canvas.height = img.naturalHeight || 512;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = url;
  }, interval);
}

document.getElementById('initBtn').onclick = initSession;
document.getElementById('startBtn').onclick = startMic;
document.getElementById('stopBtn').onclick = stopMic;
document.getElementById('recordMp4Btn').onclick = recordFixedMp4;
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
