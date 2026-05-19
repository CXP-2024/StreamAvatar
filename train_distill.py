"""
DyStream AR rollout-aware DiffusionHead Step Distillation (5→1 step)

Runs the frozen teacher with 5-step AR inference and the student with 1-step
AR inference, then trains the student's DiffusionHead (and optionally
TimeEmbed) to match the teacher motion sequence. Everything else is frozen.

Usage:
  # Smoke test (10 clips, 100 steps)
  python train_distill.py --config configs/distill/step_distill.yaml \
      --override data.max_clips=10 training.max_steps=100

  # Full training
  python train_distill.py --config configs/distill/step_distill.yaml
"""

import os, sys, time, copy, glob, argparse, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
import librosa
from tqdm import tqdm

from utils import instantiate_motion_gen


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class LRS3AudioDataset(Dataset):
    def __init__(self, lrs3_root, audio_sr=16000, duration_sec=3.0, max_clips=None):
        self.audio_sr = audio_sr
        self.num_samples = int(duration_sec * audio_sr)
        self.clips = sorted(glob.glob(os.path.join(lrs3_root, "**", "*.mp4"), recursive=True))
        if max_clips:
            self.clips = self.clips[:max_clips]
        print(f"  Dataset: {len(self.clips)} clips from {lrs3_root}")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        path = self.clips[idx]
        try:
            import soundfile as sf
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.audio_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)
        except Exception:
            try:
                audio, _ = librosa.load(path, sr=self.audio_sr, mono=True)
            except Exception:
                audio = np.zeros(self.num_samples, dtype=np.float32)

        audio = audio.astype(np.float32)
        if len(audio) >= self.num_samples:
            audio = audio[:self.num_samples]
        else:
            audio = np.pad(audio, (0, self.num_samples - len(audio)))

        return {"audio": torch.from_numpy(audio).float()}


# ═══════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════

def load_teacher(cfg):
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path
    model = instantiate_motion_gen(
        module_name=cfg.model.module_name,
        class_name=cfg.model.class_name,
        cfg=cfg.model, hfstyle=False)

    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)

    if cfg.teacher.use_ema and "ema_state" in ckpt:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=cfg.model.ema_decay)
        ema.load_state_dict(ckpt["ema_state"])
        ema.copy_to(model.parameters())
        print("  Loaded EMA weights into teacher")

    del ckpt
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ═══════════════════════════════════════════════════════════════════════
# Distillation Helpers
# ═══════════════════════════════════════════════════════════════════════

def configure_student_trainable(student, train_time_embed=False):
    student.eval()
    for p in student.parameters():
        p.requires_grad = False

    student.diffusion_head.train()
    for p in student.diffusion_head.parameters():
        p.requires_grad = True

    if train_time_embed:
        student.time_embed.train()
        for p in student.time_embed.parameters():
            p.requires_grad = True
    else:
        student.time_embed.eval()


def get_trainable_params(student, train_time_embed=False):
    params = list(student.diffusion_head.parameters())
    if train_time_embed:
        params += list(student.time_embed.parameters())
    return params


def prepare_rollout_inputs(audio, cfg, device):
    bs = audio.shape[0]
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    inpaint_len = cfg.model.cbh_window_length - 2
    pad_samples = inpaint_len * hop
    audio_padded = F.pad(audio, (pad_samples, 0))
    audio_other = torch.zeros_like(audio_padded)
    total_frames = audio_padded.shape[1] // hop
    anchor_std = cfg.data.get("anchor_std", 0.35)
    anchor = torch.randn(bs, 1, cfg.model.vae_codebook_size, device=device) * anchor_std
    motion_in = anchor.repeat(1, total_frames, 1)
    return audio_padded, audio_other, motion_in, anchor, inpaint_len


def rollout(model, audio_padded, audio_other, motion_in, anchor, scheduler_cfg, steps, seed):
    scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_cfg)
    torch.manual_seed(seed)
    return model.inference(
        audio_padded,
        audio_other=audio_other,
        cond_motion=motion_in,
        init_motion=motion_in,
        anchor_motion=anchor,
        noise_scheduler=scheduler,
        num_inference_steps=steps,
    )


def sequence_distill_loss(student_seq, teacher_seq, velocity_weight=0.1):
    min_len = min(student_seq.shape[1], teacher_seq.shape[1])
    student_seq = student_seq[:, :min_len]
    teacher_seq = teacher_seq[:, :min_len]
    loss_motion = F.mse_loss(student_seq, teacher_seq)
    loss_vel = student_seq.new_tensor(0.0)
    if min_len > 1 and velocity_weight > 0:
        loss_vel = F.mse_loss(
            student_seq[:, 1:] - student_seq[:, :-1],
            teacher_seq[:, 1:] - teacher_seq[:, :-1],
        )
    loss = loss_motion + velocity_weight * loss_vel
    return loss, loss_motion, loss_vel, min_len


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train(cfg):
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    print("=" * 60)
    print("  DyStream AR Rollout-Aware DiffusionHead Distillation")
    print("=" * 60)

    # Load teacher
    print("\n[1/4] Loading teacher model...")
    teacher = load_teacher(cfg).to(device)

    # Load a full student so its AR rollout path exactly matches inference.
    print("[2/4] Loading student model...")
    student = load_teacher(cfg).to(device)
    train_time_embed = cfg.training.get("train_time_embed", False)
    configure_student_trainable(student, train_time_embed=train_time_embed)

    trainable_params = get_trainable_params(student, train_time_embed=train_time_embed)
    num_params = sum(p.numel() for p in trainable_params)
    trainable_names = "diffusion_head + time_embed" if train_time_embed else "diffusion_head"
    print(f"  Trainable params: {num_params/1e6:.1f}M ({trainable_names})")

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay)

    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return step / max(cfg.training.warmup_steps, 1)
        return 1.0
    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.training.fp16)

    # Dataset
    print("[3/4] Loading dataset...")
    dataset = LRS3AudioDataset(
        lrs3_root=cfg.lrs3_root, audio_sr=cfg.model.audio_sr,
        duration_sec=cfg.data.audio_duration_sec,
        max_clips=cfg.data.get("max_clips", None))
    dataloader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, drop_last=True, pin_memory=True)

    # Training loop
    print(f"[4/4] Training: max_steps={cfg.training.max_steps}, bs={cfg.training.batch_size}")
    print(f"  Teacher: {cfg.teacher.denoising_steps}-step AR rollout → Student: 1-step AR rollout")
    print()

    os.makedirs(cfg.output_dir, exist_ok=True)
    global_step = 0
    best_loss = float('inf')
    loss_accum = 0.0
    t_start = time.perf_counter()

    for epoch in range(cfg.training.num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
                    leave=False, dynamic_ncols=True)
        for batch in pbar:
            if global_step >= cfg.training.max_steps:
                break

            audio = batch["audio"].to(device)
            audio_padded, audio_other, motion_in, anchor, inpaint_len = prepare_rollout_inputs(audio, cfg, device)
            rollout_seed = int(cfg.seed + global_step)

            # Teacher: complete 5-step AR rollout target.
            with torch.no_grad():
                teacher_out = rollout(
                    teacher,
                    audio_padded,
                    audio_other,
                    motion_in,
                    anchor,
                    cfg.noise_scheduler,
                    cfg.teacher.denoising_steps,
                    rollout_seed,
                )
                teacher_seq = teacher_out[:, inpaint_len:].detach()

            with autocast(enabled=cfg.training.fp16):
                student_out = rollout(
                    student,
                    audio_padded,
                    audio_other,
                    motion_in,
                    anchor,
                    cfg.noise_scheduler,
                    1,
                    rollout_seed,
                )
                student_seq = student_out[:, inpaint_len:]
                loss, loss_motion, loss_vel, matched_frames = sequence_distill_loss(
                    student_seq,
                    teacher_seq,
                    velocity_weight=cfg.loss.get("velocity_weight", 0.1),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable_params, cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler_lr.step()

            global_step += 1
            loss_accum += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.5f}",
                motion=f"{loss_motion.item():.5f}",
                vel=f"{loss_vel.item():.5f}",
                frames=matched_frames,
                step=global_step,
            )

            if global_step % cfg.training.log_every == 0:
                avg_loss = loss_accum / cfg.training.log_every
                elapsed = time.perf_counter() - t_start
                steps_per_sec = global_step / elapsed
                tqdm.write(f"  [step {global_step:>6d}] loss={avg_loss:.6f} "
                           f"last_motion={loss_motion.item():.6f} "
                           f"last_vel={loss_vel.item():.6f} "
                           f"frames={matched_frames} "
                           f"lr={scheduler_lr.get_last_lr()[0]:.2e} "
                           f"speed={steps_per_sec:.1f} steps/s")
                loss_accum = 0.0

            if global_step % cfg.training.save_every == 0:
                save_path = os.path.join(cfg.output_dir, f"distilled_step{global_step}.pt")
                torch.save({
                    "diffusion_head": student.diffusion_head.state_dict(),
                    "time_embed": student.time_embed.state_dict(),
                    "step": global_step, "loss": loss.item(),
                    "loss_motion": loss_motion.item(),
                    "loss_velocity": loss_vel.item(),
                    "mode": "ar_rollout",
                }, save_path)
                tqdm.write(f"  → Saved: {save_path}")

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = os.path.join(cfg.output_dir, "distilled_best.pt")
                    torch.save({
                        "diffusion_head": student.diffusion_head.state_dict(),
                        "time_embed": student.time_embed.state_dict(),
                        "step": global_step, "loss": loss.item(),
                        "loss_motion": loss_motion.item(),
                        "loss_velocity": loss_vel.item(),
                        "mode": "ar_rollout",
                    }, best_path)

        if global_step >= cfg.training.max_steps:
            break

    # Final save
    final_path = os.path.join(cfg.output_dir, "distilled_final.pt")
    torch.save({
        "diffusion_head": student.diffusion_head.state_dict(),
        "time_embed": student.time_embed.state_dict(),
        "step": global_step, "loss": loss.item(),
        "loss_motion": loss_motion.item(),
        "loss_velocity": loss_vel.item(),
        "mode": "ar_rollout",
    }, final_path)

    total_time = time.perf_counter() - t_start
    print(f"\n  Training complete: {global_step} steps in {total_time:.0f}s")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final checkpoint: {final_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyStream DiffusionHead Step Distillation")
    parser.add_argument("--config", type=str, default="configs/distill/step_distill.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    for override in args.override:
        key, val = override.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() == "null" or val.lower() == "none":
                    val = None
        OmegaConf.update(cfg, key, val)

    train(cfg)
