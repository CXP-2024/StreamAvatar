"""
Block-wise streaming AR distillation for DyStream.

The frozen DyStream teacher still runs frame-level AR inference. The student
learns to emit K motion latents in one forward pass from frozen audio features,
recent motion history, and the reference anchor.

Usage:
  .venv/bin/python train_blockwise_distill.py \
    --config configs/distill/blockwise_stream_distill.yaml \
    --override data.max_clips=10 training.max_steps=100
"""

import argparse
import os
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))

from train_distill import LRS3AudioDataset, load_teacher, prepare_rollout_inputs


class BlockARStudent(nn.Module):
    def __init__(
        self,
        audio_dim=768,
        motion_dim=512,
        hidden_dim=512,
        block_frames=8,
        history_frames=32,
        layers=6,
        heads=8,
        dropout=0.1,
    ):
        super().__init__()
        self.block_frames = block_frames
        self.history_frames = history_frames
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim * 2, hidden_dim)
        self.anchor_proj = nn.Linear(motion_dim, hidden_dim)
        self.type_embed = nn.Embedding(2, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_frames + block_frames, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, motion_dim)

    def forward(self, past_motion, audio_self, audio_other, anchor):
        bs, hist, _ = past_motion.shape
        block = audio_self.shape[1] - hist
        if hist != self.history_frames:
            raise ValueError(f"expected {self.history_frames} history frames, got {hist}")
        if block != self.block_frames:
            raise ValueError(f"expected {self.block_frames} block frames, got {block}")

        motion_tokens = self.motion_proj(past_motion)
        future_motion_tokens = torch.zeros(bs, block, motion_tokens.shape[-1], device=past_motion.device)
        x = torch.cat([motion_tokens, future_motion_tokens], dim=1)

        audio = torch.cat([audio_self, audio_other], dim=-1)
        x = x + self.audio_proj(audio)
        x = x + self.anchor_proj(anchor).expand(-1, hist + block, -1)

        type_ids = torch.cat(
            [
                torch.zeros(hist, dtype=torch.long, device=past_motion.device),
                torch.ones(block, dtype=torch.long, device=past_motion.device),
            ],
            dim=0,
        ).unsqueeze(0)
        x = x + self.type_embed(type_ids) + self.pos_embed[:, : hist + block]
        x = self.blocks(x)
        x = self.norm(x[:, hist:])
        return self.out(x)


def extract_audio_features(teacher, audio_padded, audio_other):
    audio_list = [item.cpu().numpy() for item in audio_padded]
    inputs = teacher.audio_processor(
        audio_list, sampling_rate=16000, return_tensors="pt", padding=True
    ).to(audio_padded.device)
    pad = torch.zeros(
        inputs.input_values.shape[0],
        80,
        device=inputs.input_values.device,
        dtype=inputs.input_values.dtype,
    )
    feat_self = teacher.audio_encoder_face(torch.cat([inputs.input_values, pad], dim=-1))["high_level"]
    feat_self = F.interpolate(
        feat_self.transpose(1, 2),
        scale_factor=(teacher.cfg.pose_fps / 50),
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)

    other_list = [item.cpu().numpy() for item in audio_other]
    inputs_other = teacher.audio_processor(
        other_list, sampling_rate=16000, return_tensors="pt", padding=True
    ).to(audio_padded.device)
    pad_other = torch.zeros(
        inputs_other.input_values.shape[0],
        80,
        device=inputs_other.input_values.device,
        dtype=inputs_other.input_values.dtype,
    )
    feat_other = teacher.audio_encoder_face_other(torch.cat([inputs_other.input_values, pad_other], dim=-1))["high_level"]
    feat_other = F.interpolate(
        feat_other.transpose(1, 2),
        scale_factor=(teacher.cfg.pose_fps / 50),
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    return feat_self, feat_other


def gather_audio_window(feat, start, total_len):
    end = start + total_len
    if start < 0:
        prefix = feat[:, :1].expand(-1, -start, -1)
        body = feat[:, :end]
        return torch.cat([prefix, body], dim=1)
    if end > feat.shape[1]:
        body = feat[:, start:]
        suffix = feat[:, -1:].expand(-1, end - feat.shape[1], -1)
        return torch.cat([body, suffix], dim=1)
    return feat[:, start:end]


def blockwise_rollout(student, feat_self, feat_other, anchor, target_frames, inpaint_len, cfg):
    block = cfg.student.block_frames
    hist = cfg.student.history_frames
    past = anchor.expand(-1, hist, -1).contiguous()
    preds = []

    for start in range(0, target_frames, block):
        current_block = min(block, target_frames - start)
        feat_start = inpaint_len + start - hist
        audio_self = gather_audio_window(feat_self, feat_start, hist + block)
        audio_other = gather_audio_window(feat_other, feat_start, hist + block)
        pred = student(past, audio_self, audio_other, anchor)
        pred_current = pred[:, :current_block]
        preds.append(pred_current)
        next_past = torch.cat([past, pred_current], dim=1)[:, -hist:]
        if cfg.student.detach_between_blocks:
            next_past = next_past.detach()
        past = next_past

    return torch.cat(preds, dim=1)


def distill_loss(student_seq, teacher_seq, cfg):
    min_len = min(student_seq.shape[1], teacher_seq.shape[1])
    student_seq = student_seq[:, :min_len]
    teacher_seq = teacher_seq[:, :min_len]
    loss_motion = F.mse_loss(student_seq, teacher_seq)
    zero = student_seq.new_tensor(0.0)

    loss_vel = zero
    if min_len > 1 and cfg.loss.velocity_weight > 0:
        loss_vel = F.mse_loss(
            student_seq[:, 1:] - student_seq[:, :-1],
            teacher_seq[:, 1:] - teacher_seq[:, :-1],
        )

    loss_acc = zero
    if min_len > 2 and cfg.loss.acceleration_weight > 0:
        student_vel = student_seq[:, 1:] - student_seq[:, :-1]
        teacher_vel = teacher_seq[:, 1:] - teacher_seq[:, :-1]
        loss_acc = F.mse_loss(student_vel[:, 1:] - student_vel[:, :-1], teacher_vel[:, 1:] - teacher_vel[:, :-1])

    loss_boundary = zero
    block = cfg.student.block_frames
    if min_len > block and cfg.loss.boundary_weight > 0:
        idx = torch.arange(block, min_len, block, device=student_seq.device)
        student_jump = student_seq[:, idx] - student_seq[:, idx - 1]
        teacher_jump = teacher_seq[:, idx] - teacher_seq[:, idx - 1]
        loss_boundary = F.mse_loss(student_jump, teacher_jump)

    loss = (
        loss_motion
        + cfg.loss.velocity_weight * loss_vel
        + cfg.loss.acceleration_weight * loss_acc
        + cfg.loss.boundary_weight * loss_boundary
    )
    return loss, loss_motion, loss_vel, loss_acc, loss_boundary, min_len


def teacher_rollout(teacher, audio_padded, audio_other, motion_in, anchor, cfg, seed):
    scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler)
    torch.manual_seed(seed)
    return teacher.inference(
        audio_padded,
        audio_other=audio_other,
        cond_motion=motion_in,
        init_motion=motion_in,
        anchor_motion=anchor,
        noise_scheduler=scheduler,
        num_inference_steps=cfg.teacher.denoising_steps,
        guidance_mode=cfg.teacher.guidance_mode,
    )


def save_checkpoint(path, student, cfg, step, loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "step": step,
            "loss": float(loss),
            "mode": "blockwise_streaming_ar_distill",
        },
        path,
    )


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DyStream blockwise distillation")

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    os.environ["DYSTREAM_WAV2VEC_PATH"] = cfg.wav2vec_path

    print("=" * 72)
    print("  DyStream Block-wise Streaming AR Distillation")
    print("=" * 72)

    print("[1/4] Loading frozen teacher...")
    teacher = load_teacher(cfg).to(device)
    teacher.eval()

    print("[2/4] Building block-wise student...")
    student = BlockARStudent(
        audio_dim=768,
        motion_dim=cfg.model.vae_codebook_size,
        hidden_dim=cfg.student.hidden_dim,
        block_frames=cfg.student.block_frames,
        history_frames=cfg.student.history_frames,
        layers=cfg.student.layers,
        heads=cfg.student.heads,
        dropout=cfg.student.dropout,
    ).to(device)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable / 1e6:.2f}M")
    print(f"  Block frames: {cfg.student.block_frames}, history frames: {cfg.student.history_frames}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return step / max(cfg.training.warmup_steps, 1)
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.training.fp16)

    print("[3/4] Loading audio dataset...")
    dataset = LRS3AudioDataset(
        lrs3_root=cfg.lrs3_root,
        audio_sr=cfg.model.audio_sr,
        duration_sec=cfg.data.audio_duration_sec,
        max_clips=cfg.data.get("max_clips", None),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    print("[4/4] Training...")
    print(f"  Teacher: {cfg.teacher.guidance_mode}, {cfg.teacher.denoising_steps} step(s)")
    print(f"  Max steps: {cfg.training.max_steps}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    loss_accum = 0.0
    t_start = time.perf_counter()

    for epoch in range(cfg.training.num_epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{cfg.training.num_epochs}", dynamic_ncols=True, leave=False)
        for batch in pbar:
            if global_step >= cfg.training.max_steps:
                break

            audio = batch["audio"].to(device, non_blocking=True)
            audio_padded, audio_other, motion_in, anchor, inpaint_len = prepare_rollout_inputs(audio, cfg, device)
            seed = int(cfg.seed + global_step)

            with torch.no_grad():
                feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
                teacher_out = teacher_rollout(teacher, audio_padded, audio_other, motion_in, anchor, cfg, seed)
                teacher_seq = teacher_out[:, inpaint_len:].detach()

            with autocast(enabled=cfg.training.fp16):
                student_seq = blockwise_rollout(
                    student,
                    feat_self,
                    feat_other,
                    anchor,
                    target_frames=teacher_seq.shape[1],
                    inpaint_len=inpaint_len,
                    cfg=cfg,
                )
                loss, loss_motion, loss_vel, loss_acc, loss_boundary, matched = distill_loss(student_seq, teacher_seq, cfg)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            loss_accum += loss.item()
            pbar.set_postfix(
                step=global_step,
                loss=f"{loss.item():.5f}",
                motion=f"{loss_motion.item():.5f}",
                frames=matched,
            )

            if global_step % cfg.training.log_every == 0:
                avg_loss = loss_accum / cfg.training.log_every
                elapsed = time.perf_counter() - t_start
                tqdm.write(
                    f"[step {global_step:>6d}] "
                    f"loss={avg_loss:.6f} "
                    f"motion={loss_motion.item():.6f} "
                    f"vel={loss_vel.item():.6f} "
                    f"acc={loss_acc.item():.6f} "
                    f"boundary={loss_boundary.item():.6f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e} "
                    f"speed={global_step / max(elapsed, 1e-6):.2f} steps/s"
                )
                loss_accum = 0.0

            if global_step % cfg.training.save_every == 0:
                last_path = os.path.join(cfg.output_dir, f"blockwise_step{global_step}.pt")
                save_checkpoint(last_path, student, cfg, global_step, loss.item())
                tqdm.write(f"  saved {last_path}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = os.path.join(cfg.output_dir, "blockwise_best.pt")
                save_checkpoint(best_path, student, cfg, global_step, loss.item())

        if global_step >= cfg.training.max_steps:
            break

    final_path = os.path.join(cfg.output_dir, "blockwise_last.pt")
    save_checkpoint(final_path, student, cfg, global_step, loss.item())
    print(f"Done. final={final_path}, best_loss={best_loss:.6f}")


def apply_overrides(cfg, overrides):
    for override in overrides:
        key, val = override.split("=", 1)
        try:
            parsed = int(val)
        except ValueError:
            try:
                parsed = float(val)
            except ValueError:
                low = val.lower()
                if low in {"true", "false"}:
                    parsed = low == "true"
                elif low in {"none", "null"}:
                    parsed = None
                else:
                    parsed = val
        OmegaConf.update(cfg, key, parsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyStream block-wise streaming AR distillation")
    parser.add_argument("--config", default="configs/distill/blockwise_stream_distill.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)
    train(cfg)
