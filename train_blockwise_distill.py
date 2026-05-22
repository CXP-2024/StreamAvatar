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
import glob
import json
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
from torch.utils.data import DataLoader, Dataset
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


class CrossAttentionResidualBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.audio_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.anchor_norm = nn.LayerNorm(hidden_dim)
        self.anchor_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, audio_memory, anchor_memory, self_mask=None):
        x_norm = self.self_norm(x)
        y, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=self_mask, need_weights=False)
        x = x + y
        y, _ = self.audio_attn(self.audio_norm(x), audio_memory, audio_memory, need_weights=False)
        x = x + y
        y, _ = self.anchor_attn(self.anchor_norm(x), anchor_memory, anchor_memory, need_weights=False)
        x = x + y
        return x + self.ffn(self.ffn_norm(x))


class BlockCrossAttnResidualStudent(nn.Module):
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
        self.future_query = nn.Parameter(torch.zeros(1, block_frames, hidden_dim))
        self.audio_self_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_other_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.anchor_proj = nn.Linear(motion_dim, hidden_dim)
        self.type_embed = nn.Embedding(2, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_frames + block_frames, hidden_dim))
        self.blocks = nn.ModuleList([
            CrossAttentionResidualBlock(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, motion_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.out.bias)

    def _prefix_mask(self, hist, block, device):
        mask = torch.zeros(hist + block, hist + block, dtype=torch.bool, device=device)
        mask[:hist, hist:] = True
        return mask

    def forward(self, past_motion, audio_self, audio_other, anchor):
        bs, hist, _ = past_motion.shape
        block = audio_self.shape[1] - hist
        if hist != self.history_frames:
            raise ValueError(f"expected {self.history_frames} history frames, got {hist}")
        if block != self.block_frames:
            raise ValueError(f"expected {self.block_frames} block frames, got {block}")

        motion_tokens = self.motion_proj(past_motion)
        future_tokens = self.future_query.expand(bs, -1, -1)
        x = torch.cat([motion_tokens, future_tokens], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros(hist, dtype=torch.long, device=past_motion.device),
                torch.ones(block, dtype=torch.long, device=past_motion.device),
            ],
            dim=0,
        ).unsqueeze(0)
        x = x + self.type_embed(type_ids) + self.pos_embed[:, : hist + block]

        audio_memory = self.audio_fusion(torch.cat([
            self.audio_self_proj(audio_self),
            self.audio_other_proj(audio_other),
        ], dim=-1))
        anchor_memory = self.anchor_proj(anchor)
        self_mask = self._prefix_mask(hist, block, past_motion.device)

        for layer in self.blocks:
            x = layer(x, audio_memory, anchor_memory, self_mask=self_mask)

        delta = self.out(self.norm(x[:, hist:]))
        base = past_motion[:, -1:, :].expand(-1, block, -1)
        return base + delta


class BlockCrossFMStudent(nn.Module):
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
        max_timestep=1000,
    ):
        super().__init__()
        self.block_frames = block_frames
        self.history_frames = history_frames
        self.max_timestep = float(max_timestep)
        self.motion_proj = nn.Linear(motion_dim, hidden_dim)
        self.noisy_future_proj = nn.Linear(motion_dim, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.audio_self_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_other_proj = nn.Linear(audio_dim, hidden_dim)
        self.audio_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.anchor_proj = nn.Linear(motion_dim, hidden_dim)
        self.type_embed = nn.Embedding(2, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_frames + block_frames, hidden_dim))
        self.blocks = nn.ModuleList([
            CrossAttentionResidualBlock(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, motion_dim)
        nn.init.normal_(self.out.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.out.bias)

    def _prefix_mask(self, hist, block, device):
        mask = torch.zeros(hist + block, hist + block, dtype=torch.bool, device=device)
        mask[:hist, hist:] = True
        return mask

    def forward(self, past_motion, audio_self, audio_other, anchor, noisy_future, timestep):
        bs, hist, _ = past_motion.shape
        block = audio_self.shape[1] - hist
        if hist != self.history_frames:
            raise ValueError(f"expected {self.history_frames} history frames, got {hist}")
        if block != self.block_frames:
            raise ValueError(f"expected {self.block_frames} block frames, got {block}")
        if noisy_future.shape[1] != block:
            raise ValueError(f"expected {block} noisy future frames, got {noisy_future.shape[1]}")

        motion_tokens = self.motion_proj(past_motion)
        future_tokens = self.noisy_future_proj(noisy_future)
        t = timestep.to(device=past_motion.device, dtype=past_motion.dtype).view(bs, 1, 1)
        future_tokens = future_tokens + self.time_proj(t / max(self.max_timestep, 1.0))
        x = torch.cat([motion_tokens, future_tokens], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros(hist, dtype=torch.long, device=past_motion.device),
                torch.ones(block, dtype=torch.long, device=past_motion.device),
            ],
            dim=0,
        ).unsqueeze(0)
        x = x + self.type_embed(type_ids) + self.pos_embed[:, : hist + block]

        audio_memory = self.audio_fusion(torch.cat([
            self.audio_self_proj(audio_self),
            self.audio_other_proj(audio_other),
        ], dim=-1))
        anchor_memory = self.anchor_proj(anchor)
        self_mask = self._prefix_mask(hist, block, past_motion.device)

        for layer in self.blocks:
            x = layer(x, audio_memory, anchor_memory, self_mask=self_mask)

        return self.out(self.norm(x[:, hist:]))


def build_blockwise_student(cfg):
    architecture = cfg.student.get("architecture", "additive")
    kwargs = dict(
        audio_dim=768,
        motion_dim=cfg.model.vae_codebook_size,
        hidden_dim=cfg.student.hidden_dim,
        block_frames=cfg.student.block_frames,
        history_frames=cfg.student.history_frames,
        layers=cfg.student.layers,
        heads=cfg.student.heads,
        dropout=cfg.student.dropout,
    )
    if architecture == "additive":
        return BlockARStudent(**kwargs)
    if architecture == "cross_attn_residual":
        return BlockCrossAttnResidualStudent(**kwargs)
    if architecture == "cross_fm":
        return BlockCrossFMStudent(
            **kwargs,
            max_timestep=cfg.noise_scheduler.get("num_train_timesteps", 1000),
        )
    raise ValueError(f"Unsupported student architecture: {architecture}")


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


def blockwise_rollout(student, feat_self, feat_other, anchor, target_frames, inpaint_len, cfg, return_residual=False):
    block = cfg.student.block_frames
    hist = cfg.student.history_frames
    past = anchor.expand(-1, hist, -1).contiguous()
    preds = []
    bases = []
    deltas = []

    detach_every_blocks = cfg.student.get("detach_every_blocks", None)
    if detach_every_blocks is not None:
        detach_every_blocks = int(detach_every_blocks)

    for block_idx, start in enumerate(range(0, target_frames, block)):
        current_block = min(block, target_frames - start)
        feat_start = inpaint_len + start - hist
        audio_self = gather_audio_window(feat_self, feat_start, hist + block)
        audio_other = gather_audio_window(feat_other, feat_start, hist + block)
        pred = student(past, audio_self, audio_other, anchor)
        base = past[:, -1:, :].expand(-1, block, -1)
        pred_current = pred[:, :current_block]
        preds.append(pred_current)
        if return_residual:
            base_current = base[:, :current_block]
            bases.append(base_current)
            deltas.append(pred_current - base_current)
        next_past = torch.cat([past, pred_current], dim=1)[:, -hist:]
        if detach_every_blocks is None:
            should_detach = bool(cfg.student.detach_between_blocks)
        else:
            should_detach = detach_every_blocks > 0 and (block_idx + 1) % detach_every_blocks == 0
        if should_detach:
            next_past = next_past.detach()
        past = next_past

    pred_seq = torch.cat(preds, dim=1)
    if return_residual:
        return pred_seq, torch.cat(bases, dim=1), torch.cat(deltas, dim=1)
    return pred_seq


def sample_flow_noisy(clean_motion, noise_scheduler):
    bs = clean_motion.shape[0]
    device = clean_motion.device
    indices = torch.randint(0, len(noise_scheduler.timesteps), (bs,), device=noise_scheduler.timesteps.device)
    timesteps = noise_scheduler.timesteps[indices].to(device)
    noise = torch.randn_like(clean_motion)
    noisy_motion = noise_scheduler.scale_noise(sample=clean_motion, timestep=timesteps, noise=noise)
    return noisy_motion, timesteps


def blockwise_fm_rollout(
    student,
    feat_self,
    feat_other,
    anchor,
    target_frames,
    inpaint_len,
    cfg,
    noise_scheduler,
    teacher_seq=None,
    seed=None,
    return_residual=False,
):
    block = cfg.student.block_frames
    hist = cfg.student.history_frames
    past = anchor.expand(-1, hist, -1).contiguous()
    preds = []
    bases = []
    deltas = []

    detach_every_blocks = cfg.student.get("detach_every_blocks", None)
    if detach_every_blocks is not None:
        detach_every_blocks = int(detach_every_blocks)

    if seed is not None:
        torch.manual_seed(int(seed))

    infer_timestep = noise_scheduler.timesteps[0].to(anchor.device)
    for block_idx, start in enumerate(range(0, target_frames, block)):
        current_block = min(block, target_frames - start)
        feat_start = inpaint_len + start - hist
        audio_self = gather_audio_window(feat_self, feat_start, hist + block)
        audio_other = gather_audio_window(feat_other, feat_start, hist + block)

        if teacher_seq is not None:
            clean_block = teacher_seq[:, start : start + block]
            if clean_block.shape[1] < block:
                clean_block = torch.cat(
                    [clean_block, clean_block[:, -1:].expand(-1, block - clean_block.shape[1], -1)],
                    dim=1,
                )
            noisy_future, timesteps = sample_flow_noisy(clean_block, noise_scheduler)
        else:
            noisy_future = torch.randn(
                past.shape[0],
                block,
                past.shape[-1],
                device=past.device,
                dtype=past.dtype,
            )
            timesteps = infer_timestep.expand(past.shape[0])

        pred = student(past, audio_self, audio_other, anchor, noisy_future, timesteps)
        base = past[:, -1:, :].expand(-1, block, -1)
        pred_current = pred[:, :current_block]
        preds.append(pred_current)
        if return_residual:
            base_current = base[:, :current_block]
            bases.append(base_current)
            deltas.append(pred_current - base_current)

        next_past = torch.cat([past, pred_current], dim=1)[:, -hist:]
        if detach_every_blocks is None:
            should_detach = bool(cfg.student.detach_between_blocks)
        else:
            should_detach = detach_every_blocks > 0 and (block_idx + 1) % detach_every_blocks == 0
        if should_detach:
            next_past = next_past.detach()
        past = next_past

    pred_seq = torch.cat(preds, dim=1)
    if return_residual:
        return pred_seq, torch.cat(bases, dim=1), torch.cat(deltas, dim=1)
    return pred_seq


def distill_loss(student_seq, teacher_seq, cfg, base_seq=None, delta_seq=None):
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

    loss_delta = zero
    delta_weight = cfg.loss.get("delta_weight", 0.0)
    if delta_weight > 0 and base_seq is not None and delta_seq is not None:
        base_seq = base_seq[:, :min_len]
        delta_seq = delta_seq[:, :min_len]
        target_delta = teacher_seq - base_seq
        loss_delta = F.mse_loss(delta_seq, target_delta)

    loss = (
        loss_motion
        + cfg.loss.velocity_weight * loss_vel
        + cfg.loss.acceleration_weight * loss_acc
        + cfg.loss.boundary_weight * loss_boundary
        + delta_weight * loss_delta
    )
    return loss, loss_motion, loss_vel, loss_acc, loss_boundary, loss_delta, min_len


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


def save_checkpoint(
    path,
    student,
    cfg,
    step,
    loss,
    optimizer=None,
    lr_scheduler=None,
    scaler=None,
    best_loss=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "student": student.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
        "loss": float(loss),
        "mode": "blockwise_streaming_ar_distill",
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if best_loss is not None:
        checkpoint["best_loss"] = float(best_loss)
    torch.save(checkpoint, path)


def read_last_metrics_step(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return 0
    last_step = 0
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = item.get("step")
            if isinstance(step, int):
                last_step = max(last_step, step)
    return last_step


def load_resume_checkpoint(path, student, optimizer, lr_scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    student.load_state_dict(checkpoint["student"], strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return {
        "step": int(checkpoint.get("step", 0)),
        "loss": float(checkpoint.get("loss", float("nan"))),
        "best_loss": float(checkpoint.get("best_loss", checkpoint.get("loss", float("inf")))),
        "has_optimizer": "optimizer" in checkpoint,
        "has_lr_scheduler": "lr_scheduler" in checkpoint,
        "has_scaler": "scaler" in checkpoint,
    }


def resolve_resume_path(resume_from, output_dir):
    if not resume_from:
        return None
    if str(resume_from).lower() != "auto":
        return resume_from

    best_path = None
    best_step = -1
    for path in glob.glob(os.path.join(output_dir, "blockwise*.pt")):
        try:
            checkpoint = torch.load(path, map_location="cpu")
            step = int(checkpoint.get("step", -1))
        except Exception:
            continue
        if step > best_step:
            best_step = step
            best_path = path
    if best_path is None:
        raise FileNotFoundError(f"no blockwise*.pt checkpoint found in {output_dir}")
    return best_path


def set_scheduler_to_step(lr_scheduler, optimizer, step):
    lr_scheduler.last_epoch = step
    lr_scheduler._step_count = step + 1
    last_lrs = []
    for group, base_lr, lr_lambda in zip(optimizer.param_groups, lr_scheduler.base_lrs, lr_scheduler.lr_lambdas):
        lr = base_lr * lr_lambda(step)
        group["lr"] = lr
        last_lrs.append(lr)
    lr_scheduler._last_lr = last_lrs


def resolve_lrs3_roots(cfg):
    mode = cfg.data.get("split_mode", "trainval")
    if mode == "trainval":
        return [cfg.get("lrs3_trainval_root", cfg.get("lrs3_root"))]
    if mode == "pretrain":
        return [cfg.lrs3_pretrain_root]
    if mode == "trainval_pretrain":
        return [cfg.get("lrs3_trainval_root", cfg.get("lrs3_root")), cfg.lrs3_pretrain_root]
    raise ValueError(
        "data.split_mode must be one of: trainval, pretrain, trainval_pretrain"
    )


class ManifestMotionDataset(Dataset):
    """Load preprocessed DyStream motion latents and aligned audio segments."""

    def __init__(
        self,
        manifest_path,
        audio_sr=16000,
        pose_fps=25,
        duration_sec=3.0,
        max_clips=None,
    ):
        self.manifest_path = os.path.abspath(manifest_path)
        self.audio_sr = int(audio_sr)
        self.pose_fps = int(pose_fps)
        self.target_frames = max(1, int(round(float(duration_sec) * self.pose_fps)))
        self.hop = self.audio_sr // self.pose_fps
        self.audio_samples = self.target_frames * self.hop

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.items = [
            item for item in items
            if item.get("status") in {"ok", "cached"} and item.get("cache_path")
        ]
        if max_clips:
            self.items = self.items[: int(max_clips)]
        if not self.items:
            raise ValueError(f"no usable cache entries in manifest: {manifest_path}")

        print(
            f"  Motion cache dataset: {len(self.items)} clips from {manifest_path}, "
            f"{self.target_frames} frames/segment"
        )

    def __len__(self):
        return len(self.items)

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        cwd_path = os.path.abspath(path)
        if os.path.exists(cwd_path):
            return cwd_path
        return os.path.join(os.path.dirname(self.manifest_path), path)

    def __getitem__(self, idx):
        item = self.items[idx]
        cache = torch.load(self._resolve_path(item["cache_path"]), map_location="cpu")
        motion = cache["motion_latent"].float()
        audio = cache["audio"].float()

        if int(cache.get("audio_sr", self.audio_sr)) != self.audio_sr:
            raise ValueError(
                f"cache audio_sr={cache.get('audio_sr')} does not match config audio_sr={self.audio_sr}"
            )
        if int(round(float(cache.get("pose_fps", self.pose_fps)))) != self.pose_fps:
            raise ValueError(
                f"cache pose_fps={cache.get('pose_fps')} does not match config pose_fps={self.pose_fps}"
            )

        max_start = max(0, motion.shape[0] - self.target_frames)
        start = torch.randint(0, max_start + 1, ()).item() if max_start > 0 else 0
        end = start + self.target_frames
        motion_seg = motion[start:end]
        if motion_seg.shape[0] < self.target_frames:
            motion_seg = torch.cat(
                [motion_seg, motion_seg[-1:].expand(self.target_frames - motion_seg.shape[0], -1)],
                dim=0,
            )

        audio_start = start * self.hop
        audio_end = audio_start + self.audio_samples
        audio_seg = audio[audio_start:audio_end]
        if audio_seg.numel() < self.audio_samples:
            audio_seg = F.pad(audio_seg, (0, self.audio_samples - audio_seg.numel()))

        return {
            "audio": audio_seg.float(),
            "motion_latent": motion_seg.float(),
            "video_id": cache.get("video_id", item.get("video_id", "")),
            "start_frame": int(start),
        }


def prepare_cached_rollout_inputs(audio, motion_latent, cfg, device):
    hop = int(cfg.model.audio_sr / cfg.model.pose_fps)
    inpaint_len = cfg.model.cbh_window_length - 2
    pad_samples = inpaint_len * hop
    audio_padded = F.pad(audio, (pad_samples, 0))
    audio_other = torch.zeros_like(audio_padded)
    anchor = motion_latent[:, :1, :].to(device)
    total_frames = audio_padded.shape[1] // hop
    motion_in = anchor.repeat(1, total_frames, 1)
    return audio_padded, audio_other, motion_in, anchor, inpaint_len


def compute_student_loss(student, teacher, batch, cfg, device, target_source, noise_scheduler, seed):
    audio = batch["audio"].to(device, non_blocking=True)

    with torch.no_grad():
        if target_source == "cache":
            target_seq = batch["motion_latent"].to(device, non_blocking=True)
            audio_padded, audio_other, motion_in, anchor, inpaint_len = prepare_cached_rollout_inputs(
                audio,
                target_seq,
                cfg,
                device,
            )
        else:
            audio_padded, audio_other, motion_in, anchor, inpaint_len = prepare_rollout_inputs(audio, cfg, device)

        feat_self, feat_other = extract_audio_features(teacher, audio_padded, audio_other)
        if target_source == "cache":
            teacher_seq = target_seq.detach()
        else:
            teacher_out = teacher_rollout(teacher, audio_padded, audio_other, motion_in, anchor, cfg, seed)
            teacher_seq = teacher_out[:, inpaint_len:].detach()

    with autocast(enabled=cfg.training.fp16):
        return_residual = cfg.loss.get("delta_weight", 0.0) > 0
        if cfg.student.get("architecture", "additive") == "cross_fm":
            rollout_out = blockwise_fm_rollout(
                student,
                feat_self,
                feat_other,
                anchor,
                target_frames=teacher_seq.shape[1],
                inpaint_len=inpaint_len,
                cfg=cfg,
                noise_scheduler=noise_scheduler,
                teacher_seq=teacher_seq,
                seed=seed,
                return_residual=return_residual,
            )
        else:
            rollout_out = blockwise_rollout(
                student,
                feat_self,
                feat_other,
                anchor,
                target_frames=teacher_seq.shape[1],
                inpaint_len=inpaint_len,
                cfg=cfg,
                return_residual=return_residual,
            )
        if return_residual:
            student_seq, base_seq, delta_seq = rollout_out
        else:
            student_seq = rollout_out
            base_seq = None
            delta_seq = None
        loss, loss_motion, loss_vel, loss_acc, loss_boundary, loss_delta, matched = distill_loss(
            student_seq,
            teacher_seq,
            cfg,
            base_seq=base_seq,
            delta_seq=delta_seq,
        )

    metrics = {
        "loss": float(loss.item()),
        "loss_motion": float(loss_motion.item()),
        "loss_velocity": float(loss_vel.item()),
        "loss_acceleration": float(loss_acc.item()),
        "loss_boundary": float(loss_boundary.item()),
        "loss_delta": float(loss_delta.item()),
        "matched_frames": int(matched),
    }
    return loss, metrics


def build_dataset(cfg, target_source, manifest_path=None, max_clips=None):
    if target_source == "cache":
        return ManifestMotionDataset(
            manifest_path=manifest_path or cfg.data.manifest_path,
            audio_sr=cfg.model.audio_sr,
            pose_fps=cfg.model.pose_fps,
            duration_sec=cfg.data.audio_duration_sec,
            max_clips=max_clips if max_clips is not None else cfg.data.get("max_clips", None),
        )
    if target_source == "teacher":
        lrs3_roots = resolve_lrs3_roots(cfg)
        print(f"  Split mode: {cfg.data.get('split_mode', 'trainval')}")
        return LRS3AudioDataset(
            lrs3_root=lrs3_roots,
            audio_sr=cfg.model.audio_sr,
            duration_sec=cfg.data.audio_duration_sec,
            max_clips=max_clips if max_clips is not None else cfg.data.get("max_clips", None),
        )
    raise ValueError("data.target_source must be one of: teacher, cache")


def evaluate_student(student, teacher, dataloader, cfg, device, target_source, noise_scheduler, max_batches, seed):
    if dataloader is None:
        return None
    was_training = student.training
    student.eval()
    totals = {
        "loss": 0.0,
        "loss_motion": 0.0,
        "loss_velocity": 0.0,
        "loss_acceleration": 0.0,
        "loss_boundary": 0.0,
        "loss_delta": 0.0,
        "matched_frames": 0.0,
    }
    batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            _, metrics = compute_student_loss(
                student,
                teacher,
                batch,
                cfg,
                device,
                target_source,
                noise_scheduler,
                seed + batch_idx,
            )
            for key in totals:
                totals[key] += metrics[key]
            batches += 1
    if was_training:
        student.train()
    if batches == 0:
        return None
    return {key: value / batches for key, value in totals.items()} | {"batches": batches}


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
    student = build_blockwise_student(cfg).to(device)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable / 1e6:.2f}M")
    print(f"  Architecture: {cfg.student.get('architecture', 'additive')}")
    print(f"  Block frames: {cfg.student.block_frames}, history frames: {cfg.student.history_frames}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    def lr_lambda(step):
        if step < cfg.training.warmup_steps:
            return step / max(cfg.training.warmup_steps, 1)
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.training.fp16)
    train_flow_scheduler = FlowMatchEulerDiscreteScheduler(**cfg.noise_scheduler)

    os.makedirs(cfg.output_dir, exist_ok=True)
    metrics_path = cfg.training.get("resume_metrics", None) or os.path.join(cfg.output_dir, "metrics.jsonl")
    resume_from = resolve_resume_path(cfg.training.get("resume_from", None), cfg.output_dir)
    global_step = 0
    best_loss = float("inf")
    resume_info = None

    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")
        print(f"  Resuming from: {resume_from}")
        resume_info = load_resume_checkpoint(resume_from, student, optimizer, lr_scheduler, scaler, device)
        global_step = resume_info["step"]
        best_loss = resume_info["best_loss"]
        if not resume_info["has_lr_scheduler"]:
            set_scheduler_to_step(lr_scheduler, optimizer, global_step)
        last_metrics_step = read_last_metrics_step(metrics_path)
        if last_metrics_step > global_step:
            print(
                f"  Note: metrics already reached step {last_metrics_step}, "
                f"but checkpoint is step {global_step}; continuing from checkpoint weights."
            )

    print("[3/4] Loading audio dataset...")
    target_source = cfg.data.get("target_source", "teacher")
    print(f"  Target source: {target_source}")
    dataset = build_dataset(cfg, target_source)

    drop_last = bool(cfg.training.get("drop_last", target_source == "teacher"))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )

    val_loader = None
    val_summary_path = os.path.join(cfg.output_dir, "eval_summary.json")
    eval_history = []
    best_val_loss = float("inf")
    validation_enabled = bool(cfg.get("validation", {}).get("enabled", False))
    if validation_enabled:
        if target_source != "cache":
            raise ValueError("validation is currently supported for cache target_source only")
        val_dataset = build_dataset(
            cfg,
            target_source,
            manifest_path=cfg.validation.manifest_path,
            max_clips=cfg.validation.get("max_clips", None),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.validation.get("batch_size", cfg.training.batch_size),
            shuffle=False,
            num_workers=cfg.validation.get("num_workers", cfg.training.num_workers),
            drop_last=False,
            pin_memory=True,
        )
        print(
            f"  Validation: {len(val_dataset)} clips, "
            f"every {cfg.validation.eval_every_steps} step(s), "
            f"max_batches={cfg.validation.get('max_batches', None)}"
        )

    print("[4/4] Training...")
    print(f"  Teacher: {cfg.teacher.guidance_mode}, {cfg.teacher.denoising_steps} step(s)")
    print(f"  Max steps: {cfg.training.max_steps}")
    metrics_mode = "a" if resume_from and os.path.exists(metrics_path) else "w"
    metrics_file = open(metrics_path, metrics_mode, encoding="utf-8")
    if metrics_mode == "w":
        metrics_file.write(json.dumps({
            "event": "config",
            "config": OmegaConf.to_container(cfg, resolve=True),
        }) + "\n")
    else:
        metrics_file.write(json.dumps({
            "event": "resume",
            "step": global_step,
            "checkpoint": resume_from,
            "checkpoint_loss": resume_info["loss"] if resume_info else None,
            "has_optimizer": resume_info["has_optimizer"] if resume_info else False,
            "has_lr_scheduler": resume_info["has_lr_scheduler"] if resume_info else False,
            "has_scaler": resume_info["has_scaler"] if resume_info else False,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }) + "\n")
    metrics_file.flush()
    print(f"  Metrics: {metrics_path}")

    loss_accum = 0.0
    t_start = time.perf_counter()
    loss = torch.tensor(float("nan"), device=device)

    try:
        steps_per_epoch = max(len(dataloader), 1)
        start_epoch = global_step // steps_per_epoch
        for epoch in range(start_epoch, cfg.training.num_epochs):
            epoch_loss_accum = 0.0
            epoch_motion_accum = 0.0
            epoch_vel_accum = 0.0
            epoch_acc_accum = 0.0
            epoch_boundary_accum = 0.0
            epoch_delta_accum = 0.0
            epoch_steps = 0
            pbar = tqdm(dataloader, desc=f"epoch {epoch + 1}/{cfg.training.num_epochs}", dynamic_ncols=True, leave=False)
            for batch in pbar:
                if global_step >= cfg.training.max_steps:
                    break

                seed = int(cfg.seed + global_step)

                loss, loss_parts = compute_student_loss(
                    student,
                    teacher,
                    batch,
                    cfg,
                    device,
                    target_source,
                    train_flow_scheduler,
                    seed,
                )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(student.parameters(), cfg.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                global_step += 1
                epoch_steps += 1
                loss_accum += loss.item()
                epoch_loss_accum += loss.item()
                epoch_motion_accum += loss_parts["loss_motion"]
                epoch_vel_accum += loss_parts["loss_velocity"]
                epoch_acc_accum += loss_parts["loss_acceleration"]
                epoch_boundary_accum += loss_parts["loss_boundary"]
                epoch_delta_accum += loss_parts["loss_delta"]
                elapsed = time.perf_counter() - t_start
                lr = lr_scheduler.get_last_lr()[0]
                step_metrics = {
                    "event": "train",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "loss_motion": float(loss_parts["loss_motion"]),
                    "loss_velocity": float(loss_parts["loss_velocity"]),
                    "loss_acceleration": float(loss_parts["loss_acceleration"]),
                    "loss_boundary": float(loss_parts["loss_boundary"]),
                    "loss_delta": float(loss_parts["loss_delta"]),
                    "matched_frames": int(loss_parts["matched_frames"]),
                    "lr": float(lr),
                    "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                    "steps_per_sec": float(global_step / max(elapsed, 1e-6)),
                    "elapsed_sec": float(elapsed),
                }
                metrics_file.write(json.dumps(step_metrics) + "\n")
                metrics_file.flush()

                pbar.set_postfix(
                    step=global_step,
                    loss=f"{loss.item():.5f}",
                    motion=f"{loss_parts['loss_motion']:.5f}",
                    frames=loss_parts["matched_frames"],
                )

                if global_step % cfg.training.log_every == 0:
                    avg_loss = loss_accum / cfg.training.log_every
                    step_metrics["event"] = "log"
                    step_metrics["avg_loss"] = float(avg_loss)
                    metrics_file.write(json.dumps(step_metrics) + "\n")
                    metrics_file.flush()
                    os.fsync(metrics_file.fileno())
                    tqdm.write(
                        f"[step {global_step:>6d}] "
                        f"loss={avg_loss:.6f} "
                        f"motion={loss_parts['loss_motion']:.6f} "
                        f"vel={loss_parts['loss_velocity']:.6f} "
                        f"acc={loss_parts['loss_acceleration']:.6f} "
                        f"boundary={loss_parts['loss_boundary']:.6f} "
                        f"delta={loss_parts['loss_delta']:.6f} "
                        f"lr={lr:.2e} "
                        f"speed={global_step / max(elapsed, 1e-6):.2f} steps/s"
                    )
                    loss_accum = 0.0

                if global_step % cfg.training.save_every == 0:
                    last_path = os.path.join(cfg.output_dir, f"blockwise_step{global_step}.pt")
                    save_checkpoint(
                        last_path,
                        student,
                        cfg,
                        global_step,
                        loss.item(),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        best_loss=best_loss,
                    )
                    metrics_file.write(json.dumps({
                        "event": "checkpoint",
                        "step": global_step,
                        "path": last_path,
                        "loss": float(loss.item()),
                    }) + "\n")
                    metrics_file.flush()
                    os.fsync(metrics_file.fileno())
                    tqdm.write(f"  saved {last_path}")

                save_latest_every = int(cfg.training.get("save_latest_every", 0) or 0)
                if save_latest_every > 0 and global_step % save_latest_every == 0:
                    latest_path = os.path.join(cfg.output_dir, "blockwise_latest.pt")
                    save_checkpoint(
                        latest_path,
                        student,
                        cfg,
                        global_step,
                        loss.item(),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        best_loss=best_loss,
                    )
                    metrics_file.write(json.dumps({
                        "event": "checkpoint",
                        "reason": "latest",
                        "step": global_step,
                        "path": latest_path,
                        "loss": float(loss.item()),
                    }) + "\n")
                    metrics_file.flush()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_path = os.path.join(cfg.output_dir, "blockwise_best.pt")
                    save_checkpoint(
                        best_path,
                        student,
                        cfg,
                        global_step,
                        loss.item(),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        best_loss=best_loss,
                    )

                if validation_enabled and global_step % int(cfg.validation.eval_every_steps) == 0:
                    val_metrics = evaluate_student(
                        student,
                        teacher,
                        val_loader,
                        cfg,
                        device,
                        target_source,
                        train_flow_scheduler,
                        max_batches=cfg.validation.get("max_batches", None),
                        seed=int(cfg.seed + 1000000 + global_step),
                    )
                    if val_metrics is not None:
                        val_event = {
                            "event": "eval",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "elapsed_sec": float(time.perf_counter() - t_start),
                            **{f"val_{key}": float(value) for key, value in val_metrics.items() if key != "batches"},
                            "val_batches": int(val_metrics["batches"]),
                        }
                        metrics_file.write(json.dumps(val_event) + "\n")
                        metrics_file.flush()
                        os.fsync(metrics_file.fileno())
                        eval_history.append(val_event)
                        if val_metrics["loss"] < best_val_loss:
                            best_val_loss = val_metrics["loss"]
                            best_val_path = os.path.join(cfg.output_dir, "blockwise_best_val.pt")
                            save_checkpoint(
                                best_val_path,
                                student,
                                cfg,
                                global_step,
                                loss.item(),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                scaler=scaler,
                                best_loss=best_loss,
                            )
                            val_event["best_val_checkpoint"] = best_val_path
                        with open(val_summary_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "latest": val_event,
                                    "best_val_loss": float(best_val_loss),
                                    "history": eval_history,
                                },
                                f,
                                indent=2,
                            )
                        tqdm.write(
                            f"[eval step {global_step:>6d}] "
                            f"val_loss={val_metrics['loss']:.6f} "
                            f"val_motion={val_metrics['loss_motion']:.6f} "
                            f"batches={int(val_metrics['batches'])}"
                        )

            stop_after_epoch = global_step >= cfg.training.max_steps

            if epoch_steps > 0:
                epoch_metrics = {
                    "event": "epoch",
                    "reason": "max_steps" if stop_after_epoch else "epoch_end",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "avg_loss": float(epoch_loss_accum / epoch_steps),
                    "avg_loss_motion": float(epoch_motion_accum / epoch_steps),
                    "avg_loss_velocity": float(epoch_vel_accum / epoch_steps),
                    "avg_loss_acceleration": float(epoch_acc_accum / epoch_steps),
                    "avg_loss_boundary": float(epoch_boundary_accum / epoch_steps),
                    "avg_loss_delta": float(epoch_delta_accum / epoch_steps),
                    "epoch_steps": int(epoch_steps),
                    "elapsed_sec": float(time.perf_counter() - t_start),
                }
                metrics_file.write(json.dumps(epoch_metrics) + "\n")
                metrics_file.flush()
                os.fsync(metrics_file.fileno())

            if cfg.training.get("save_every_epoch", True):
                epoch_path = os.path.join(cfg.output_dir, f"blockwise_epoch{epoch + 1:03d}_step{global_step}.pt")
                save_checkpoint(
                    epoch_path,
                    student,
                    cfg,
                    global_step,
                    loss.item(),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=scaler,
                    best_loss=best_loss,
                )
                metrics_file.write(json.dumps({
                    "event": "checkpoint",
                    "reason": "epoch_end",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "path": epoch_path,
                    "loss": float(loss.item()),
                }) + "\n")
                metrics_file.flush()
                print(f"  saved epoch checkpoint: {epoch_path}")

            if stop_after_epoch:
                break

        final_path = os.path.join(cfg.output_dir, "blockwise_last.pt")
        save_checkpoint(
            final_path,
            student,
            cfg,
            global_step,
            loss.item(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            best_loss=best_loss,
        )
        metrics_file.write(json.dumps({
            "event": "done",
            "step": global_step,
            "best_loss": float(best_loss),
            "best_val_loss": float(best_val_loss) if validation_enabled else None,
            "final_checkpoint": final_path,
            "metrics": metrics_path,
            "eval_summary": val_summary_path if validation_enabled else None,
        }) + "\n")
        metrics_file.flush()
        print(f"Done. final={final_path}, best_loss={best_loss:.6f}")
    finally:
        metrics_file.close()


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
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path to resume from, or 'auto' to use the highest-step blockwise*.pt in output_dir.",
    )
    parser.add_argument(
        "--resume-metrics",
        default=None,
        help="Existing metrics.jsonl to append resume logs to. Defaults to output_dir/metrics.jsonl.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    apply_overrides(cfg, args.override)
    if args.resume is not None:
        OmegaConf.update(cfg, "training.resume_from", args.resume)
    if args.resume_metrics is not None:
        OmegaConf.update(cfg, "training.resume_metrics", args.resume_metrics)
    train(cfg)
