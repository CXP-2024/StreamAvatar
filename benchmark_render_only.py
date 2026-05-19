"""
Precise per-frame render timing (excluding model load and video encoding).
"""
import sys, os, time
os.chdir("/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/tools/visualization_0416")
sys.path.insert(0, ".")
sys.path.insert(0, "../..")

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Load render model
print("Loading render model...")
t0 = time.perf_counter()
sys.path.insert(0, './utils/model_0506')
from utils import instantiate
from omegaconf import OmegaConf

config_path = "./configs/head_animator_best_0506.yaml"
config = OmegaConf.load(config_path)
module = instantiate(config.model, instantiate_module=False)
model = module(config=config)
checkpoint = torch.load(config.resume_ckpt, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"], strict=False)
model.eval().to("cuda")
model_load_time = time.perf_counter() - t0
print(f"  Model load: {model_load_time:.1f}s")

transform = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5], [0.5])])

# Load test data
npz_path = "/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/results/benchmark_motion_output.npz"
data = np.load(npz_path, allow_pickle=True)
motion_latents = torch.from_numpy(data["motion_latent"]).cuda()  # (1, T, 512)
ref_img_path = "/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/img_files/person2_resize.png"

# Load reference image
ref_img = Image.open(ref_img_path).convert("RGB")
ref_img_t = transform(ref_img).unsqueeze(0).cuda()

# Encode reference (one-time)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    face_feat = model.face_encoder(ref_img_t)
    ref_latent = motion_latents[0, 0:1]  # first frame as reference
torch.cuda.synchronize()
ref_enc_time = time.perf_counter() - t0
print(f"  Reference encode: {ref_enc_time*1000:.1f}ms")

# Benchmark frame rendering
T_frames = motion_latents.shape[1]
num_test = min(200, T_frames)
print(f"\n  Rendering {num_test} frames...")

# Warmup
with torch.no_grad():
    for i in range(3):
        drv_latent = motion_latents[0, i:i+1]
        tgt = model.flow_estimator(ref_latent, drv_latent)
        _ = model.face_generator(tgt, face_feat)
torch.cuda.synchronize()

# Timed run
frame_times = []
torch.cuda.synchronize()
total_start = time.perf_counter()
with torch.no_grad():
    for i in range(num_test):
        torch.cuda.synchronize()
        ft0 = time.perf_counter()
        drv_latent = motion_latents[0, i:i+1]
        tgt = model.flow_estimator(ref_latent, drv_latent)
        out = model.face_generator(tgt, face_feat)
        torch.cuda.synchronize()
        frame_times.append(time.perf_counter() - ft0)

total_render = time.perf_counter() - total_start
avg_ms = np.mean(frame_times) * 1000
p50 = np.percentile(frame_times, 50) * 1000
p95 = np.percentile(frame_times, 95) * 1000
fps = 1000 / avg_ms

print(f"\n  ═══════════════════════════════════════")
print(f"  RENDER BENCHMARK RESULTS ({num_test} frames)")
print(f"  ═══════════════════════════════════════")
print(f"  Total: {total_render:.2f}s")
print(f"  Avg:   {avg_ms:.2f}ms/frame ({fps:.1f} fps)")
print(f"  P50:   {p50:.2f}ms")
print(f"  P95:   {p95:.2f}ms")
print(f"  60s video estimate: {avg_ms/1000 * 1499:.1f}s")
print(f"  ═══════════════════════════════════════")
