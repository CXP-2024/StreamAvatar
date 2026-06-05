"""
批处理优化版本的 latent_to_video
相比原版逐帧处理，使用批处理加速约 10-30 倍
v2: 优化 GPU→CPU 传输和视频编码，使用流式处理
"""
import sys
import os

# 获取项目根目录并添加到 sys.path 最前面，确保导入正确的 utils 模块
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from omegaconf import OmegaConf
import fire
import imageio
import moviepy.editor as mp
from tqdm import tqdm
import time
import subprocess
import tempfile


def init_fn(config_path, version):
    sys.path.insert(0, f'./utils/model_{version}')
    from utils import instantiate
    config = OmegaConf.load(config_path)
    module = instantiate(config.model, instantiate_module=False)
    model = module(config=config)
    checkpoint = torch.load(config.resume_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval().to("cuda")
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    return {
        "transform": transform,
        "flow_estimator": model.flow_estimator,
        "face_generator": model.face_generator,
        "face_encoder": model.face_encoder,
    }


def latent_to_video_batch(
    npz_dir="./test_case/",
    save_dir="./test_case/",
    save_fps: int = 25,
    config_path: str = './configs/head_animator_best_0416.yaml',
    version: str = '0416',
    batch_size: int = 32,
    use_fp16: bool = True,
):
    """
    批处理优化版本的 latent_to_video

    Args:
        npz_dir: NPZ 文件目录
        save_dir: 输出视频目录
        save_fps: 输出视频帧率
        config_path: 模型配置文件路径
        version: 模型版本
        batch_size: 批处理大小，根据显存调整 (默认 32，显存不足可降到 16 或 8)
        use_fp16: 是否使用混合精度加速 (默认 True)
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = config_path.replace("0416", version)

    # Initialize models only once
    print("Initializing models...")
    ctx = init_fn(config_path, version)
    transform = ctx["transform"]
    flow_estimator = ctx["flow_estimator"]
    face_generator = ctx["face_generator"]
    face_encoder = ctx["face_encoder"]

    # Get all npz files
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('_output.npz')]
    print(f"Found {len(npz_files)} files to process")
    print(f"Batch size: {batch_size}, FP16: {use_fp16}")

    total_frames = 0
    total_time = 0

    # Process each file
    for npz_file in tqdm(npz_files, desc="Processing files"):
        if not npz_file.endswith('.npz'):
            continue
        try:
            npz_path = os.path.join(npz_dir, npz_file)
            data = np.load(npz_path, allow_pickle=True)
            motion_latent = torch.from_numpy(data["motion_latent"]).to("cuda").float()
            if len(motion_latent.shape) == 3:
                motion_latent = motion_latent.squeeze(0)
            num_frames = motion_latent.shape[0]
            print(f"\nProcessing {npz_file} with {num_frames} frames")

            # 处理 ref_img_path - 如果是相对路径，基于项目根目录解析
            ref_img_path = str(data["ref_img_path"])
            if not os.path.isabs(ref_img_path):
                ref_img_path = os.path.join(_PROJECT_ROOT, ref_img_path)
            ref_img = Image.open(ref_img_path).convert("RGB")
            ref_img = transform(ref_img).unsqueeze(0).to("cuda")

            video_id = str(data["video_id"])
            # Remove leading dash to prevent FFMPEG command line parsing issues
            if video_id.startswith('-'):
                video_id = video_id[1:]

            # 处理 audio_path
            audio_path = str(data["audio_path"]) if "audio_path" in data.files else None
            if audio_path and not os.path.isabs(audio_path):
                audio_path = os.path.join(_PROJECT_ROOT, audio_path)

            start_time = time.time()

            # 准备输出路径
            temp_mp4 = os.path.join(save_dir, f"{video_id}_temp.mp4")
            final_mp4 = os.path.join(save_dir, f"{video_id}.mp4")
            finalfinal_mp4 = os.path.join(save_dir, f"{str(data['video_id'])}.mp4")

            if num_frames == 1:
                # 单帧情况
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        face_feat = face_encoder(ref_img)
                        tgt = flow_estimator(motion_latent[0:1], motion_latent[0:1])
                        recon = face_generator(tgt, face_feat)
                        if use_fp16:
                            recon = recon.float()

                video_np = recon.permute(0, 2, 3, 1).cpu().numpy()
                video_np = np.clip((video_np + 1) / 2 * 255, 0, 255).astype("uint8")
                out_path = os.path.join(save_dir, f"{video_id}_rec.png")
                Image.fromarray(video_np[0]).save(out_path)
            else:
                # 多帧情况 - 使用 FFmpeg pipe 流式编码
                # 启动 FFmpeg 进程
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', '512x512',
                    '-pix_fmt', 'rgb24',
                    '-r', str(save_fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    temp_mp4
                ]

                ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        face_feat = face_encoder(ref_img)  # (1, 32, 16, 64, 64)
                        ref_latent = motion_latent[0:1]  # 参考帧的 latent

                        # 批处理推理 + 流式写入
                        for i in range(0, num_frames, batch_size):
                            batch_end = min(i + batch_size, num_frames)
                            current_batch_size = batch_end - i

                            # 获取当前批次的 motion latent
                            batch_motion = motion_latent[i:batch_end]

                            # 扩展参考帧 latent 到批次大小
                            ref_latent_expanded = ref_latent.expand(current_batch_size, -1)

                            # 扩展 face_feat 到批次大小
                            face_feat_expanded = face_feat.expand(current_batch_size, -1, -1, -1, -1)

                            # 批量计算 flow
                            tgt = flow_estimator(ref_latent_expanded, batch_motion)

                            # 批量生成图像
                            recon = face_generator(tgt, face_feat_expanded)

                            # 转换并写入 - 直接在 GPU 上做归一化
                            # (batch, 3, 512, 512) -> (batch, 512, 512, 3)
                            recon = recon.float()
                            recon = (recon + 1) / 2 * 255
                            recon = recon.clamp(0, 255).to(torch.uint8)
                            recon = recon.permute(0, 2, 3, 1).contiguous()

                            # 分块传输到 CPU 并写入
                            frames_np = recon.cpu().numpy()
                            ffmpeg_process.stdin.write(frames_np.tobytes())

                # 关闭 FFmpeg
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                elapsed = time.time() - start_time
                total_frames += num_frames
                total_time += elapsed
                fps = num_frames / elapsed
                print(f"  Rendered + encoded {num_frames} frames in {elapsed:.2f}s ({fps:.1f} fps)")

                # 合并音频
                if audio_path and os.path.exists(audio_path):
                    # 使用 FFmpeg 直接合并音频（比 moviepy 快很多）
                    final_with_audio = os.path.join(save_dir, f"{video_id}_with_audio.mp4")
                    ffmpeg_audio_cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_mp4,
                        '-i', audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',
                        final_with_audio
                    ]
                    subprocess.run(ffmpeg_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(temp_mp4)
                    os.rename(final_with_audio, finalfinal_mp4)
                else:
                    os.rename(temp_mp4, finalfinal_mp4)

        except Exception as e:
            import traceback
            print(f"Error processing {npz_file}: {str(e)}")
            traceback.print_exc()
            continue

    # 打印总体统计
    if total_time > 0:
        print(f"\n{'='*50}")
        print(f"总计: {total_frames} 帧, {total_time:.2f} 秒")
        print(f"平均渲染速度: {total_frames / total_time:.1f} fps")
        print(f"{'='*50}")


if __name__ == "__main__":
    fire.Fire(latent_to_video_batch)
    # Example usage:
    # python latent_to_video_batch.py --npz_dir ./test_case/ --save_dir ./test_case/ --batch_size 32 --use_fp16 True
