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

def latent_to_video(
    npz_dir="./test_case/",
    save_dir="./test_case/",
    save_fps: int = 25,
    config_path: str = './configs/head_animator_best_0416.yaml',
    version: str = '0416',
):
    # 处理相对路径：
    # - npz_dir 和 save_dir：如果是相对路径，转换为基于项目根目录的绝对路径
    # - config_path：如果是相对路径，转换为基于当前脚本目录（tools/visualization_0416/）的绝对路径
    if not os.path.isabs(npz_dir):
        npz_dir = os.path.join(_PROJECT_ROOT, npz_dir)
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT_ROOT, save_dir)
    if not os.path.isabs(config_path):
        config_path = os.path.join(_SCRIPT_DIR, config_path)

    # 规范化路径（去除多余的 . 和 ..）
    npz_dir = os.path.normpath(npz_dir)
    save_dir = os.path.normpath(save_dir)
    config_path = os.path.normpath(config_path)

    os.makedirs(save_dir, exist_ok=True)
    # 只在文件名上做版本号替换，避免把路径里的 "0416" 一并替换成 "0506"
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    config_name = config_name.replace("0416", version)
    config_path = os.path.join(config_dir, config_name)

    # Initialize models only once
    print("Initializing models...")
    print(f"NPZ directory: {npz_dir}")
    print(f"Save directory: {save_dir}")
    ctx = init_fn(config_path, version)
    transform = ctx["transform"]
    flow_estimator = ctx["flow_estimator"]
    face_generator = ctx["face_generator"]
    face_encoder = ctx["face_encoder"]

    # Get all npz files
    if not os.path.exists(npz_dir):
        print(f"Error: NPZ directory does not exist: {npz_dir}")
        return

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('_output.npz')]
    print(f"Found {len(npz_files)} files to process")

    # Process each file
    for npz_file in tqdm(npz_files, desc="Processing files"):
        if not npz_file.endswith('.npz'): continue
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
            # np.save("/mnt/weka/haiyang_workspace/ckpts/good_train_case/image_example/face_encoder_input.npy", ref_img.cpu().numpy())

            with torch.no_grad():
                face_feat = face_encoder(ref_img)
                # np.save("/mnt/weka/haiyang_workspace/ckpts/good_train_case/image_example/face_encoder_output.npy", face_feat.cpu().numpy())
                recon_list = []
                for i in range(0, num_frames):
                    tgt = flow_estimator(motion_latent[0:1], motion_latent[i:i+1])
                    recon_list.append(face_generator(tgt, face_feat))

            recon = torch.cat(recon_list, dim=0)
            video_np = recon.permute(0, 2, 3, 1).cpu().numpy()
            video_np = np.clip((video_np + 1) / 2 * 255, 0, 255).astype("uint8")

            video_id = str(data["video_id"])
            # Remove leading dash to prevent FFMPEG command line parsing issues
            if video_id.startswith('-'):
                video_id = video_id[1:]

            if num_frames == 1:
                out_path = os.path.join(save_dir, f"{video_id}_rec.png")
                Image.fromarray(video_np[0]).save(out_path)
            else:
                temp_mp4 = os.path.join(save_dir, f"{video_id}_temp.mp4")
                final_mp4 = os.path.join(save_dir, f"{video_id}.mp4")
                finalfinal_mp4 = os.path.join(save_dir, f"{str(data['video_id'])}.mp4")
                with imageio.get_writer(temp_mp4, fps=save_fps) as writer:
                    for frame in video_np:
                        writer.append_data(frame)
                # 处理 audio_path - 如果是相对路径，基于项目根目录解析
                audio_path = str(data["audio_path"]) if "audio_path" in data.files else None
                if audio_path and not os.path.isabs(audio_path):
                    audio_path = os.path.join(_PROJECT_ROOT, audio_path)
                if audio_path and os.path.exists(audio_path):
                    clip = mp.VideoFileClip(temp_mp4)
                    audio = mp.AudioFileClip(audio_path)
                    clip.set_audio(audio).write_videofile(final_mp4, codec="libx264", audio_codec="aac")
                    clip.close()
                    audio.close()
                    os.remove(temp_mp4)
                else:
                    os.rename(temp_mp4, final_mp4)
                os.rename(final_mp4, finalfinal_mp4)
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue

if __name__ == "__main__":
    fire.Fire(latent_to_video)
    # Example usage:
    # python latent_to_video.py --npz_dir ./test_case/ --save_dir ./test_case/ --config_path ./configs/head_animator_best_0409.yaml --version 0416
