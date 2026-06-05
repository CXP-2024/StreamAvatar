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

def init_fn(config_path):
    from utils import instantiate
    transform = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    config = OmegaConf.load(config_path)
    module = instantiate(config.model, instantiate_module=False)
    model = module(config=config)
    checkpoint = torch.load(config.resume_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    motion_encoder = model.motion_encoder
    return {"transform": transform, "motion_encoder": motion_encoder}

def extract_motion_latent(
    mask_image_path='./test_case/test_img_masked.png',
    config_path='./configs/head_animator_best_0506.yaml',
    save_npz_path='./test_case/test_img_resize.npz',
    version="0506"):
    sys.path.insert(0, f'./utils/model_{version}')
    config_path = config_path.replace("0506", version)
    context = init_fn(config_path)
    transform = context["transform"]
    motion_encoder = context["motion_encoder"]
    img = Image.open(mask_image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        latent = motion_encoder(img_tensor)[0]  # [1, 512]
    latent_np = latent.numpy()

    # 如果文件已存在，先加载原有数据
    if os.path.exists(save_npz_path):
        existing_data = np.load(save_npz_path, allow_pickle=True)
        data_dict = dict(existing_data)
        existing_data.close()  # 关闭文件
    else:
        data_dict = {}

    # 更新或添加新的键值对
    data_dict.update({
        'video_id': os.path.basename(save_npz_path)[:-4],
        'mask_img_path': mask_image_path,
        'ref_img_path': save_npz_path.replace('npz', 'png'),
        'motion_latent': latent_np
    })

    # 保存更新后的数据
    np.savez(save_npz_path, **data_dict)
    # np.savez(
    #     save_npz_path,
    #     video_id=os.path.basename(save_npz_path)[:-4],
    #     mask_img_path=mask_image_path,
    #     ref_img_path=save_npz_path.replace('npz', 'png'),
    #     motion_latent=latent_np
    # )
if __name__ == '__main__':
    fire.Fire(extract_motion_latent)
