import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import torchvision
from model.head_animation.VASA3.resnet18 import Resnet18
from model.head_animation.VASA3.rigid_pose_encoder import RigidPoseEncoder
from model.head_animation.VASA3.hopenet import Hopenet, euler_to_rotation_matrix, headposeprediction_to_euler

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim: int, size: int, resize: bool, hopenet_checkpoint_path: str, use_gt_rotation: bool):
        super().__init__()
        self.resize=resize
        self.expression_encoder = Resnet18(input_dim=3, output_dim=latent_dim, add_fc=True)

        # Hopenet
        self.hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.hopenet.load_state_dict(
            torch.load(Path(hopenet_checkpoint_path)),
            strict=False,  # Don't need vestigial layer
        )
        self.hopenet.eval()
        for param in self.hopenet.parameters():
            param.requires_grad = False

        self.hopenet_test_transformations = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224), # only support 224
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        def gt_rotation_callback(inp):
            with torch.no_grad():
                # Important: need [0, 1]
                if inp.min() <= -0.5:
                    inp_org = (inp + 1) / 2 # need [0, 1]
                rot = euler_to_rotation_matrix(*headposeprediction_to_euler(*self.hopenet(self.hopenet_test_transformations(inp_org))))
                # img_org = inp.permute(0, 2,3, 1)
                # img_org = torch.clamp(img_org, min=0.0, max=1.0) * 255
                # img_org = img_org.cpu().numpy().astype('uint8')
                # cv2.imwrite(f'debug/debug_ref_callback.png', img_org[0][:,:,::-1])

                return rot

        self.rigid_pose_encoder = RigidPoseEncoder(use_gt_rotation=use_gt_rotation, gt_rotation_callback=gt_rotation_callback)

    def forward(self, masked_inp, inp):
        bs = inp.size(0)

        if self.resize:
            masked_inp = F.interpolate(masked_inp, size=(256, 256), mode='bilinear')

        rigid_pose = self.rigid_pose_encoder(masked_inp, inp) # rotation [batch_size, 3, 3] and translation [batch_size, 3]
        expression_code = self.expression_encoder(masked_inp)  # [batch_size, latent_dim]

        # so motion latent is 128 + 9 + 3 = 140?
        motion_latent = torch.cat((expression_code, rigid_pose["rotation"].view(bs, -1), rigid_pose["translation"]), dim=1)

        return motion_latent, rigid_pose
