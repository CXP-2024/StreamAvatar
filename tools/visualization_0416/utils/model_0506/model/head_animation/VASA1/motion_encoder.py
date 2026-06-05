import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import torchvision
from model.head_animation.VASA1.resnet18 import Resnet18
from model.head_animation.VASA1.rigid_pose_encoder import RigidPoseEncoder
from model.head_animation.VASA1.hopenet import Hopenet, euler_to_rotation_matrix, headposeprediction_to_euler

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim: int, size: int, normalize_output: bool, hopenet_checkpoint_path: str, use_gt_rotation: bool):
        super().__init__()

        self.expression_encoder = Resnet18(input_dim=3, output_dim=latent_dim, normalize_output=normalize_output)

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
                    inp = (inp + 1) / 2 # need [0, 1]
                return euler_to_rotation_matrix(*headposeprediction_to_euler(*self.hopenet(self.hopenet_test_transformations(inp))))

        self.rigid_pose_encoder = RigidPoseEncoder(use_gt_rotation=use_gt_rotation, gt_rotation_callback=gt_rotation_callback)

    def forward(self, inp: torch.Tensor):
        bs = inp.size(0)
        expression_code = self.expression_encoder(inp)  # [batch_size, latent_dim]
        rigid_pose = self.rigid_pose_encoder(inp) # rotation [batch_size, 3, 3] and translation [batch_size, 3]

        # so motion latent is 128 + 9 + 3 = 140?
        motion_latent = torch.cat((expression_code, rigid_pose["rotation"].view(bs, -1), rigid_pose["translation"]), dim=1)

        return motion_latent, rigid_pose
