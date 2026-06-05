import torch
import torch.nn as nn
import sys
from pathlib import Path
from model.head_animation.VASA1.building_blocks import USE_BIAS, ResBlock2d, ResBlock3d, ReshapeTo3DLayer
from model.head_animation.VASA1.resnet50 import Resnet50

class FaceEncoder(nn.Module):
    def __init__(self, latent_dim: int, normalize_output: bool):
        super().__init__()

        num_channels_per_group = 8
        self.VolumetricFieldEncoder = nn.Sequential(
            # 2D conv layers
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=USE_BIAS),
            ResBlock2d(64, 128, num_channels_per_group),
            nn.AvgPool2d(2, 2),
            ResBlock2d(128, 256, num_channels_per_group),
            nn.AvgPool2d(2, 2),
            ResBlock2d(256, 512, num_channels_per_group),
            nn.AvgPool2d(2, 2),
            # Prepare for reshaping
            nn.GroupNorm(512 // num_channels_per_group, 512, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 96 * 16, kernel_size=1, bias=USE_BIAS),
            # Reshape 2D tensor as a 3D tensor with depth 16.
            ReshapeTo3DLayer(16),
            # 3D conv layers
            ResBlock3d(96, 96, num_channels_per_group),
            ResBlock3d(96, 96, num_channels_per_group),
            ResBlock3d(96, 96, num_channels_per_group),
        )

        self.global_descriptor_encoder = Resnet50(input_dim=3, output_dim=latent_dim, normalize_output=normalize_output)

    def forward(self, inp: torch.Tensor):
        feature_volume = self.VolumetricFieldEncoder(inp)
        global_descriptor = self.global_descriptor_encoder(inp)
        return [feature_volume, global_descriptor]
