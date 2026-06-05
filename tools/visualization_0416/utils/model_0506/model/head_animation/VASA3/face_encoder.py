import torch
import torch.nn as nn
import sys
from pathlib import Path
import torch.nn.functional as F
from model.head_animation.VASA3.building_blocks import USE_BIAS, ResBlock2d, ResBlock3d, ReshapeTo3DLayer
from model.head_animation.VASA3.resnet50 import Resnet50

class FaceEncoder(nn.Module):
    def __init__(self, latent_dim: int, resize: bool, freeze: bool):
        super().__init__()

        depth, channel = 16, 96
        self.resize = resize

        num_channels_per_group = 32
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
            nn.Conv2d(512, channel * depth, kernel_size=1, bias=USE_BIAS),
            # Reshape 2D tensor as a 3D tensor with depth 16.
            ReshapeTo3DLayer(depth),
            # 3D conv layers
            ResBlock3d(channel, channel, num_channels_per_group),
            ResBlock3d(channel, channel, num_channels_per_group),
            ResBlock3d(channel, channel, num_channels_per_group),
        )

        self.global_descriptor_encoder = Resnet50(input_dim=3, output_dim=latent_dim)

    def forward(self, inp: torch.Tensor):
        feature_volume = self.VolumetricFieldEncoder(inp)

        if self.resize:
            inp = F.interpolate(inp, size=(256, 256), mode='bilinear')
        global_descriptor = self.global_descriptor_encoder(inp)
        return [feature_volume, global_descriptor]
