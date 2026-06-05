import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head_animation.VASA1.building_blocks import USE_BIAS, ResBasic


class Resnet18(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, normalize_output: bool = False):
        super().__init__()

        num_channels_per_group = 8

        # The following architecture follows Resnet-50.
        self.layers = nn.Sequential(
            # Initial layers
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=USE_BIAS),
            nn.GroupNorm(64 // num_channels_per_group, 64, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Layer 1
            ResBasic(64, 64, 1, num_channels_per_group),
            ResBasic(64, 64, 1, num_channels_per_group),
            # Layer 2
            ResBasic(64, 128, 2, num_channels_per_group),
            ResBasic(128, 128, 1, num_channels_per_group),
            # Layer 3
            ResBasic(128, 256, 2, num_channels_per_group),
            ResBasic(256, 256, 1, num_channels_per_group),
            # Layer 4
            ResBasic(256, 512, 2, num_channels_per_group),
            ResBasic(512, 512, 1, num_channels_per_group),
            # Global average pooling.
            nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten.
            nn.Flatten(start_dim=1),
            # Final layer
            nn.Linear(512, output_dim),
        )
        self.normalize_output = normalize_output

    def forward(self, inp: torch.Tensor):
        out = self.layers(inp)
        if self.normalize_output:
            out = F.normalize(out, dim=1)

        return out
