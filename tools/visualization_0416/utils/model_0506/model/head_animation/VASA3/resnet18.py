import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head_animation.VASA1.building_blocks import USE_BIAS, ResBasic


class Resnet18(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, add_fc: bool, dropout: float = 0.2):
        super().__init__()

        num_channels_per_group = 32

        # The following architecture follows Resnet-18.
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
        )

        if add_fc:
            self.final_layers = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(
                    in_channels=512,
                    out_channels=output_dim,
                    kernel_size=1,
                    bias=USE_BIAS),
                nn.Dropout(p=dropout),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(output_dim*4**2, output_dim, bias=False),
            )
        else:
            self.final_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, output_dim, bias=False),
            )

    def forward(self, inp: torch.Tensor):
        out = self.layers(inp)
        out = self.final_layers(out)

        return out
