import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from model.head_animation.VASA3.building_blocks import USE_BIAS, ResBlock2d, ReshapeTo2DLayer

class Generator(nn.Module):
    def __init__(self, freeze: bool):
        super().__init__()
        self.freeze = freeze

        num_channels_per_group = 32
        self.layers = nn.Sequential(
            # Projection layers
            ReshapeTo2DLayer(),
            spectral_norm(nn.Conv2d(96 * 16, 512, kernel_size=1, bias=USE_BIAS)),
            # Residual blocks
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            ResBlock2d(512, 512, num_channels_per_group, use_spectral_norm=True),
            # Upsample layers
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            ResBlock2d(512, 256, num_channels_per_group, use_spectral_norm=True),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            ResBlock2d(256, 128, num_channels_per_group, use_spectral_norm=True),
            nn.Upsample(scale_factor=(2, 2), mode="nearest"),
            ResBlock2d(128, 64, num_channels_per_group, use_spectral_norm=True),
            # Final layers
            nn.GroupNorm(64 // num_channels_per_group, 64, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=USE_BIAS)),
            nn.Tanh()
        )
    def forward(self, inp: torch.Tensor):
        return self.layers(inp)
