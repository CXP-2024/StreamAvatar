import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models
from torch.cuda import amp
import math

from .utils import blocks, norm_layers, activations
from dataclasses import dataclass

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class AppearanceEncoder(nn.Module):

    @dataclass
    class Config:
        gen_upsampling_type: str
        gen_downsampling_type: str
        gen_input_image_size: int
        gen_latent_texture_size: int
        gen_latent_texture_depth: int
        gen_latent_texture_channels: int
        gen_num_channels: int
        enc_channel_mult: float
        norm_layer_type: str
        gen_max_channels: int
        enc_block_type: str
        gen_activation_type: str
        warp_norm_grad: bool
        in_channels: int = 3

    def __init__(self, cfg: Config):
        super(AppearanceEncoder, self).__init__()

        self.cfg = cfg
        self.upsample_type = self.cfg.gen_upsampling_type
        self.downsample_type = self.cfg.gen_downsampling_type
        self.ratio = self.cfg.gen_input_image_size // self.cfg.gen_latent_texture_size
        self.num_2d_blocks = int(math.log(self.ratio, 2))
        self.init_depth = self.cfg.gen_latent_texture_depth
        spatial_size = self.cfg.gen_input_image_size

        out_channels = int(self.cfg.gen_num_channels * self.cfg.enc_channel_mult)

        setattr(
            self,
            f'from_rgb_{spatial_size}px',
            nn.Conv2d(
                in_channels=self.cfg.in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                ))

        norm = self.cfg.norm_layer_type

        for i in range(self.num_2d_blocks):
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.cfg.gen_max_channels)
            setattr(
                self,
                f'enc_{i}_block={spatial_size}px',
                blocks[self.cfg.enc_block_type](
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    norm_layer_type=norm,
                    activation_type=self.cfg.gen_activation_type,
                    resize_layer_type=self.cfg.gen_downsampling_type))
            spatial_size //= 2

        in_channels = out_channels
        out_channels = self.cfg.gen_latent_texture_channels
        finale_layers = []
        if self.cfg.enc_block_type == 'res':
            finale_layers += [
                norm_layers[norm](in_channels),
                activations[self.cfg.gen_activation_type](inplace=True)]

        finale_layers += [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * self.init_depth,
                kernel_size=1)]


        self.finale_layers = nn.Sequential(*finale_layers)

    def forward(self, source_img):

        s = source_img.shape[2]

        x = getattr(self, f'from_rgb_{s}px')(source_img)

        for i in range(self.num_2d_blocks):
            x = getattr(self, f'enc_{i}_block={s}px')(x)
            s //= 2

        x = self.finale_layers(x)

        return x
