import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path
import math

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.util import *

class SPADEGenerator(nn.Module):
    def __init__(self, cfg):
        super(SPADEGenerator, self).__init__()

        # settings for image_generator
        self.in_channels = cfg.in_channels
        self.proj_channels = cfg.proj_channels # channel of projected feature volume
        self.flag_estimate_occlusion_map = cfg.flag_estimate_occlusion_map
        self.final_activation = cfg.final_activation
        self.zero_to_one = cfg.zero_to_one

        self.norm_G = 'spadespectralinstance'
        self.label_num_channels = self.proj_channels
        self.out_channels = 64 # output channel of final SPADEResnetBlock

        ### generator
        self.init_image_generator()

    def init_image_generator(self):
        # Projection layers
        self.projection = nn.Sequential(
            SameBlock2d(self.in_channels, self.proj_channels, kernel_size=(3, 3), padding=(1, 1), lrelu=True),
            nn.Conv2d(self.proj_channels, self.proj_channels, kernel_size=1, stride=1)
        )

        self.fc = nn.Conv2d(self.proj_channels, 2 * self.proj_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * self.proj_channels, 2 * self.proj_channels, self.norm_G, self.label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * self.proj_channels, self.proj_channels, self.norm_G, self.label_num_channels)
        self.up_1 = SPADEResnetBlock(self.proj_channels, self.out_channels, self.norm_G, self.label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        self.conv_img = nn.Sequential(
            nn.Conv2d(self.out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )

        if self.final_activation:
            self.final_activation_fn = nn.Sigmoid() if self.zero_to_one else nn.Tanh()
        else:
            self.final_activation_fn = nn.Identity()

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.in_channels, 1, kernel_size=7, padding=3)

    def image_generation(self, warping_feature_volume):
        seg = self.projection(warping_feature_volume) # Bx256x64x64

        if self.flag_estimate_occlusion_map:
            occlusion_map = torch.sigmoid(self.occlusion(warping_feature_volume))  # Bx1x64x64
            seg = seg * occlusion_map

        x = self.fc(seg)  # Bx512x64x64
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3xHxW
        x = self.final_activation_fn(x)  # Bx3xHxW

        return x

    def forward(self, data_dict, target_warp_embed_dict, aligned_target_volume):
        # decoding
        img = self.image_generation(aligned_target_volume)

        return img, None, None, None
