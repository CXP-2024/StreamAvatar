import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.modules import *
from model.head_animation.LIA.util import SameBlock2d, DownBlock2d, ResBlock3d

class FaceEncoder(nn.Module):
    def __init__(self, output_channels, size=512):
        super(FaceEncoder, self).__init__()

        channel = [32, 64, 128, 256, 512, 512, 512, output_channels]

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channel[0], 1))

        in_channel = channel[0]
        for i in range(1, len(channel)):
            out_channel = channel[i]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*self.convs)

    def forward(self, source_image):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[source_image],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[source_image])

    def manual_forward(self, source_image):
        res = []
        h = source_image
        for conv in self.convs:
            h = conv(h)
            res.append(h)
        # TODO(wei): remove the last layer?
        feats = res[::-1][1:] # from 8x8 to 512x512
        return feats



"""
Appearance extractor(F) defined in LivePortrait, which maps the source image s to a 3D appearance feature volume.
"""
class FaceEncoder3D(nn.Module):
    def __init__(self, image_size, image_channel, block_expansion, num_down_blocks, max_features, reshape_channel, reshape_depth, num_resblocks):
        super(FaceEncoder3D, self).__init__()
        self.image_size = image_size
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        self.max_features = max_features
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

    def forward(self, source_image):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[source_image],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[source_image])

    def manual_forward(self, source_image):
        if source_image.size(-1) != self.image_size:
            source_image = F.interpolate(source_image, size=(self.image_size, self.image_size), mode='bilinear')

        out = self.first(source_image)  # Bx3x256x256 -> Bx64x256x256

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape  # ->Bx512x64x64

        f_s = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)  # ->Bx32x16x64x64
        f_s = self.resblocks_3d(f_s)  # ->Bx32x16x64x64
        return f_s
