# Adapted from https://github.com/eriklindernoren/PyTorch-GAN/tree/master, which is MIT license

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm



class DiscriminatorBlock(nn.Module):
    def __init__(self, in_filters, out_filters, normalization=True):
        super(DiscriminatorBlock, self).__init__()

        self.layers = nn.Sequential()
        self.layers.append(
            spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1))
        )
        if normalization:
            self.layers.append(nn.InstanceNorm2d(out_filters))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

    def forward(self, inp):
        return self.layers(inp)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.down_blocks = nn.ModuleList(
            [
                DiscriminatorBlock(in_channels, 64, normalization=False),
                DiscriminatorBlock(64, 128),
                DiscriminatorBlock(128, 256),
                DiscriminatorBlock(256, 512),
            ]
        )
        self.final_layer = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, 4, padding=1, bias=False))
        )

    def forward(self, inp):
        feature_maps = []
        for block in self.down_blocks:
            inp = block(inp)
            feature_maps.append(inp)

        return feature_maps, self.final_layer(inp)
