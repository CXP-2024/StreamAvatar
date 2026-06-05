import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


USE_BIAS = False


# https://github.com/joe-siyuan-qiao/WeightStandardization?tab=readme-ov-file#pytorch
class WSConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(WSConv2d, self).__init__(*args, **kwargs)

    def forward(self, inp):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(inp, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class WSConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(WSConv3d, self).__init__(*args, **kwargs)

    def forward(self, inp):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(inp, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_channels_per_group: int, use_spectral_norm: bool = False):
        super().__init__()

        norm_func = lambda x: x
        if use_spectral_norm:
            norm_func = spectral_norm

        if in_channels != out_channels:
            self.skip_layer = norm_func(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS))
        else:
            self.skip_layer = lambda x: x

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // num_channels_per_group, in_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            norm_func(WSConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)),
            nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            norm_func(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)),
        )

    def forward(self, inp: torch.Tensor):
        return self.skip_layer(inp) + self.layers(inp)


class ResBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_channels_per_group: int):
        super().__init__()

        if in_channels != out_channels:
            self.skip_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)
        else:
            self.skip_layer = lambda x: x

        self.layers = nn.Sequential(
            nn.GroupNorm(in_channels // num_channels_per_group, in_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            WSConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS),
            nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS),
        )

    def forward(self, inp: torch.Tensor):
        return self.skip_layer(inp) + self.layers(inp)


class ResBasic(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, num_channels_per_group: int):
        super().__init__()

        if stride != 1 and stride != 2:
            raise NotImplementedError(f"Stride can be only 1 or 2 but '{stride}' is passed.")

        if in_channels != out_channels or stride != 1:
            self.skip_layer = nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=USE_BIAS),
                nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
            )
        else:
            self.skip_layer = lambda x: x

        self.layers = nn.Sequential(
            WSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=USE_BIAS),
            nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            WSConv2d(out_channels, out_channels, kernel_size=1, bias=USE_BIAS),
            nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
        )


    def forward(self, inp: torch.Tensor):
        return F.relu(self.skip_layer(inp) + self.layers(inp))


class ResBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, num_channels_per_group: int):
        super().__init__()

        if stride != 1 and stride != 2:
            raise NotImplementedError(f"Stride can be only 1 or 2 but '{stride}' is passed.")

        if in_channels != out_channels or stride != 1:
            self.skip_layer = nn.Sequential(
                WSConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=USE_BIAS),
                nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
            )
        else:
            self.skip_layer = lambda x: x

        temp_out_channels = out_channels // 4
        self.layers = nn.Sequential(
            WSConv2d(in_channels, temp_out_channels, kernel_size=1, bias=USE_BIAS),
            nn.GroupNorm(temp_out_channels // num_channels_per_group, temp_out_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            WSConv2d(temp_out_channels, temp_out_channels, kernel_size=3, stride=stride, padding=1, bias=USE_BIAS),
            nn.GroupNorm(temp_out_channels // num_channels_per_group, temp_out_channels, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            WSConv2d(temp_out_channels, out_channels, kernel_size=1, bias=USE_BIAS),
            nn.GroupNorm(out_channels // num_channels_per_group, out_channels, affine=not USE_BIAS),
        )


    def forward(self, inp: torch.Tensor):
        return F.relu(self.skip_layer(inp) + self.layers(inp))


class ReshapeTo3DLayer(nn.Module):
    def __init__(self, out_depth: int):
        super().__init__()

        self.out_depth = out_depth

    def forward(self, inp: torch.Tensor):
        batch_size, channels, height, width = inp.shape
        return inp.view(batch_size, channels // self.out_depth, self.out_depth, height, width)


class ReshapeTo2DLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp: torch.Tensor):
        batch_size, channels, depth, height, width = inp.shape
        return inp.view(batch_size, channels * depth, height, width)
