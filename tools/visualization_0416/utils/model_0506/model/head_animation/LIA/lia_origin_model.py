import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.LIA.modules import *

class MotionEncoder(nn.Module):
    def __init__(self, latent_dim, size=512):
        super(MotionEncoder, self).__init__()

        # self.input_size = size
        # channel = [32, 64, 128, 256, 512, 512, 512, 512]
        # 128, 128, 128 -> 256, 64, 64 -> 512, 16, 16 -> 512, 8, 8
        # -> 512, 4, 4 -> 512, 2, 2 -> 512, 1, 1 -> 512
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }
        log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channels[size], 1))

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, latent_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[x],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[x])

    def manual_forward(self, x):
        res = []
        h = x
        res = []
        for conv in self.convs:
            h = conv(h)
            res.append(h)
        res = res[::-1]
        feats = res[2:] # from 8x8 to 512x512
        latent_code = res[0]
        # [B * T, D]
        latent_code = latent_code.view(x.size(0), -1)
        return latent_code, feats

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

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

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[x],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[x])

    def manual_forward(self, x):
        h = x
        res = []
        for conv in self.convs:
            h = conv(h)
            res.append(h)
        feats = res[::-1][1:] # from 8x8 to 512x512
        return feats


class FaceGenerator(nn.Module):
    def __init__(self, size, latent_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(FaceGenerator, self).__init__()

        self.size = size
        self.latent_dim = latent_dim

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4]) # 512, 4, 4
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, latent_dim, blur_kernel=blur_kernel)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            # print(i, 2 ** i, in_channel, out_channel)
            # import pdb; pdb.set_trace()
            self.convs.append(StyledConv(in_channel, out_channel, 3, latent_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, latent_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, latent_dim))

            self.to_flows.append(ToFlow(out_channel, latent_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, tgt_latent, ref_feats):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[tgt_latent, ref_feats],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[tgt_latent, ref_feats])
        # return self.manual_forward(*[tgt_latent, ref_feats])

    def manual_forward(self, tgt_latent, ref_feats):
        bs = tgt_latent.size(0)

        inject_index = self.n_latent
        latent = tgt_latent.unsqueeze(1).repeat(1, inject_index, 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])
        # print('0', out.shape)

        i = 1
        for conv1, conv2, to_rgb, to_flow, feat in zip(self.convs[::2], self.convs[1::2], self.to_rgbs,
                                                       self.to_flows, ref_feats):
            out = conv1(out, latent[:, i])
            out = conv2(out, latent[:, i + 1])
            if out.size(2) == 8:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat)
                skip = to_rgb(out_warp)
            else:
                out_warp, out, skip_flow = to_flow(out, latent[:, i + 2], feat, skip_flow)
                skip = to_rgb(out_warp, skip)
            i += 2
            # print(i, out.shape, skip.shape, feat.shape)

        img = skip
        # import pdb; pdb.set_trace()
        return img

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.size = size

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[input],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[input])
        # return self.manual_forward(*[input])

    def manual_forward(self, input):
        out = self.convs(input)
        batch, channel, height, width = out.shape

        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
