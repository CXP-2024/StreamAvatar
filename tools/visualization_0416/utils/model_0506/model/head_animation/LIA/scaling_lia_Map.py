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

        self.input_size = size
        channel = [32, 64, 128, 256, 512, 512, 512, 512]
        # 3, 512, 512 -> 32, 512, 512 -> 64, 256, 256 ->
        # 128, 128, 128 -> 256, 64, 64 -> 512, 32, 32 -> 512, 16, 16 -> 512, 8, 8
        # -> 512, 4, 4 -> 512, 1, 1 -> 512

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channel[0], 1))

        in_channel = channel[0]
        for i in range(1, len(channel)):
            out_channel = channel[i]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs.append(EqualConv2d(in_channel, latent_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        res = []
        h = x
        # gradiuent checkpoint ---------------------------
        def ckpt_wrapper(convs):
            def ckpt_forward(h):
                res = []
                for conv in convs:
                    h = conv(h)
                    res.append(h)
                return res
            return ckpt_forward

        if self.training:
            res = torch.utils.checkpoint.checkpoint( \
                ckpt_wrapper(self.convs), *[h],
                use_reentrant=False)
        else:
            res = ckpt_wrapper(self.convs)(*[h])
        # gradiuent checkpoint ---------------------------
        # res = []
        # for conv in self.convs:
        #     h = conv(h)
        #     res.append(h)
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

        channel = [32, 64, 128, 256, 512, output_channels]
        # 3, 512, 512 -> 32, 512, 512 -> 64, 256, 256 ->
        # 128, 128, 128 -> 256, 64, 64 -> 512, 32, 32 -> 512, 16, 16

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, channel[0], 1))

        in_channel = channel[0]
        for i in range(1, len(channel)):
            out_channel = channel[i]
            self.convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        h = x
        # gradiuent checkpoint ---------------------------
        def ckpt_wrapper(convs):
            def ckpt_forward(h):
                res = []
                for conv in convs:
                    h = conv(h)
                    res.append(h)
                return res
            return ckpt_forward

        if self.training:
            res = torch.utils.checkpoint.checkpoint( \
                ckpt_wrapper(self.convs), *[h],
                use_reentrant=False)
        else:
            res = ckpt_wrapper(self.convs)(*[h])
        # gradiuent checkpoint ---------------------------
        # res = []
        # for conv in self.convs:
        #     h = conv(h)
        #     res.append(h)
        feats = res[::-1][1:] # from 8x8 to 512x512
        return feats

class FaceGenerator(nn.Module):
    def __init__(self, size, latent_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(FaceGenerator, self).__init__()

        self.size = size
        self.latent_dim = latent_dim

        self.channels = {
            # 4: 512,
            # 8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[16], 16) # 512, 4, 4
        self.conv1 = StyledConv(self.channels[16], self.channels[16], 3, latent_dim, blur_kernel=blur_kernel)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        in_channel = self.channels[16]

        for i in range(5, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            # print(i, 2 ** i, in_channel, out_channel)
            # import pdb; pdb.set_trace()
            self.convs.append(StyledConv(in_channel, out_channel, 3, latent_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, latent_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, latent_dim))

            self.to_flows.append(ToFlow(out_channel, latent_dim))

            in_channel = out_channel

        self.n_latent = 16 * 16

    def forward(self, tgt_latent, ref_feats):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[tgt_latent, ref_feats],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[tgt_latent, ref_feats])

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
            if out.size(2) == 32:
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
