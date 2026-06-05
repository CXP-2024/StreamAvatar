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

        self.channels = [
            (32, 64, True), # 32, 512, 512 -> 64, 256, 256
            (64, 128, True), # 64, 256, 256 -> 128, 128, 128
            (128, 256, True), # 128, 128, 128 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 512, True), # 256, 64, 64 -> 512, 32, 32
            (512, output_channels, True), # 512, 32, 32 -> 512, 16, 16
        ]
        # 3, 512, 512 -> 32, 512, 512 -> 64, 256, 256 ->
        # 128, 128, 128 -> 256, 64, 64 -> 512, 32, 32 -> 512, 16, 16 -> 1024, 8, 8 -> 2048, 4, 4

        self.convs = nn.ModuleList()
        self.convs.append(ConvLayer(3, 32, 1))

        for in_channel, out_channel, downsample in self.channels:
            self.convs.append(ResBlock(in_channel, out_channel, downsample=downsample))

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

        # self.channels = {
        #     # 4: 512,
        #     # 8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }

        self.channels = [
            (latent_dim, 512, True), # 512, 16, 16 -> 512, 32, 32
            (512, 256, True), # 512, 32, 32 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 256, False), # 256, 64, 64 -> 256, 64, 64
            (256, 128, True), # 256, 64, 64 -> 128, 128, 128
            (128, 64, True), # 128, 128, 128 -> 64, 256, 256
            (64, 32, True), # 64, 256, 256 -> 32, 512, 512
        ]

        self.input = ConstantInput(latent_dim, 16) # 512, 4, 4
        self.conv1 = StyledConv(latent_dim, latent_dim, 3, latent_dim, blur_kernel=blur_kernel)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.to_flows = nn.ModuleList()

        for in_channel, out_channel, upsample in self.channels:
            self.convs.append(StyledConv(in_channel, out_channel, 3, latent_dim, upsample=upsample, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, latent_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, latent_dim, upsample=upsample))

            self.to_flows.append(ToFlow(out_channel, latent_dim, upsample=upsample))

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

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ConvLayer(in_channel, 3, 1)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip) if hasattr(self, "upsample") else skip
            out = out + skip

        return out


class ToFlow(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, feat, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        # warping
        xs = np.linspace(-1, 1, input.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)

        xs = torch.tensor(xs, requires_grad=False).float().unsqueeze(0).repeat(input.size(0), 1, 1, 1).to(input.device)

        if skip is not None:
            skip = self.upsample(skip) if hasattr(self, "upsample") else skip
            out = out + skip

        sampler = torch.tanh(out[:, 0:2, :, :])
        mask = torch.sigmoid(out[:, 2:3, :, :])
        flow = sampler.permute(0, 2, 3, 1) + xs

        feat_warp = F.grid_sample(feat, flow) * mask

        return feat_warp, feat_warp + input * (1.0 - mask), out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out
