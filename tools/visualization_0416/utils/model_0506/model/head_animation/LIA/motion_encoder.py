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

        self.size = size

        if self.size==256:
            channel = [64, 128, 256, 512, 512, 512, 512]

        elif self.size==512:
            channel = [32, 64, 128, 256, 512, 512, 512, 512]

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
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[x],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[x])

    def manual_forward(self, x):

        if x.size(-1) != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear')

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


class MotionEncoderLight(nn.Module):
    def __init__(self, latent_dim, size=512):
        super().__init__()

        self.size = size
        self.layers = resnet18(pretrained=False, num_classes=latent_dim) # 11.4M

    def forward(self, x):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[x],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[x])

    def manual_forward(self, x):
        if x.size(-1) != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear')
        out = self.layers(x)
        return out, None
