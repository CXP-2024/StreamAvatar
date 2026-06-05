import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head_animation.VASA3.building_blocks import USE_BIAS, ResBlock3d, ReshapeTo3DLayer, WSConv3d
import math

class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_features, eps=1e-5, affine=True):
        super(AdaptiveGroupNorm, self).__init__(num_groups, num_features, eps, False)
        self.num_features = num_features

        gen_max_channels, gen_embed_size = 512, 4
        self.u = nn.Parameter(torch.empty(num_features, gen_max_channels))
        self.v = nn.Parameter(torch.empty(gen_embed_size ** 2, 2))

        nn.init.uniform_(self.u, a=-math.sqrt(3 / gen_max_channels),  b=math.sqrt(3 / gen_max_channels))
        nn.init.uniform_(self.v, a=-math.sqrt(3 / gen_embed_size ** 2), b=math.sqrt(3 / gen_embed_size ** 2))

    def forward(self, inputs, condition_emb):
        outputs = super(AdaptiveGroupNorm, self).forward(inputs)

        param = self.u[None].matmul(condition_emb).matmul(self.v[None])
        ada_weight, ada_bias = param.split(1, dim=2)

        outputs = outputs * ada_weight[:, :, :, None, None] + ada_bias[:, :, :, None, None]
        return outputs


class ResBlock3dStar(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_channels_per_group: int, condition_dim: int):
        super().__init__()

        if in_channels != out_channels:
            self.skip_layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)
        else:
            self.skip_layer = lambda x: x

        self.agn1 = AdaptiveGroupNorm(in_channels // num_channels_per_group, in_channels)
        self.conv1 = WSConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)

        self.agn2 = AdaptiveGroupNorm(out_channels // num_channels_per_group, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=USE_BIAS)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inp, condition):
        x = self.relu(self.agn1(inp, condition))
        x = self.conv1(x)
        x = self.relu(self.agn2(x, condition))
        x = self.conv2(x)
        x = self.skip_layer(inp) + x
        return x


class NonrigidPoseEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        num_channels_per_group = 32
        app_fea_size = (512, 4, 4)

        self.conv1 = nn.Conv2d(app_fea_size[0], 2048, kernel_size=1, bias=USE_BIAS)
        self.reshap3d = ReshapeTo3DLayer(out_depth=4)
        self.resblock1 = ResBlock3dStar(512, 256, num_channels_per_group, input_dim)
        self.resblock2 = ResBlock3dStar(256, 128, num_channels_per_group, input_dim)
        self.resblock3 = ResBlock3dStar(128, 64, num_channels_per_group, input_dim)
        self.resblock4 = ResBlock3dStar(64, 32, num_channels_per_group, input_dim)
        self.gn = nn.GroupNorm(32 // num_channels_per_group, 32, affine=not USE_BIAS)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 3, kernel_size=3, padding=1, bias=USE_BIAS)

        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode="nearest")
        self.upsample2 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")

        self.extend_layer = nn.Linear(input_dim, app_fea_size[0] * app_fea_size[1] ** 2, bias=USE_BIAS)
        self.warp_layer = nn.Conv2d(in_channels=app_fea_size[0], out_channels=app_fea_size[0], kernel_size=(1, 1), bias=USE_BIAS)

        # Greate a meshgrid, which is used for warping calculation from deltas
        volumn_size, volumn_depth = 64, 16
        grid_s = torch.linspace(-1, 1, volumn_size)
        grid_z = torch.linspace(-1, 1, volumn_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        self.register_buffer('identity_grid', torch.stack([u, v, w], 0)[None])

    def forward(self, z: torch.Tensor, e_s: torch.Tensor):

        # ALign size to e_s
        z_emb = self.extend_layer(z).view(z.size(0), -1, 4, 4)
        warp_emb = self.warp_layer((z_emb + e_s) * 0.5)

        batch_size, c, h, w = e_s.shape
        condition = warp_emb.view(-1, c, h * w).clone()

        z = self.conv1(warp_emb)
        z = self.reshap3d(z)

        z = self.upsample(z)
        z = self.resblock1(z, condition)

        z = self.upsample(z)
        z = self.resblock2(z, condition)

        z = self.upsample2(z)
        z = self.resblock3(z, condition)

        z = self.upsample2(z)
        z = self.resblock4(z, condition)

        z = self.gn(z)
        z = self.relu(z)
        z = self.conv2(z)
        deltas = F.tanh(z)

        warp = (self.identity_grid + deltas).permute(0, 2, 3, 4, 1)

        return warp
