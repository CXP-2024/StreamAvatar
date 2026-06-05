import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.VASA1.building_blocks import USE_BIAS, ResBlock3d, ReshapeTo3DLayer, WSConv3d, ReshapeTo2DLayer
import math
# from .util import *
from model.head_animation.LIA_3d.util import *

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


class FaceGenerator(nn.Module):
    def __init__(self, size, reshape_channel, group_norm_channel, latent_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1, outputsize=512, flag_estimate_occlusion_map=False):
        super(FaceGenerator, self).__init__()

        self.size = size
        self.latent_dim = latent_dim
        self.flag_estimate_occlusion_map = flag_estimate_occlusion_map

        ## warping field generator
        num_channels_per_group = group_norm_channel
        app_fea_size = (512, 4, 4)
        input_dim = app_fea_size[0]

        self.extend_layer = nn.Linear(latent_dim, app_fea_size[0] * app_fea_size[1] ** 2, bias=USE_BIAS)
        self.conv1 = nn.Conv2d(latent_dim, 2048, kernel_size=1, bias=USE_BIAS)
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

        d, h, w = 16, 64, 64
        grids = torch.meshgrid(
            torch.linspace(-1, 1, d),
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij"
        )
        self.identity_grid = torch.stack(grids, dim=-1).flip(-1)

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(reshape_channel*d, 1, kernel_size=7, padding=3)

        ### generator
        # Projection layers
        self.projection = nn.Sequential(
            ReshapeTo2DLayer(),
            SameBlock2d(reshape_channel * d, 256, kernel_size=(3, 3), padding=(1, 1), lrelu=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1)
        )

        input_channels = 256
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256
        out_channels = 64

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)
        self.up = nn.Upsample(scale_factor=2)

        self.conv_img = nn.Sequential(
            nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2)
        )
        self.final_activation = nn.Tanh()

    def forward(self, tgt_latent, ref_feats):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[tgt_latent, ref_feats],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[tgt_latent, ref_feats])

    def manual_forward(self, tgt_latent, ref_feats):
        bs = tgt_latent.size(0)

        # generate warping field
        z_emb = self.extend_layer(tgt_latent).view(tgt_latent.size(0), -1, 4, 4)

        batch_size, c, h, w = z_emb.shape
        condition = z_emb.view(-1, c, h * w).clone()

        z = self.conv1(z_emb)
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
        deltas = F.tanh(z).permute(0, 2, 3, 4, 1)

        warping_field = self.identity_grid[None].to(tgt_latent.device) + deltas
        warping_feature_volume = F.grid_sample(ref_feats, warping_field, mode="bilinear", align_corners=False)

        # decoding
        seg = self.projection(warping_feature_volume) # Bx256x64x64

        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = warping_feature_volume.shape
            warping_feature_volume_reshape = warping_feature_volume.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(warping_feature_volume_reshape))  # Bx1x64x64
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
        x = self.final_activation(x)  # Bx3xHxW

        return x
