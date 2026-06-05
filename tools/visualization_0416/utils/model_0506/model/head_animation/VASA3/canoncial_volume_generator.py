import torch
import torch.nn as nn
from model.head_animation.VASA3.building_blocks import USE_BIAS, ResBlock3d

class CanonicalVolumeGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        num_channels_per_group = 32
        self.downblock1 = nn.Sequential(
            ResBlock3d(96, 192, num_channels_per_group),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.skip1 = nn.Sequential(
            ResBlock3d(192, 192, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        )
        self.downblock2 = nn.Sequential(
            ResBlock3d(192, 384, num_channels_per_group),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.bottleneck1 = ResBlock3d(384, 512, num_channels_per_group)
        self.bottleneck2 = ResBlock3d(512, 512, num_channels_per_group)
        self.upblock1 = nn.Sequential(
            ResBlock3d(512, 384, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        )
        self.skip2 = nn.Sequential(
            ResBlock3d(384, 384, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        )

        self.upblock2 = nn.Sequential(
            ResBlock3d(384, 192, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        )

        self.final_layer = nn.Sequential(
            ResBlock3d(192, 96, num_channels_per_group),
            nn.GroupNorm(96 // num_channels_per_group, 96, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 96, kernel_size=3, padding=1, bias=USE_BIAS),
        )

    def forward(self, inp: torch.Tensor):
        down_out1 = self.downblock1(inp)
        down_out2 = self.downblock2(down_out1)

        bottleneck_out1 = self.bottleneck1(down_out2)
        bottleneck_out2 = self.bottleneck2(bottleneck_out1)

        up_out1 = self.upblock1(bottleneck_out2 + bottleneck_out1)
        up_out2 = self.upblock2(up_out1 + self.skip2(down_out2))


        out = self.final_layer(up_out2 + self.skip1(down_out1))

        # print(down_out2.shape, up_out1.shape)
        # print(down_out1.shape, up_out2.shape)


        return out
