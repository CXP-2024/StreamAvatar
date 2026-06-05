import torch
import torch.nn as nn
import torch.nn.functional as F
from model.head_animation.VASA1.building_blocks import USE_BIAS, ResBlock3d, ReshapeTo3DLayer


def compute_warping_grid(
    rotation: torch.Tensor,  # (bs, 3, 3)
    translation: torch.Tensor,  # (bs, 3)
    nonrigid_pose: torch.Tensor,  # (bs, 3, 16, 16, 16)
    identity_grid: torch.Tensor,  # (16, 32, 32, 3) for a 256x256 input image.
    inverse: bool,
):
    batch_size = rotation.shape[0]
    depth, height, width = identity_grid.shape[:3]

    nonrigid_pose_sampled = F.grid_sample(
        input=nonrigid_pose,
        grid=identity_grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)  # (bs, 16, 32, 32, 3)

    identity_grid = identity_grid.view(-1, 3).unsqueeze(0)
    nonrigid_pose_sampled = nonrigid_pose_sampled.reshape(batch_size, -1, 3)

    # Map grid points in the driving frame to the source frame.
    if inverse:
        warping_grid = (rotation.transpose(1, 2).unsqueeze(1) @ (identity_grid - nonrigid_pose_sampled - translation.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
    else:
        warping_grid = (rotation.unsqueeze(1) @ identity_grid.unsqueeze(-1)).squeeze(-1) + translation.unsqueeze(1) + nonrigid_pose_sampled

    return warping_grid.view(batch_size, depth, height, width, 3)


class NonrigidPoseEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        num_channels_per_group = 8
        self.layers = nn.Sequential(
            # 2D conv layers
            nn.Conv2d(input_dim, 2048, kernel_size=1, bias=USE_BIAS),
            ReshapeTo3DLayer(out_depth=4),
            ResBlock3d(512, 256, num_channels_per_group),
            nn.Upsample(scale_factor=(2, 2, 2), mode="nearest"),
            ResBlock3d(256, 128, num_channels_per_group),
            nn.Upsample(scale_factor=(2, 2, 2), mode="nearest"),
            ResBlock3d(128, 64, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            ResBlock3d(64, 32, num_channels_per_group),
            nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
            nn.GroupNorm(32 // num_channels_per_group, 32, affine=not USE_BIAS),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 3, kernel_size=3, padding=1, bias=USE_BIAS),
        )

    def forward(self, inp: torch.Tensor):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(-1).unsqueeze(-1)

        return F.tanh(self.layers(inp))
