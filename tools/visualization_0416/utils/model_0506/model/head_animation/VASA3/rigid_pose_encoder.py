import torch
import torch.nn as nn
import torch.nn.functional as F

from model.head_animation.VASA3.resnet18 import Resnet18
from model.head_animation.VASA3.hopenet import euler_to_rotation_matrix, headposeprediction_to_euler


def rotation_6d_to_matrix(rotation_6d: torch.Tensor):
    # Via Gram-Schmidt orthogonalization.
    a1, a2 = rotation_6d[..., 0:3], rotation_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-2)


class RigidPoseEncoder(nn.Module):
    def __init__(self, use_gt_rotation, gt_rotation_callback):
        super().__init__()
        self.use_gt_rotation = use_gt_rotation
        if use_gt_rotation:
            self.gt_rotation_callback = gt_rotation_callback
            self.net = Resnet18(input_dim=3, output_dim=3, add_fc=False)  # 3D translation.
        else:
            self.net = Resnet18(input_dim=3, output_dim=6 + 3, add_fc=False)  # 6D rotation and 3D translation.

    def forward(self, masked_inp, inp):

        out = self.net(masked_inp)

        if not self.use_gt_rotation:
            rot_out = rotation_6d_to_matrix(out[:, 0:6])
            trans_out = F.tanh(out[:, 6:9])
        else:
            rot_out = self.gt_rotation_callback(inp)
            trans_out = F.tanh(out[:, 0:3])

        return {
            "rotation": rot_out,
            "translation": trans_out,
        }
