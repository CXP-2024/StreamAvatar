import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import math

from . import point_transforms



class HeadPoseRegressor(nn.Module):
    def __init__(self, model_path, size=128) -> None:
        super(HeadPoseRegressor, self).__init__()
        self.net = models.resnet18(num_classes=9)
        self.net.load_state_dict(torch.load(model_path, map_location='cpu'))
        # self.net.eval()
        # for param in self.net.parameters():
        #     param.requires_grad = False
        self.size = size

    # @torch.no_grad()
    def forward(self, x, return_srt=False):
        if x.shape[2] != self.size or x.shape[3] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bilinear')

        scale, rotation, translation = self.net(x).split([3, 3, 3], dim=1)
        # print(scale.shape, rotation.shape, translation.shape)
        thetas = point_transforms.get_transform_matrix(scale, rotation, translation)

        if return_srt:
            return thetas, scale, rotation, translation
        else:
            return thetas
