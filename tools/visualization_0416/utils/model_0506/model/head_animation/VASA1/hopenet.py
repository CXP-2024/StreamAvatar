# Copied from https://github.com/natanielruiz/deep-head-pose/blob/master/code/hopenet.py
# Apache license 2.0: https://github.com/natanielruiz/deep-head-pose/tree/master?tab=License-1-ov-file#readme

import torch
import torch.nn as nn
import math
import cv2


def headposeprediction_to_euler(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor):
    # Conversion from bins to radians is adapted from:
    # https://github.com/natanielruiz/deep-head-pose/blob/f7bbb9981c2953c2eca67748d6492a64c8243946/code/test_hopenet.py#L120
    # Apache 2.0 license.
    bin_indices = torch.arange(0, 66, device=yaw.device, dtype=yaw.dtype)

    # From bins to angles between [-99, +95]
    yaw = (yaw.softmax(1) * bin_indices).sum(1, keepdim=True) * 3 - 99
    pitch = (pitch.softmax(1) * bin_indices).sum(1, keepdim=True) * 3 - 99
    roll = (roll.softmax(1) * bin_indices).sum(1, keepdim=True) * 3 - 99

    # Angles to radians
    yaw = yaw * (torch.pi / 180)
    pitch = pitch * (torch.pi / 180)
    roll = roll * (torch.pi / 180)

    return yaw, pitch, roll


def euler_to_rotation_matrix(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor):
    # Yaw is around y-axis
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    roty = torch.stack(
        [
            torch.cat([cos_yaw, zeros, sin_yaw], dim=-1),
            torch.cat([zeros, ones, zeros], dim=-1),
            torch.cat([-sin_yaw, zeros, cos_yaw], dim=-1),
        ],
        dim=1,
    )

    # Pitch is around x-axis
    cos_pitch = torch.cos(pitch)
    sin_pitch = torch.sin(pitch)
    rotx = torch.stack(
        [
            torch.cat([ones, zeros, zeros], dim=-1),
            torch.cat([zeros, cos_pitch, -sin_pitch], dim=-1),
            torch.cat([zeros, sin_pitch, cos_pitch], dim=-1),
        ],
        dim=1,
    )

    # Roll is around z-axis
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    rotz = torch.stack(
        [
            torch.cat([cos_roll, -sin_roll, zeros], dim=-1),
            torch.cat([sin_roll, cos_roll, zeros], dim=-1),
            torch.cat([zeros, zeros, ones], dim=-1),
        ],
        dim=1,
    )

    return rotx @ roty @ rotz


def draw_axis(img, rotmat, tdx=None, tdy=None, size = 100):
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis
    x1 = size * rotmat[0, 0] + tdx
    y1 = size * rotmat[1, 0] + tdy

    # Y-Axis
    x2 = size * rotmat[0, 1] + tdx
    y2 = size * rotmat[1, 1] + tdy

    # Z-Axis
    x3 = size * rotmat[0, 2] + tdx
    y3 = size * rotmat[1, 2] + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)), (1,0,0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)), (0,1,0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)), (0,0,1), 3)

    return img


class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Alex: remove unused layer, necessary for DDP
        # Vestigial layer from previous experiments
        # self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll
