import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Any, Dict, Tuple
from utils import instantiate
import cv2
from PIL import Image
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp


class Encoder2D(nn.Module):
    def __init__(self, output_channels=512):
        super(Encoder2D, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(3, output_channels, 4, 2, 1),  # 512x512 -> 256x256
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 4, 2, 1),  # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 4, 2, 1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1),  # Final Convolutional layer before residuals
            ResidualBlock(output_channels),  # Residual block 1
            ResidualBlock(output_channels),   # Residual block 2
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder2D(nn.Module):
    def __init__(self, input_dim=512):
        super(Decoder2D, self).__init__()

        self.fea_map_size=16

        self.block = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, 1, 1),  # Initial convolution in the decoder
            ResidualBlock(input_dim),  # Residual block 1
            ResidualBlock(input_dim),  # Residual block 2
            nn.ConvTranspose2d(input_dim, input_dim, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, 4, 2, 1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, 4, 2, 1),  # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, input_dim, 4, 2, 1),  # 128x128 -> 256x256
            nn.ReLU(),
            nn.ConvTranspose2d(input_dim, 3, 4, 2, 1)  # 256x256 -> 512x512
        )

    def forward(self, x):
        x_hat = self.block(x)

        return x_hat


class Encoder(Encoder2D):
    def __init__(self, output_channels=512):
        super().__init__(output_channels)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class Decoder(Decoder2D):
    def __init__(self, input_dim=512):
        super().__init__(input_dim)

        self.fc = nn.Linear(input_dim, input_dim*self.fea_map_size*self.fea_map_size)

    def forward(self, x):
        x = self.fc(x.view(x.size(0), -1))
        x = x.view(x.size(0), 512, self.fea_map_size, self.fea_map_size)
        x_hat = self.block(x)

        return x_hat
