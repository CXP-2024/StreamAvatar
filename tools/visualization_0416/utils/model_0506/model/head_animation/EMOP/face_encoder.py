import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import models
from torch.cuda import amp
import math

from model.head_animation.EMOP.appearance_encoder import AppearanceEncoder
from model.head_animation.EMOP.identity_embedder import IdtEmbed
from model.head_animation.EMOP.spectral_norm import apply_sp_to_nets
from model.head_animation.EMOP.utils import apply_ws_to_nets

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FaceEncoder(nn.Module):
    def __init__(self, cfg):
        super(FaceEncoder, self).__init__()

        self.app_encoder = AppearanceEncoder(cfg.face_encoder)
        self.idt_encoder = IdtEmbed(cfg.idt_embedder)

        apply_sp_to_nets(self)
        apply_ws_to_nets(self)

    def forward(self, source_img):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[source_img],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[source_img])

    def manual_forward(self, source_img):

        latent_volume = self.app_encoder(source_img)
        idt_embedding = self.idt_encoder(source_img)

        return latent_volume, idt_embedding
