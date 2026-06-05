import torch
from torch import nn
from torch import optim
from torch.cuda import amp
import torch.nn.functional as F
import math
import numpy as np
import itertools
import copy
from torch.cuda import amp
from scipy import linalg
from model.head_animation.EMOP.face_decoder import Generator
from model.head_animation.EMOP.face_decoder_spade import SPADEGenerator
from model.head_animation.EMOP.spectral_norm import apply_sp_to_nets
from model.head_animation.EMOP.utils import apply_ws_to_nets


class FaceGenerator(nn.Module):
    def __init__(self, cfg):
        super(FaceGenerator, self).__init__()

        use_spade = cfg.get('use_spade', False)
        if use_spade:
            self.decoder = SPADEGenerator(cfg)
        else:
            self.decoder = Generator(cfg)

            apply_sp_to_nets(self)
            apply_ws_to_nets(self)

    def forward(self, data_dict, target_warp_embed_dict, aligned_target_volume):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[data_dict, target_warp_embed_dict, aligned_target_volume],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[data_dict, target_warp_embed_dict, aligned_target_volume])

    def manual_forward(self, data_dict, target_warp_embed_dict, aligned_target_volume):
        data_dict['pred_target_img'], _, _, _ = self.decoder(data_dict, target_warp_embed_dict, aligned_target_volume)
        return data_dict
