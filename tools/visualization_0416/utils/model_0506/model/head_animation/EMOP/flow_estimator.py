import math
import torch
from torch import nn
from torch.nn import functional as F

import sys
from pathlib import Path

from model.head_animation.EMOP.warp_generator_resnet import WarpGenerator
from model.head_animation.EMOP.vpn_resblocks import VPN_ResBlocks
from model.head_animation.EMOP.unet_3d import Unet3D

from model.head_animation.EMOP.spectral_norm import apply_sp_to_nets
from model.head_animation.EMOP.utils import apply_ws_to_nets

class FlowEstimator(nn.Module):
    def __init__(self, cfg):
        super(FlowEstimator, self).__init__()

        # Operator that transform exp_emb to extended exp_emb (to match idt_emb size)
        self.pose_unsqueeze_nw = nn.Linear(cfg.exp_embedder.lpe_output_channels_expression, cfg.face_encoder.gen_max_channels * cfg.exp_embedder.embed_size ** 2, bias=False)

        # Operator that combine idt_imb and extended exp_emb together (a "+" sign of a scheme)
        self.warp_embed_head_orig_nw = nn.Conv2d(
            in_channels=cfg.face_encoder.gen_max_channels,
            out_channels=cfg.face_encoder.gen_max_channels,
            kernel_size=(1, 1),
            bias=False)  ## 相加之后再加一层变换，之前缺少这层

        self.src2ref = WarpGenerator(cfg.warp_generator)
        self.ref2tgt = WarpGenerator(cfg.warp_generator)

        self.volume_source_nw = VPN_ResBlocks(cfg.vpn_resblocks)

        # Net that process volume after first duble-warping
        self.volume_process_nw = Unet3D(cfg.unet3d)

        self.grid_sample = lambda inputs, grid: F.grid_sample(inputs.float(), grid.float(), padding_mode=cfg.grid_sample_padding_mode)

        self.warp_size = [cfg.face_encoder.gen_max_channels, cfg.exp_embedder.embed_size, cfg.exp_embedder.embed_size]
        self.volume_size = [cfg.face_encoder.latent_volume_channels, cfg.face_encoder.latent_volume_depth, cfg.face_encoder.latent_volume_size, cfg.face_encoder.latent_volume_size]

        apply_sp_to_nets(self)
        apply_ws_to_nets(self)


    def forward(self, data_dict):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[data_dict],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[data_dict])

    def predict_warping(self, data_dict):
        warp_source_embed = self.pose_unsqueeze_nw(data_dict['source_exp_embed']).view(-1, self.warp_size[0], self.warp_size[1], self.warp_size[2]) # Bx512x4x4
        warp_source_embed_orig = self.warp_embed_head_orig_nw((warp_source_embed + data_dict['idt_embed']) * 0.5) # Bx512x4x4
        source_warp_embed_dict = {'orig': warp_source_embed_orig.view(-1, self.warp_size[0], self.warp_size[1] ** 2), 'ada_v': data_dict['source_exp_embed']}
        data_dict['source_warp'], data_dict['source_delta_xy'] = self.src2ref(source_warp_embed_dict) # predict warping field from source embedding

        warp_target_embed = self.pose_unsqueeze_nw(data_dict['target_exp_embed']).view(-1, self.warp_size[0], self.warp_size[1], self.warp_size[2])
        warp_target_embed_orig = self.warp_embed_head_orig_nw((warp_target_embed + data_dict['idt_embed']) * 0.5)
        target_warp_embed_dict = {'orig': warp_target_embed_orig.view(-1, self.warp_size[0], self.warp_size[1] ** 2), 'ada_v': data_dict['target_exp_embed']}
        data_dict['target_warp'], data_dict['target_delta_uv'] = self.ref2tgt(target_warp_embed_dict) # predict warping field from target embedding

        data_dict['target_warp_embed_dict'] = target_warp_embed_dict
        return data_dict

    def predict_target2canonical_warping(self, data_dict):
        target_warp_embed_dict = data_dict['target_warp_embed_dict']
        target2ref_warp, _ = self.src2ref(target_warp_embed_dict) # predict warping field from target embedding
        return target2ref_warp

    def manual_forward(self, data_dict):
        b = data_dict['source_exp_embed'].size(0)
        # Predict warping from src and tgt expression embedding
        data_dict = self.predict_warping(data_dict)

        # Reshape latents into 3D volume
        c, d, s, s = self.volume_size
        latent_volume = data_dict['latent_volume']
        latent_volume = latent_volume.view(b, c, d, s, s)
        latent_volume = self.volume_source_nw(latent_volume)

        # Warp from source pose
        embed_dict = {}
        warp_latent_volume = self.grid_sample(self.grid_sample(latent_volume, data_dict['source_rotation_warp']), data_dict['source_warp'])
        canonical_latent_volume = self.volume_process_nw(warp_latent_volume, embed_dict)

        # Warp to target pose
        aligned_target_volume = self.grid_sample(self.grid_sample(canonical_latent_volume, data_dict['target_warp']), data_dict['target_rotation_warp'])
        aligned_target_volume = aligned_target_volume.view(b, c * d, s, s)

        data_dict['target_volume'] = aligned_target_volume

        if 'ref_matching' in data_dict and data_dict['ref_matching'] > 0:
            # import pdb; pdb.set_trace()

            target2ref_warp = self.predict_target2canonical_warping(data_dict)
            _latent_volume = self.volume_source_nw(data_dict['latent_volume_target'].view(b, c, d, s, s))
            _latent_volume = self.grid_sample(self.grid_sample(_latent_volume, data_dict['target_inv_rotation_warp']), target2ref_warp)
            data_dict['canonical_volume_from_tgt'] = self.volume_process_nw(_latent_volume, embed_dict)
            data_dict['canonical_volume_from_src'] = canonical_latent_volume

        return data_dict
