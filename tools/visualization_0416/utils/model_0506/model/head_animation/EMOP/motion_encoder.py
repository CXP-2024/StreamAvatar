import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import torchvision

from model.head_animation.EMOP.expression_embedder import ExpressionEmbed
from model.head_animation.EMOP.head_pose_regressor import HeadPoseRegressor

from model.head_animation.EMOP.spectral_norm import apply_sp_to_nets
from model.head_animation.EMOP.utils import apply_ws_to_nets

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.use_mask_image = False # this setting is only for rigid_pose_encoder
        self.zero_to_one = cfg.zero_to_one
        self.expression_encoder = ExpressionEmbed(cfg.exp_embedder)
        self.rigid_pose_encoder = HeadPoseRegressor(cfg.head_pose_regressor_path, size=128)
        # for param in self.rigid_pose_encoder.parameters(): # we need finetune it on masked image
        #     param.requires_grad = True

        grid_s = torch.linspace(-1, 1, cfg.face_encoder.latent_volume_size)
        grid_z = torch.linspace(-1, 1, cfg.face_encoder.latent_volume_depth)
        w, v, u = torch.meshgrid(grid_z, grid_s, grid_s)
        e = torch.ones_like(u)
        self.identity_grid_3d = torch.stack([u, v, w, e], dim=3).view(1, -1, 4)

        apply_sp_to_nets(self)
        apply_ws_to_nets(self)

    def get_motion_latent(self, img, src):
        rotation_warp, theta, rotation = self.predict_pose(img, src)
        ## TODO: predict expression
        # motion_latent: rotation, scale, translation, expression
        return motion_latent

    def predict_pose(self, img, src):
        b, d, s = img.shape[0], 16, 64

        if not self.zero_to_one:
            img = (img + 1) / 2
        theta, scale, rotation, translation = self.rigid_pose_encoder.forward(img, return_srt=True)

        grid = self.identity_grid_3d.repeat_interleave(b, dim=0).to(img.device)
        if src:
            inv_source_theta = theta.float().inverse().type(theta.type())
            rotation_warp = grid.bmm(inv_source_theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)
        else:
            rotation_warp = grid.bmm(theta[:, :3].transpose(1, 2)).view(-1, d, s, s, 3)

        # rotation_warp = rotation_warp.detach()
        # theta = theta.detach()
        # rotation = rotation.detach()
        rotation_warp = rotation_warp.detach()

        return rotation_warp, theta, rotation, scale, translation

    def predict_expression(self, data_dict):
        data_dict = self.expression_encoder(data_dict) ## align src and tgt image, then predict expression code
        return data_dict

    def forward(self, data_dict):
        if self.training:
            return torch.utils.checkpoint.checkpoint( \
                    self.manual_forward, *[data_dict],
                    use_reentrant=False)
        else:
            return self.manual_forward(*[data_dict])

    def manual_forward(self, data_dict):
        # predict pose
        if self.use_mask_image and 'masked_source_img' in data_dict:
            src_img = data_dict['masked_source_img']
            tgt_img = data_dict['masked_target_img']
        else:
            src_img = data_dict['source_img']
            tgt_img = data_dict['target_img']

        data_dict['source_rotation_warp'], data_dict['source_theta'],  data_dict['source_rotation'], data_dict['source_scale'], data_dict['source_translation'] = self.predict_pose(src_img, src=True)
        data_dict['target_rotation_warp'], data_dict['target_theta'],  data_dict['target_rotation'], data_dict['target_scale'], data_dict['target_translation'] = self.predict_pose(tgt_img, src=False)

        # predict expression
        data_dict = self.predict_expression(data_dict)

        ## predict inverse transformation from target image for canonical volumn consistency supervision
        if 'ref_matching' in data_dict and data_dict['ref_matching'] > 0:
            data_dict['target_inv_rotation_warp'], _,  _, _, _ = self.predict_pose(tgt_img, src=True)


        return data_dict
