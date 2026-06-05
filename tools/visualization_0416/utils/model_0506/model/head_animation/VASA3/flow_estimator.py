import torch
import torch.nn as nn
import torch.nn.functional as F

from model.head_animation.VASA3.nonrigid_pose_encoder import NonrigidPoseEncoder
from model.head_animation.VASA3.util import compute_grid_points
from model.head_animation.VASA3.canoncial_volume_generator import CanonicalVolumeGenerator


class FlowEstimator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.src_nonrigid_pose_encoder = NonrigidPoseEncoder(input_dim=latent_dim)
        self.tgt_nonrigid_pose_encoder = NonrigidPoseEncoder(input_dim=latent_dim)
        self.canonical_volume_generator = CanonicalVolumeGenerator()

    def generate_canonical_feature_volume(self, feature_volume, global_descriptor, expression_code, rigid_pose, inverse, batch_idx):

        nonrigid_deformation = self.src_nonrigid_pose_encoder(expression_code, global_descriptor)
        rigid_transformation = self.compute_rigid_transformation(feature_volume, rigid_pose["rotation"], rigid_pose["translation"], inverse=inverse)

        # We warp the source volume to construct the canonical feature volume.
        exp_feature_volume = F.grid_sample(feature_volume.float(), rigid_transformation.float(), padding_mode='zeros')
        warping_feature_volume = F.grid_sample(exp_feature_volume.float(), nonrigid_deformation.float(), padding_mode='zeros')

        canonical_feature_volume = self.canonical_volume_generator(warping_feature_volume)

        return canonical_feature_volume, rigid_transformation, nonrigid_deformation


    def generate_warping_feature_volume(self, feature_volume, global_descriptor, expression_code, rigid_pose, inverse, batch_idx):

        nonrigid_deformation = self.tgt_nonrigid_pose_encoder(expression_code, global_descriptor)
        rigid_transformation = self.compute_rigid_transformation(feature_volume, rigid_pose["rotation"], rigid_pose["translation"], inverse=inverse)

        # We warp the canonical volume to construct the driving feature volume.
        exp_feature_volume = F.grid_sample(feature_volume.float(), nonrigid_deformation.float(), padding_mode='zeros')
        warping_feature_volume = F.grid_sample(exp_feature_volume.float(), rigid_transformation.float(), padding_mode='zeros')

        # if batch_idx == 200:
        #     import pdb; pdb.set_trace()

        return warping_feature_volume, rigid_transformation, nonrigid_deformation


    def compute_rigid_transformation(self, feature_volume, rotation, translation, inverse):

        identity_grid = compute_grid_points(feature_volume)
        identity_grid = identity_grid.view(-1, 3).unsqueeze(0)

        if inverse:
            rigid_transformation = (rotation.transpose(1, 2).unsqueeze(1) @ (identity_grid - translation.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        else:
            rigid_transformation =  (rotation.unsqueeze(1) @ (identity_grid).unsqueeze(-1)).squeeze(-1) + translation.unsqueeze(1)

        batch_size, _, depth, height, width = feature_volume.shape
        rigid_transformation = rigid_transformation.view(batch_size, depth, height, width, 3)

        return rigid_transformation

    def forward(self, src_dict, tgt_dict, batch_idx=None):

        global_descriptor = src_dict["global_descriptor"]

        ## generate_canonical_volume
        canonical_feature_volume, src_rigid_grid, src_delta_grid = self.generate_canonical_feature_volume(src_dict["feature_volume"], global_descriptor, src_dict["expression_code"], src_dict["rigid_pose"], inverse=True, batch_idx=batch_idx)

        ## warp canonical_feature_volume
        warped_driving_feature_volume, tgt_rigid_grid, tgt_delta_grid = self.generate_warping_feature_volume(canonical_feature_volume, global_descriptor, tgt_dict["expression_code"], tgt_dict["rigid_pose"], inverse=False, batch_idx=batch_idx)

        return canonical_feature_volume, warped_driving_feature_volume, src_delta_grid, tgt_delta_grid
