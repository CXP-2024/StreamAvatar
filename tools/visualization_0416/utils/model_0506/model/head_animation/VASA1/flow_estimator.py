import torch
import torch.nn as nn
import torch.nn.functional as F

from model.head_animation.VASA1.nonrigid_pose_encoder import NonrigidPoseEncoder, compute_warping_grid
from model.head_animation.VASA1.util import compute_grid_points


class FlowEstimator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.nonrigid_pose_encoder = NonrigidPoseEncoder(input_dim=latent_dim * 2)

    def generate_warping_feature_volume(self, feature_volume, global_descriptor, expression_code, rigid_pose, inverse):

        nonrigid_pose = self.nonrigid_pose_encoder(torch.cat((expression_code, global_descriptor), dim=1))

        # We warp the source volume to construct the canonical feature volume.
        identity_grid = compute_grid_points(feature_volume)

        warping_grid = compute_warping_grid(
            rotation=rigid_pose["rotation"],
            translation=rigid_pose["translation"],
            nonrigid_pose=nonrigid_pose,
            identity_grid=identity_grid,
            inverse=inverse,
        )

        warping_feature_volume = F.grid_sample(
            feature_volume,
            warping_grid,
            mode="bilinear",
            align_corners=False,
        )

        #### TODO: Add volume_refiner ?
        # if self.use_canonical_volume_refiner:
        #     canonical_feature_volume = self.canonical_volume_refiner(canonical_feature_volume)

        return warping_feature_volume

    def forward(self, src_dict, tgt_dict):

        global_descriptor = src_dict["global_descriptor"]

        ## generate_canonical_volume
        canonical_feature_volume = self.generate_warping_feature_volume(src_dict["feature_volume"], global_descriptor, src_dict["expression_code"], src_dict["rigid_pose"], inverse=False)

        ## warp canonical_feature_volume
        warped_driving_feature_volume = self.generate_warping_feature_volume(canonical_feature_volume, global_descriptor, tgt_dict["expression_code"], tgt_dict["rigid_pose"], inverse=True)

        return canonical_feature_volume, warped_driving_feature_volume
