import torch
from torch import nn
import sys
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.head_animation.head_animator import HeadAnimatorModule
from model.head_animation.VASA1.hopenet import headposeprediction_to_euler, euler_to_rotation_matrix
from utils import instantiate

class HeadAnimator3DModule(HeadAnimatorModule):
    def __init__(self, config):
        super().__init__(config)

    def motion_encode(self, masked_img, img):
        latent_code, rigid_pose = self.motion_encoder(masked_img, img)
        return latent_code, rigid_pose

    def forward(self, source_img, target_img, masked_source_img, masked_target_img, batch_idx=None):
        [feature_volume,  global_descriptor]= self.face_encoder(source_img)

        if self.using_hybrid_mask:
            tgt_latent, tgt_rigid_pose = self.motion_encoder(masked_target_img, target_img) # project target image to latent space
            src_latent, src_rigid_pose = self.motion_encoder(masked_source_img, source_img) # project source image to latent space
        else:
            tgt_latent, tgt_rigid_pose = self.motion_encoder(target_img, target_img) # project target image to latent space
            src_latent, src_rigid_pose = self.motion_encoder(source_img, source_img) # project source image to latent space

        src_dict = {}
        src_dict["feature_volume"] = feature_volume
        src_dict["global_descriptor"] = global_descriptor
        src_dict["expression_code"] = src_latent[:, :self.config.model.motion_encoder.latent_dim]
        src_dict["rigid_pose"] = src_rigid_pose

        tgt_dict = {}
        tgt_dict["expression_code"] = tgt_latent[:, :self.config.model.motion_encoder.latent_dim]
        tgt_dict["rigid_pose"] = tgt_rigid_pose

        canonical_feature_volume, warped_driving_feature_volume, src_delta_grid, tgt_delta_grid = self.flow_estimator(src_dict, tgt_dict, batch_idx)

        recon_img = self.face_generator(warped_driving_feature_volume)

        # if batch_idx == 20:
        #     recon_src_img = self.face_generator(feature_volume)
        #     recon_can_img = self.face_generator(canonical_feature_volume)

        #     recon_src_img = 255 * (recon_src_img.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2
        #     recon_img_vis = 255 * (recon_img.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2
        #     recon_img_can = 255 * (recon_can_img.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) / 2
        #     cv2.imwrite("recon_src_img.png", recon_src_img[0][:,:,::-1])
        #     cv2.imwrite("recon_tgt_img.png", recon_img_vis[0][:,:,::-1])
        #     cv2.imwrite("recon_can_img.png", recon_img_can[0][:,:,::-1])
        #     import pdb; pdb.set_trace()

        [feature_volume_tgt,  global_descriptor_tgt]= self.face_encoder(target_img)
        tgt_dict["feature_volume"] = feature_volume_tgt
        tgt_dict["global_descriptor"] = global_descriptor_tgt
        canonical_feature_volume_tgt, warped_driving_feature_volume_tgt, _, _ = self.flow_estimator(tgt_dict, tgt_dict, batch_idx)
        feature_volume = {}
        feature_volume["canonical_feature_volume"] = canonical_feature_volume
        feature_volume["warped_driving_feature_volume"] = warped_driving_feature_volume
        feature_volume["canonical_feature_volume_tgt"] = canonical_feature_volume_tgt
        feature_volume["warped_driving_feature_volume_tgt"] = warped_driving_feature_volume_tgt
        feature_volume["feature_volume_tgt"] = feature_volume_tgt
        feature_volume["src_delta_grid"] = src_delta_grid
        feature_volume["tgt_delta_grid"] = tgt_delta_grid

        return recon_img, tgt_rigid_pose, feature_volume

    def compute_headpose_loss(self, tgt_image, tgt_rigid_pose):
        if tgt_image.min() <= -0.5:
            tgt_image_org = (tgt_image + 1) / 2 #[-1, 1] -> [0, 1]

        with torch.no_grad():
            gt_rotation_matrix = euler_to_rotation_matrix(*headposeprediction_to_euler(*self.motion_encoder.hopenet(self.motion_encoder.hopenet_test_transformations(tgt_image_org))))

        headpose_loss = torch.acos(
            # Clip is needed to avoid NAN values.
            torch.clip(
                # (trace(m1, m2.T) - 1) / 2
                input=torch.linalg.diagonal(gt_rotation_matrix @ tgt_rigid_pose["rotation"].transpose(1, 2)).sum(-1) * 0.5 - 0.5,
                min=-0.999999,
                max=0.999999,
            )
        ).mean() * self.config.loss.l_w_headpose

        return headpose_loss

    def compute_feat_loss(self, feature_volume):
        canonical_feature_volume = feature_volume["canonical_feature_volume"]
        warped_driving_feature_volume = feature_volume["warped_driving_feature_volume"]

        canonical_feature_volume_tgt = feature_volume["canonical_feature_volume_tgt"]
        warped_driving_feature_volume_tgt = feature_volume["warped_driving_feature_volume_tgt"]
        feature_volume_tgt = feature_volume["feature_volume_tgt"].detach()

        feat_loss = F.mse_loss(canonical_feature_volume, canonical_feature_volume_tgt) + F.mse_loss(warped_driving_feature_volume, feature_volume_tgt)
        return feat_loss

    def compute_driving_feat_loss(self, feature_volume):
        warped_driving_feature_volume = feature_volume["warped_driving_feature_volume"]
        feature_volume_tgt = feature_volume["feature_volume_tgt"].detach()

        feat_loss = F.mse_loss(warped_driving_feature_volume, feature_volume_tgt)
        return feat_loss

    def compute_non_rigid_warping_loss(self, feature_volume):
        # import pdb; pdb.set_trace()
        batch_size, channel, depth, height, width = feature_volume["src_delta_grid"].shape
        src_delta_grid = feature_volume["src_delta_grid"].reshape(batch_size, -1)
        tgt_delta_grid = feature_volume["tgt_delta_grid"].reshape(batch_size, -1)
        delta_grid_penalty = torch.mean(torch.abs(src_delta_grid).sum(dim=1)) + torch.mean(torch.abs(tgt_delta_grid).sum(dim=1))
        return delta_grid_penalty


    def compute_loss(self, img_target, img_target_recon, face_mask=None, tgt_rigid_pose=None, feature_volume=None):

        vgg_loss, l1_loss, face_loss, face_l1_loss = self.compute_base_loss(img_target, img_target_recon, face_mask)

        loss = vgg_loss + l1_loss + face_loss + face_l1_loss
        loss_dict = {'loss': loss, 'l1_loss': l1_loss, 'face_l1_loss': face_l1_loss, 'vgg_loss': vgg_loss, 'face_loss': face_loss}

        if not self.config.model.motion_encoder.use_gt_rotation:
            # TODO: check this loss works or not
            assert self.config.loss.l_w_headpose > 0
            headpose_loss = self.config.loss.l_w_headpose * self.compute_headpose_loss(img_target, tgt_rigid_pose)
            loss_dict['loss'] += headpose_loss
            loss_dict['headpose_loss'] = headpose_loss

        if feature_volume is not None and self.config.loss.l_w_feat > 0:
            feat_loss = self.config.loss.l_w_feat * self.compute_feat_loss(feature_volume)
            loss_dict['loss'] += feat_loss
            loss_dict['feat_loss'] = feat_loss

        if feature_volume is not None and self.config.loss.l_w_dri_feat > 0:
            dri_feat_loss = self.config.loss.l_w_dri_feat * self.compute_driving_feat_loss(feature_volume)
            loss_dict['loss'] += dri_feat_loss
            loss_dict['dri_feat_loss'] = dri_feat_loss

        if feature_volume is not None and self.config.loss.l_w_warping_penalty > 0:
            warping_penalty_loss = self.config.loss.l_w_warping_penalty * self.compute_non_rigid_warping_loss(feature_volume)
            loss_dict['loss'] += warping_penalty_loss
            loss_dict['warping_penalty_loss'] = warping_penalty_loss

        return loss_dict

if __name__ == "__main__":
    from model.head_animation.VASA1.motion_encoder import MotionEncoder
    from torchsummaryX import summary

    IMAGE_SIZE = 512
    latent_dim = 512

    encoder = MotionEncoder(latent_dim=latent_dim, size=IMAGE_SIZE)
    summary(encoder, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
