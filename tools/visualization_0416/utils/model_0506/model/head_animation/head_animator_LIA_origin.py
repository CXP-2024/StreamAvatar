import numpy as np
import torch
from torch import nn
import sys
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
import math

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.lightning.base_modules import BaseModule
from utils import instantiate
from model.head_animation.VASA3.building_blocks import *
from model.head_animation.VASA3.nonrigid_pose_encoder import AdaptiveGroupNorm
from model.modnets.modnet import MODNet

class HeadAnimatorModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.automatic_optimization = False
        self.config = config
        self.using_hybrid_mask = config.model.get("using_hybrid_mask", True)
        print(f'Using Hybird Mask: {self.using_hybrid_mask}')
        if not self.using_hybrid_mask:
            self.face_encoder = nn.Identity()

        self.criterion_recon = nn.L1Loss()
        self.criterion_masked_face_l1 = nn.L1Loss(reduction='none')

        self.l_w_recon = config.loss.l_w_recon
        self.l_w_vgg = config.loss.l_w_vgg
        self.l_w_face = config.loss.get("l_w_face", 0)
        self.l_w_gan = config.loss.get("l_w_gan", 0)
        self.l_w_face_l1 = config.loss.get("l_w_face_l1", 0)

        # support GAN training & normal training
        self.automatic_optimization = False

        if 'VASA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'VASA'

        if 'LIA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'LIA'

        print(f'Using {self.model_name} for Head Animation')
        self.init_model()

    def init_model(self):
        config = self.config
        self.motion_encoder = instantiate(config.model.motion_encoder)
        self.flow_estimator = instantiate(config.model.flow_estimator)
        self.face_generator = instantiate(config.model.face_generator)
        self.face_encoder = instantiate(config.model.face_encoder)
        if not self.config.model.get("using_hybrid_mask", True):
            self.face_encoder = nn.Identity()

        self.use_modnets = False
        if config.model.get("modnets", None) is not None:
            self.use_modnets = True
            self.modnet = MODNet(backbone_pretrained=False)
            self.modnet.eval()
            modnet_state_dict = torch.load(config.model.modnets.pretrained_weights)
            modnet_state_ckpt = {}
            for k, v in modnet_state_dict.items():
                modnet_state_ckpt[k.replace("module.", "")] = v
            self.modnet.load_state_dict(modnet_state_ckpt)
            for name, param in self.modnet.named_parameters():
                param.requires_grad = False

        if config.loss.l_w_vgg > 0 or config.loss.l_w_face > 0:
            # self.criterion_vgg = VGGLoss()
            self.criterion_vgg = instantiate(config.model.vgg_loss)
            for name, param in self.criterion_vgg.named_parameters():
                param.requires_grad = False
            self.criterion_vgg.eval()

        if config.loss.l_w_gan > 0:
            self.discriminator = instantiate(config.model.discriminator)

        if self.config.model.get('pretrained_ckpt', None) is not None:
            checkpoint = torch.load(self.config.model.pretrained_ckpt)["state_dict"]
            ckpt = {}
            for k, v in checkpoint.items():
                if 'motion_encoder' in k:
                    ckpt[k.replace('motion_encoder.', '')] = v
            self.motion_encoder.load_state_dict(ckpt, strict=True)

            ckpt = {}
            for k, v in checkpoint.items():
                if 'flow_estimator' in k:
                    ckpt[k.replace('flow_estimator.', '')] = v
            self.flow_estimator.load_state_dict(ckpt, strict=True)

            ckpt = {}
            for k, v in checkpoint.items():
                if 'face_generator' in k:
                    ckpt[k.replace('face_generator.', '')] = v
            self.face_generator.load_state_dict(ckpt, strict=True)
            if self.face_generator.freeze:
                for param in self.face_generator.parameters():
                    param.requires_grad = False


            ckpt = {}
            for k, v in checkpoint.items():
                if 'face_encoder' in k:
                    ckpt[k.replace('face_encoder.', '')] = v
            self.face_encoder.load_state_dict(ckpt, strict=True)
            if self.face_encoder.freeze:
                for name, param in self.face_encoder.named_parameters():
                    # Note: we only pretrain the VolumetricFieldEncoder
                    if 'global_descriptor_encoder' not in name:
                        param.requires_grad = False


    def configure_model(self):
        pass
    # def setup(self, stage=None):
    #     if stage == "fit" or stage is None:
    #         print('Model is initializing weights...')
    #         # self.initialize_weights()

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (AdaptiveGroupNorm)):
    #             pass

    #         # -------------------- Initialize convolutional layers (WSConv2d/WSConv3d) --------------------
    #         if isinstance(m, (WSConv2d, WSConv3d)):
    #             # Check if the weights need to be updated
    #             if m.weight.requires_grad:
    #                 # Adapt the initialization of WSConv to the scaling
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #                 m.weight.data.mul_(math.sqrt(2))  # Compensate for the scaling of the weights
    #             # Initialize the bias (if it exists and needs to be updated)
    #             if hasattr(m, 'bias') and m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0)

    #         # -------------------- Initialize ordinary convolutional layers (non-WSConv) --------------------
    #         elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
    #             if m.weight.requires_grad:
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0)

    #         # -------------------- Handle the last layer of the residual block's GroupNorm --------------------
    #         # Rule: Initialize the weight of the last GroupNorm in the main path to 0 (if the parameter needs to be updated)
    #         elif isinstance(m, (ResBlock2d, ResBlock3d, ResBasic, ResBottleneck)):
    #             # Find the last GroupNorm in the main path
    #             last_group_norm = None
    #             for layer in reversed(m.layers):
    #                 if isinstance(layer, nn.GroupNorm):
    #                     last_group_norm = layer
    #                     break
    #             # Initialize the weight of the last GroupNorm to 0 (only if the parameter needs to be updated)
    #             if last_group_norm is not None and last_group_norm.weight.requires_grad:
    #                 nn.init.constant_(last_group_norm.weight, 0)

    #             # Initialize the convolutional layer of the skip connection (if it exists and needs to be updated)
    #             if not isinstance(m.skip_layer, (nn.Identity, type(None))):
    #                 if isinstance(m.skip_layer, (nn.Conv2d, nn.Conv3d)):
    #                     if m.skip_layer.weight.requires_grad:
    #                         nn.init.kaiming_normal_(m.skip_layer.weight, mode='fan_in', nonlinearity='relu')
    #                 elif isinstance(m.skip_layer, nn.Sequential):
    #                     # Handle the convolutional layer in the skip connection (e.g. in ResBasic)
    #                     for subm in m.skip_layer:
    #                         if isinstance(subm, (nn.Conv2d, nn.Conv3d)):
    #                             if subm.weight.requires_grad:
    #                                 nn.init.kaiming_normal_(subm.weight, mode='fan_in', nonlinearity='relu')

    #         # -------------------- Initialize GroupNorm layers --------------------
    #         elif isinstance(m, nn.GroupNorm):
    #             # Only initialize the parameters that need to be updated
    #             if m.weight is not None and m.weight.requires_grad:
    #                 nn.init.constant_(m.weight, 1.0)
    #             if m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0)

    #         # -------------------- Initialize linear layers --------------------
    #         elif isinstance(m, nn.Linear):
    #             if m.weight.requires_grad:
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    #             if m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0)

    #         # -------------------- Handle Spectral Normalization parameters --------------------
    #         # If using spectral normalization, initialize the original weights instead of the parameterized weights
    #         if hasattr(m, 'parametrizations') and 'weight' in m.parametrizations:
    #             parametrization = m.parametrizations.weight[0]
    #             if hasattr(parametrization, 'original'):
    #                 original_weight = parametrization.original
    #                 if original_weight.requires_grad:
    #                     # Initialize the original weights (e.g. using Kaiming)
    #                     nn.init.kaiming_normal_(original_weight, mode='fan_in', nonlinearity='relu')

    def motion_encode(self, source_img):
        latent_code, pyramid_feat = self.motion_encoder(source_img)
        return latent_code, pyramid_feat

    def forward(self, source_img, target_img, masked_source_img, masked_target_img, batch_idx=None):
        if self.using_hybrid_mask:
            tgt_latent, _ = self.motion_encoder(masked_target_img) # project target image to reference latent space
            src_latent, _ = self.motion_encoder(masked_source_img) # project source image to reference latent space

            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space

            face_feat = self.face_encoder(source_img)
            recon_img = self.face_generator(tgt_latent, face_feat)
        else:
            tgt_latent, _ = self.motion_encoder(target_img) # project target image to reference latent space
            src_latent, face_feat = self.motion_encoder(source_img) # project source image to reference latent space

            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space
            recon_img = self.face_generator(tgt_latent, face_feat)
            # import pdb; pdb.set_trace()

        return recon_img, None, None

    def compute_base_loss(self, img_target, img_target_recon, face_mask=None):

        l1_loss = self.l_w_recon * self.criterion_recon(img_target_recon, img_target)

        # Perceptual Loss
        if self.l_w_vgg > 0:
            # img_target_recon = F.interpolate(img_target_recon, size=(256, 256), mode='bilinear', align_corners=False)
            # img_target = F.interpolate(img_target, size=(256, 256), mode='bilinear', align_corners=False)
            vgg_loss, vgg_loss_dict = self.criterion_vgg(img_target_recon, img_target)
            vgg_loss = self.l_w_vgg * vgg_loss.mean()
        else:
            vgg_loss = torch.zeros(1).to(self.device)

        # Facial Experssion Perceptual Loss
        if face_mask is not None and self.l_w_face > 0:
            face_loss, face_vgg_loss_dict = self.criterion_vgg(img_target_recon, img_target, face_mask)
            face_loss = self.l_w_face * face_loss.mean()
        else:
            face_loss = torch.zeros(1).to(self.device)

        if face_mask is not None and self.l_w_face_l1 > 0:
            face_l1_loss = self.criterion_masked_face_l1(img_target_recon*face_mask, img_target*face_mask)
            face_l1_loss = face_l1_loss.view(face_mask.size(0), -1).sum(-1) / face_mask.view(face_mask.size(0), -1).sum(-1)
            face_l1_loss = self.l_w_face_l1 * face_l1_loss.mean()
        else:
            face_l1_loss = torch.zeros(1).to(self.device)

        return vgg_loss, l1_loss, face_loss, face_l1_loss

    def compute_loss(self, img_target, img_target_recon, face_mask=None, tgt_rigid_pose=None, feature_volume=None):
        vgg_loss, l1_loss, face_loss, face_l1_loss = self.compute_base_loss(img_target, img_target_recon, face_mask)

        loss = vgg_loss + l1_loss + face_loss + face_l1_loss
        loss_dict = {'loss': loss, 'l1_loss': l1_loss, 'face_l1_loss': face_l1_loss, 'vgg_loss': vgg_loss, 'face_loss': face_loss}
        return loss_dict

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def prepare_datapair(self, batch):
        # when not zero_to_one, all of bellow is [-1, 1]
        masked_target_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        masked_past_frames = batch['pixel_values_past_frames']
        masked_target_vid = torch.cat([masked_past_frames, masked_target_vid], dim=1)
        masked_ref_img = batch['pixel_values_ref_img'] # b c h w

        # when not zero_to_one, all of bellow is [-1, 1]
        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([past_frames, target_vid_original], dim=1)

        # import pdb; pdb.set_trace()

        # construct ref-tgt pairs
        masked_ref_img = masked_ref_img[:,None].repeat(1, masked_target_vid.size(1), 1, 1, 1)
        masked_ref_img = rearrange(masked_ref_img, "b t c h w -> (b t) c h w")
        masked_target_vid = rearrange(masked_target_vid, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original.to(self.device)
        target_vid_original = target_vid_original.to(self.device)
        masked_ref_img = masked_ref_img.to(self.device)
        masked_target_vid = masked_target_vid.to(self.device)

        if self.use_modnets:
            with torch.no_grad():
                _, _, target_vid_original_mask = self.modnet((target_vid_original + 1.) / 2., True)
        # target_vid_original_mask is (b 1 h w), range is (0, 1).
        # Making to Binary mask: You can set value >= 0.5 to be 1 and value <= 0.5 to be 0.

        # visual -----------------------------------
        # import imageio
        # visual_list = []
        # for ref_img_original_i, target_vid_original_i, masked_ref_img_i, \
        #     masked_target_vid_i, target_vid_original_mask_i in zip(ref_img_original, target_vid_original, \
        #                                masked_ref_img, masked_target_vid, target_vid_original_mask):
        #     ref_img_original_i = (((ref_img_original_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     target_vid_original_i = (((target_vid_original_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     masked_ref_img_i = (((masked_ref_img_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     masked_target_vid_i = (((masked_target_vid_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     target_vid_original_mask_i[target_vid_original_mask_i < 0.5] = 0
        #     target_vid_original_mask_i = (target_vid_original_mask_i.repeat(3, 1, 1).cpu().numpy().transpose(1, 2, 0) * 255.).astype("uint8")
        #     visuals = np.concatenate([ref_img_original_i, target_vid_original_i, masked_ref_img_i, masked_target_vid_i, target_vid_original_mask_i], axis=1)
        #     visual_list.append(visuals)
        # import os
        # os.makedirs("visual_train_data", exist_ok=True)
        # imageio.mimwrite(f"./visual_train_data/{self.trainer.global_step}_{self.trainer.global_rank}.mp4", visual_list, fps=8)
        # video_path = batch["video_path"][0]
        # print(f"{video_path=}")

        # import imageio
        # visual_list = []
        # for ref_img_original_i, target_vid_original_i, masked_ref_img_i, \
        #     masked_target_vid_i in zip(ref_img_original, target_vid_original, \
        #                                masked_ref_img, masked_target_vid):
        #     ref_img_original_i = (((ref_img_original_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     target_vid_original_i = (((target_vid_original_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     masked_ref_img_i = (((masked_ref_img_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     masked_target_vid_i = (((masked_target_vid_i.cpu().numpy() + 1.) / 2.) * 255.).transpose(1, 2, 0).astype("uint8")
        #     visuals = np.concatenate([ref_img_original_i, target_vid_original_i, masked_ref_img_i, masked_target_vid_i], axis=1)
        #     visual_list.append(visuals)
        # import os
        # video_base = os.path.basename(video_path)
        # os.makedirs("GAN_DEBUG_VISUAL", exist_ok=True)
        # imageio.mimwrite(f"./GAN_DEBUG_VISUAL/{self.trainer.global_step}_{self.trainer.global_rank}_{video_base}", visual_list, fps=self.config.data.train_fps)
        # visual -----------------------------------


        return ref_img_original, target_vid_original, masked_ref_img, masked_target_vid

    def set_module_eval_train_state(self, state_is_g=True):
        if self.l_w_gan > 0:
            self.discriminator.train()
        self.motion_encoder.train()
        self.flow_estimator.train()
        self.face_generator.train()
        self.face_encoder.train()
        # if state_is_g:
        #     self.discriminator.eval()
        #     self.motion_encoder.train()
        #     self.flow_estimator.train()
        #     self.face_generator.train()
        #     self.face_encoder.train()
        # else:
        #     self.discriminator.train()
        #     self.motion_encoder.eval()
        #     self.flow_estimator.eval()
        #     self.face_generator.eval()
        #     self.face_encoder.eval()

    def _step(self, batch, batch_idx):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)

        if self.l_w_gan > 0: # using GAN training
            optimizer_g, optimizer_d = self.optimizers()
            ## train generator
            # toggle is same to set grad is true
            self.set_module_eval_train_state(True)
            self.toggle_optimizer(optimizer_g)
            # get reconstructed image
            predicted_img, tgt_rigid_pose, feature_volume = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)

            if self.l_w_face > 0 or self.l_w_face_l1 > 0:
                eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
                eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
                face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
                face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask, tgt_rigid_pose, feature_volume)

            else:
                loss_dict = self.compute_loss(target_vid_original, predicted_img, tgt_rigid_pose=tgt_rigid_pose, feature_volume=feature_volume)

            # adversarial loss
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss

            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

            # import pdb; pdb.set_trace()

            ## train discriminator
            self.set_module_eval_train_state(False)
            self.toggle_optimizer(optimizer_d)

            real_img_pred = self.discriminator(target_vid_original)
            predicted_img, tgt_rigid_pose, feature_volume = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)
            recon_img_pred = self.discriminator(predicted_img.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

            self.log("d_loss", d_loss, prog_bar=True)

        else:
            optimizer_g = self.optimizers()
            self.set_module_eval_train_state(True)
            self.toggle_optimizer(optimizer_g)
            # get reconstructed image
            predicted_img, tgt_rigid_pose, feature_volume = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)

            if self.l_w_face > 0 or self.l_w_face_l1 > 0:
                eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
                eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
                face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
                face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask, tgt_rigid_pose, feature_volume)

            else:
                loss_dict = self.compute_loss(target_vid_original, predicted_img, tgt_rigid_pose=tgt_rigid_pose, feature_volume=feature_volume)

            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

        for k, v in loss_dict.items():
            self.log(k, v, prog_bar=True)


        if False:
            checkpoint = torch.load(self.config.model.pretrained_ckpt)["state_dict"]
            (self.motion_encoder.convs[0][0].weight - checkpoint['motion_encoder.convs.0.0.weight']).sum()

            # check vgg16 weight
            from torchvision import models
            vgg_model = models.vgg19(pretrained=True).cuda()
            vgg_params = []
            for p in vgg_model.parameters():
                vgg_params.append(p)

            (self.criterion_vgg.vgg.slice1[0].weight - vgg_params[0]).mean()
            (self.criterion_vgg.vgg.slice2[0].weight - vgg_params[2]).mean()
            import pdb; pdb.set_trace()

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self._step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step > 5:
            # get source-target image pair
            ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)

            # get reconstructed image
            with torch.no_grad():
                predicted_img, tgt_rigid_pose, feature_volume = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)
            loss_dict = self.compute_loss(target_vid_original, predicted_img, tgt_rigid_pose=tgt_rigid_pose)

            self.log('val_recon_loss', loss_dict['l1_loss'], prog_bar=True)

            return loss_dict['l1_loss']

    def configure_optimizers(self):
        params_to_update = list(self.motion_encoder.parameters()) + list(self.flow_estimator.parameters()) + \
                           list(self.face_encoder.parameters()) + list(self.face_generator.parameters())
        params_to_update = [p for p in params_to_update if p.requires_grad]

        g_reg_ratio = self.config.optimizer.g_reg_every / (self.config.optimizer.g_reg_every + 1)
        d_reg_ratio = self.config.optimizer.d_reg_every / (self.config.optimizer.d_reg_every + 1)
        optimizer = torch.optim.Adam(
            params_to_update,
            lr=self.config.optimizer.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        if self.l_w_gan > 0:
            optimizer_dis = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.optimizer.lr * d_reg_ratio,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
            )
            # return [
            #     {"optimizer": optimizer},
            #     {"optimizer": optimizer_dis, "do_not_count_global_step": True},
            # ]
            return [optimizer, optimizer_dis], []
        else:
            # import pdb; pdb.set_trace()
            return [optimizer], []


if __name__ == "__main__":
    from model.head_animation.LIA.motion_encoder import MotionEncoder
    from model.head_animation.LIA.flow_estimator import FlowEstimator
    from model.head_animation.LIA.face_encoder import FaceEncoder
    from model.head_animation.LIA.face_generator import FaceGenerator
    from torchsummaryX import summary

    IMAGE_SIZE = 512
    latent_dim = 512

    encoder = MotionEncoder(latent_dim=latent_dim, size=IMAGE_SIZE)
    # summary(encoder, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    motion_space=20
    flow_estimator = FlowEstimator(latent_dim=latent_dim, motion_space=motion_space)
    # summary(flow_estimator, torch.zeros(1, latent_dim), torch.zeros(1, latent_dim))
    tgt_latent = flow_estimator(torch.zeros(1, latent_dim), torch.zeros(1, latent_dim))

    face_encoder = FaceEncoder(output_channels=latent_dim)
    # summary(face_encoder, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    feat = face_encoder(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    # for fea in feat: print(fea.shape)

    face_generator = FaceGenerator(IMAGE_SIZE, latent_dim, channel_multiplier=1)
    face_generator(tgt_latent, feat)
