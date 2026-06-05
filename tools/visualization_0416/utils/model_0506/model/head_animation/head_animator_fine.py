import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
import sys
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
import math
import cv2

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.lightning.base_modules import BaseModule
from utils import instantiate
from losses.face_parsing_loss.face_parser import FaceParser

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
        self.l_w_gaze = config.loss.get("l_w_gaze", 0)

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

        if config.loss.get("l_mask_parse", False):
            self.face_parse_model = FaceParser()
            self.face_parse_model.eval()
            for name, param in self.face_parse_model.named_parameters():
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

        if self.l_w_gaze > 0:
            self.criterion_gaze = instantiate(config.model.gaze_loss)
            for name, param in self.criterion_gaze.named_parameters():
                param.requires_grad = False
            self.criterion_gaze.eval()

    def configure_model(self):
        pass

    def motion_encode(self, source_img):
        latent_code, pyramid_feat = self.motion_encoder(source_img)
        return latent_code, pyramid_feat

    def forward(self, source_img, target_img, masked_source_img, masked_target_img):
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
        out_dict = {}
        out_dict['recon_img'] = recon_img
        return out_dict

    @torch.no_grad()
    def get_face_parse(self, img):
        # import pdb; pdb.set_trace()
        parsing_map_dict = self.face_parse_model(img)
        return parsing_map_dict["mouth"] | parsing_map_dict["eye"]

    def compute_loss(self, img_target, img_target_recon, face_mask=None, face_keypoints=None):

        l1_loss = self.l_w_recon * self.criterion_recon(img_target_recon, img_target)

        # Perceptual Loss
        if self.l_w_vgg > 0:
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
            face_l1_loss = face_l1_loss.view(face_mask.size(0), -1).sum(-1) / (face_mask.view(face_mask.size(0), -1).sum(-1) + 1e-6)
            face_l1_loss = self.l_w_face_l1 * face_l1_loss.mean()
        else:
            face_l1_loss = torch.zeros(1).to(self.device)

        # Gaze Loss
        if self.l_w_gaze > 0:
            gaze_loss = self.criterion_gaze(img_target_recon, img_target, face_keypoints)
            gaze_loss = self.l_w_gaze * gaze_loss.to(self.device)
            print(f"[DEBUG] Gaze Loss Value: {gaze_loss.item()}")
            print(f"[DEBUG] Gaze Loss requires_grad: {gaze_loss.requires_grad}")
            if isinstance(self.criterion_gaze, torch.nn.Module):
                if hasattr(self.criterion_gaze, "_last_input_feature"):
                    grad_tensor = self.criterion_gaze._last_input_feature
                    if grad_tensor is not None:
                        print("[DEBUG] _last_input_feature.grad_fn:", grad_tensor.grad_fn)
                        print("[DEBUG] _last_input_feature.requires_grad:", grad_tensor.requires_grad)
                        grad_tensor.retain_grad()
        else:
            gaze_loss = torch.zeros(1).to(self.device)

        loss = vgg_loss + l1_loss + face_loss + face_l1_loss + gaze_loss
        loss_dict = {
            'loss': loss,
            'l1_loss': l1_loss,
            'face_l1_loss': face_l1_loss,
            'vgg_loss': vgg_loss,
            'face_loss': face_loss,
            'gaze_loss': gaze_loss
        }
        return loss_dict

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def prepare_datapair(self, batch):
        pixel_values_vid_key = "eyeball_pixel_values_vid" if self.config.data.get("eyeball_enable", False) else "pixel_values_vid"
        pixel_values_past_frames_key = "eyeball_past_frames" if self.config.data.get("eyeball_enable", False) else "pixel_values_past_frames"
        pixel_values_ref_img_key = "eyeball_ref_img" if self.config.data.get("eyeball_enable", False) else "pixel_values_ref_img"
        # when not zero_to_one, all of bellow is [-1, 1]
        masked_target_vid = batch[pixel_values_vid_key] # this is a video batch: [B, T, C, H, W]
        masked_past_frames = batch[pixel_values_past_frames_key]
        masked_target_vid = torch.cat([masked_past_frames, masked_target_vid], dim=1)
        masked_ref_img = batch[pixel_values_ref_img_key] # b c h w

        # when not zero_to_one, all of bellow is [-1, 1]
        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([past_frames, target_vid_original], dim=1)


        # get face keypoints
        face_keypoints = batch['keypoints']
        if face_keypoints is not None:
            face_keypoints = face_keypoints.to(self.device)
            face_keypoints = rearrange(face_keypoints, "b t v c -> (b t) v c")



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

        return ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, face_keypoints
    def set_module_eval_train_state(self, state_is_g=True):
        if self.l_w_gan > 0:
            self.discriminator.train()
        self.motion_encoder.train()
        self.flow_estimator.train()
        self.face_generator.train()
        self.face_encoder.train()

    def _step(self, batch, batch_idx):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, face_keypoints = self.prepare_datapair(batch)

        # 添加关键点数据的调试信息
        # if face_keypoints is not None:
        #     print(f"[DEBUG] Face Keypoints Shape: {face_keypoints.shape}")
        #     print(f"[DEBUG] Face Keypoints Range: min={face_keypoints.min().item():.2f}, max={face_keypoints.max().item():.2f}")
        #     print(f"[DEBUG]ref_img_original: {ref_img_original.shape}")
        #     # print(f"[DEBUG] Face Keypoints: {face_keypoints}")

        # normalize face keypoints
        if face_keypoints is not None:
            face_keypoints = face_keypoints / 512.0 * 2 - 1

        accumulate_grad_batches = self.config.data.get("accumulate_grad_batches", 1)
        is_grad_step = ((batch_idx + 1 )% accumulate_grad_batches == 0)
        if self.l_w_gan > 0: # using GAN training
            optimizer_g, optimizer_d = self.optimizers()
            lr_scheduler, dis_lr_scheduler = self.lr_schedulers()
            ## train generator
            # toggle is same to set grad is true
            self.set_module_eval_train_state(True)
            self.toggle_optimizer(optimizer_g)
            # get reconstructed image
            predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)['recon_img']

            if self.l_w_face > 0 or self.l_w_face_l1 > 0:
                eye_mouth_mask_vid_key = "eye_mouth_mask_vid" if self.config.data.get("l_mask_scale", True) else "eye_mouth_mask_no_scale_vid"
                eye_mouth_mask_past_frames_key = "eye_mouth_mask_past_frames" if self.config.data.get("l_mask_scale", True) else "eye_mouth_mask_no_scale_past_frames"
                eye_mouth_mask_vid = batch[eye_mouth_mask_vid_key]
                eye_mouth_mask_past_frames = batch[eye_mouth_mask_past_frames_key]
                face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
                face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")
                if self.config.loss.get("l_mask_parse", False):
                    face_mask = self.get_face_parse(target_vid_original)
                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask, face_keypoints)
            else:
                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_keypoints=face_keypoints)

            # adversarial loss
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss

            if is_grad_step:
                optimizer_g.zero_grad()
            print("[DEBUG] Before backward - Total Loss:", loss_dict['loss'].item())
            print("[DEBUG] Before backward - Gaze Loss:", loss_dict['gaze_loss'].item())
            self.manual_backward(loss_dict['loss'])

            if is_grad_step:
                optimizer_g.step()
                lr_scheduler.step()
            self.untoggle_optimizer(optimizer_g)

            ## train discriminator
            self.set_module_eval_train_state(False)
            self.toggle_optimizer(optimizer_d)

            real_img_pred = self.discriminator(target_vid_original)
            predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)['recon_img']
            recon_img_pred = self.discriminator(predicted_img.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

            if is_grad_step:
                optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            if is_grad_step:
                optimizer_d.step()
                dis_lr_scheduler.step()
            self.untoggle_optimizer(optimizer_d)

            self.log("d_loss", d_loss, prog_bar=True)

            lr_g = optimizer_g.param_groups[0]['lr']
            lr_d = optimizer_d.param_groups[0]['lr']
            self.log("learning_rate_g", lr_g)
            self.log("learning_rate_d", lr_d)

        else:
            optimizer_g = self.optimizers()
            lr_scheduler = self.lr_schedulers()
            self.set_module_eval_train_state(True)
            self.toggle_optimizer(optimizer_g)
            # get reconstructed image
            predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)['recon_img']

            if self.l_w_face > 0 or self.l_w_face_l1 > 0:
                eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
                eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
                face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
                face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask, face_keypoints=face_keypoints)

            else:
                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_keypoints=face_keypoints)

            if is_grad_step:
                optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            if is_grad_step:
                optimizer_g.step()
                lr_scheduler.step()
            self.untoggle_optimizer(optimizer_g)

            lr_g = optimizer_g.param_groups[0]['lr']
            self.log("learning_rate_g", lr_g)

        for k, v in loss_dict.items():
            if k in ['loss', 'l1_loss',  'gaze_loss', 'face_l1_loss', 'vgg_loss', 'g_loss']:
                self.log(k, v, prog_bar=True)
            else:
                self.log(k, v)


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
        return
        # if self.trainer.global_step > 5:
        #     # get source-target image pair
        #     ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)

        #     # get reconstructed image
        #     with torch.no_grad():
        #         predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)
        #     loss_dict = self.compute_loss(target_vid_original, predicted_img, tgt_rigid_pose=tgt_rigid_pose)

        #     self.log('val_recon_loss', loss_dict['l1_loss'], prog_bar=True)

        #     return loss_dict['l1_loss']

    def configure_optimizers(self):
        params_to_update = list(self.motion_encoder.parameters()) + list(self.flow_estimator.parameters()) + \
                           list(self.face_encoder.parameters()) + list(self.face_generator.parameters())
        params_to_update = [p for p in params_to_update if p.requires_grad]
        params_name_to_update = [name for name, p in self.named_parameters() if p.requires_grad]

        g_reg_every, d_reg_every = 4, 16
        g_reg_ratio = g_reg_every / (g_reg_every + 1)
        d_reg_ratio = d_reg_every / (d_reg_every + 1)
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )
        if (self.config.get("lr_scheduler", None) is not None) and (self.config.lr_scheduler.type == "cos_anneal"):
            lr_scheduler = CosineAnnealingLR(optimizer,
                                             T_max=self.config.lr_scheduler.T_max,
                                             eta_min=self.config.lr_scheduler.eta_min)
        else:
            lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        if self.l_w_gan > 0:
            optimizer_dis = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.optimizer.discriminator_lr * d_reg_ratio,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
                eps=self.config.optimizer.adam_epsilon,
            )
            if (self.config.get("dis_lr_scheduler", None) is not None) and (self.config.dis_lr_scheduler.type == "cos_anneal"):
                dis_lr_scheduler = CosineAnnealingLR(optimizer,
                                                T_max=self.config.dis_lr_scheduler.T_max,
                                                eta_min=self.config.dis_lr_scheduler.eta_min)
            else:
                dis_lr_scheduler = LambdaLR(optimizer_dis, lr_lambda=lambda step: 1.0)

            return [optimizer, optimizer_dis], [lr_scheduler, dis_lr_scheduler]
        else:
            # import pdb; pdb.set_trace()
            return [optimizer], [lr_scheduler]


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
