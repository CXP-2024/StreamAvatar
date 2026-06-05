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
import lpips
import numpy as np
from skimage.metrics import structural_similarity
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR
from time import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.lightning.base_modules import BaseModule
from utils import instantiate


class HeadAnimatorModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.validation_step_outputs = []
        self.output_dir = config.model.get("output_dir", "outputs")

        self.config = config
        self.using_hybrid_mask = config.model.get("using_hybrid_mask", True)
        self.using_seg = config.model.get("using_seg", False)

        print(f'Using Hybird Mask: {self.using_hybrid_mask}')
        print(f'Using Segmentation: {self.using_seg}')
        print(f'Results will be saved to: {self.output_dir}')

        self.criterion_recon = nn.L1Loss()
        self.criterion_masked_face_l1 = nn.L1Loss(reduction='none')

        if self.config.model.get('sdm_loss', None) is not None and self.config.loss.get('l_w_sdm', 0) > 0:
            self.criterion_sdm = instantiate(self.config.model.sdm_loss)

        if self.config.model.get('vgg_loss', None) is not None:
            self.criterion_vgg = instantiate(self.config.model.vgg_loss)

        if self.config.get('loss', None) is not None:
            self.l_w_recon = self.config.loss.get("l_w_recon", 0)
            self.l_w_vgg = self.config.loss.get("l_w_vgg", 0)
            self.l_w_face = self.config.loss.get("l_w_face", 0)
            self.l_w_gan = self.config.loss.get("l_w_gan", 0)
            self.l_w_face_l1 = self.config.loss.get("l_w_face_l1", 0)
            self.l_w_gaze = self.config.loss.get("l_w_gaze", 0)
            self.l_w_foreground = self.config.loss.get("l_w_foreground", 0)
            self.l_w_local = self.config.loss.get("l_w_local", 0)
            self.l_w_sdm = self.config.loss.get("l_w_sdm", 0)
            self.l_w_ref_consistency = self.config.loss.get("l_w_ref_consistency", 0)
            self.add_gan_step = self.config.loss.get("add_gan_step", 0)
        else:
            self.l_w_recon = 1
            self.l_w_vgg = 0
            self.l_w_face = 0
            self.l_w_gan = 0
            self.l_w_face_l1 = 0
            self.l_w_gaze = 0
            self.l_w_foreground = 0
            self.l_w_local = 0
            self.l_w_sdm = 0
            self.l_w_ref_consistency = 0
            self.add_gan_step = 0
        self.step_cnt = 0

        self.face_parsing_en = self.l_w_foreground > 0 or self.l_w_local > 0

        # support GAN training & normal training
        self.automatic_optimization = False

        if 'VASA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'VASA'

        if 'LIA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'LIA'

        print(f'Using {self.model_name} for Head Animation')

    def configure_model(self):
        config = self.config
        self.motion_encoder = instantiate(config.model.motion_encoder)
        self.flow_estimator = instantiate(config.model.flow_estimator)
        self.face_generator = instantiate(config.model.face_generator)
        self.face_encoder = instantiate(config.model.face_encoder)

        if self.config.get('loss', None) is not None:
            if config.loss.l_w_gan > 0:
                self.discriminator = instantiate(config.model.discriminator)

            if self.config.loss.get('l_w_gaze', None) is not None and config.loss.l_w_gaze > 0:
                self.gaze_estimator = instantiate(config.model.gaze_estimator)

            if self.config.loss.get('l_w_foreground', None) is not None and config.loss.l_w_foreground > 0 or \
                self.config.loss.get('l_w_local', None) is not None and config.loss.l_w_local > 0 or \
                self.config.model.get('using_seg', None) is not None and config.model.using_seg:
                self.face_parser = instantiate(config.model.face_parser)

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

            ckpt = {}
            for k, v in checkpoint.items():
                if 'face_encoder' in k:
                    ckpt[k.replace('face_encoder.', '')] = v
            self.face_encoder.load_state_dict(ckpt, strict=True)

    def motion_encode(self, source_img):
        latent_code, pyramid_feat = self.motion_encoder(source_img)
        return latent_code, pyramid_feat

    def forward(self, source_img, target_img, masked_source_img, masked_target_img, batch_idx=None):

        face_feat = self.face_encoder(source_img) # get source appearance feature

        source_motion_img = masked_source_img if self.using_hybrid_mask else source_img
        tgt_motion_img = masked_target_img if self.using_hybrid_mask else target_img
        src_latent, _ = self.motion_encoder(source_motion_img) # project target image to reference latent space
        tgt_latent, _ = self.motion_encoder(tgt_motion_img) # project source image to reference latent space

        tgt_latent_from_src = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space
        recon_img = self.face_generator(tgt_latent_from_src, face_feat)


        out_dict = {}
        out_dict['recon_img'] = recon_img
        out_dict['tgt_latent'] = tgt_latent
        out_dict['src_latent'] = src_latent
        out_dict['face_mask'] = None

        if self.l_w_ref_consistency > 0:
            tgt_fea = self.face_encoder(target_img)
            out_dict['tgt_fea'] = tgt_fea.detach() # avoid to optimize face_encoder
            out_dict['src_fea'] = face_feat.detach() # avoid to optimize face_encoder

        return out_dict

    def compute_base_loss(self, img_target, img_target_recon, face_mask=None, tgt_parsing_map_dict=None):

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
            face_l1_loss = face_l1_loss.view(face_mask.size(0), -1).sum(-1) / face_mask.view(face_mask.size(0), -1).sum(-1)
            face_l1_loss = self.l_w_face_l1 * face_l1_loss.mean()
        else:
            face_l1_loss = torch.zeros(1).to(self.device)

        gaze_loss = torch.zeros(1).to(self.device)

        if self.face_parsing_en:
            assert tgt_parsing_map_dict is not None
            face_mask = tgt_parsing_map_dict['face_mask']
            face_body = tgt_parsing_map_dict['face_body']
            cloth_mask = tgt_parsing_map_dict['cloth_mask']
            mouth = tgt_parsing_map_dict['mouth']
            eye = tgt_parsing_map_dict['eye']
            ear = tgt_parsing_map_dict['ear']

            if self.l_w_foreground > 0:
                human_mask = face_body + cloth_mask
                img_target_human = img_target * human_mask
                foreground_loss, _ = self.criterion_vgg(img_target_recon, img_target_human)
                foreground_loss = self.l_w_foreground * foreground_loss.mean()
            else:
                foreground_loss = torch.zeros(1).to(self.device)

            if self.l_w_local > 0:
                eye_mouth_ear_mask = eye + mouth + ear
                img_target_local = img_target * eye_mouth_ear_mask
                img_target_recon_local = img_target_recon * eye_mouth_ear_mask

                local_loss, _  = self.criterion_vgg(img_target_recon_local, img_target_local)
                local_loss = self.l_w_local * local_loss.mean()
            else:
                local_loss = torch.zeros(1).to(self.device)
        else:
            foreground_loss = torch.zeros(1).to(self.device)
            local_loss = torch.zeros(1).to(self.device)

        return vgg_loss, l1_loss, face_loss, face_l1_loss, gaze_loss, foreground_loss, local_loss

    def compute_loss(self, img_target, out_dict, tgt_parsing_map_dict=None):
        vgg_loss, l1_loss, face_loss, face_l1_loss, gaze_loss, foreground_loss, local_loss = self.compute_base_loss(img_target, out_dict['recon_img'], out_dict['face_mask'], tgt_parsing_map_dict)

        loss = vgg_loss + l1_loss + face_loss + face_l1_loss + gaze_loss + foreground_loss + local_loss
        loss_dict = {'loss': loss, 'l1_loss': l1_loss, 'face_l1_loss': face_l1_loss, 'vgg_loss': vgg_loss,
                     'gaze_loss': gaze_loss, 'face_loss': face_loss, 'foreground_loss': foreground_loss, 'local_loss': local_loss,}

        if self.l_w_sdm > 0:
            sdm_loss = self.l_w_sdm * self.criterion_sdm(out_dict['src_latent'], out_dict['tgt_latent'])
            loss_dict['loss'] += sdm_loss
            loss_dict['sdm_loss'] = sdm_loss

        if self.l_w_ref_consistency > 0:
            ref_consistency_loss = self.l_w_ref_consistency * self.ref_consistency_loss(out_dict['src_latent'], out_dict['tgt_latent'], out_dict['src_fea'], out_dict['tgt_fea'])
            loss_dict['loss'] += ref_consistency_loss
            loss_dict['ref_consistency_loss'] = ref_consistency_loss

        return loss_dict

    def ref_consistency_loss(self, src_latent, tgt_latent, src_fea, tgt_fea):
        ref_img_from_src = self.face_generator(src_latent, src_fea)
        ref_img_from_tgt = self.face_generator(tgt_latent, tgt_fea)

        return F.l1_loss(ref_img_from_src, ref_img_from_tgt)

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
        masked_ref_img = batch['pixel_values_ref_img']

        # when not zero_to_one, all of bellow is [-1, 1]
        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([past_frames, target_vid_original], dim=1)

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

        return ref_img_original, target_vid_original, masked_ref_img, masked_target_vid

    def _step(self, batch, batch_idx):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)

        if self.using_seg or self.face_parsing_en:
            # get human parsing maps
            tgt_parsing_map_dict = self.face_parser.forward(target_vid_original)

            if self.using_seg:
                src_parsing_map_dict = self.face_parser.forward(ref_img_original)
                src_face_body = src_parsing_map_dict['face_body']
                src_cloth_mask = src_parsing_map_dict['cloth_mask']
                src_human_mask = src_face_body + src_cloth_mask
                ref_img_original = ref_img_original * src_human_mask

                tgt_face_body = tgt_parsing_map_dict['face_body']
                tgt_cloth_mask = tgt_parsing_map_dict['cloth_mask']
                tgt_human_mask = tgt_face_body + tgt_cloth_mask
                target_vid_original = target_vid_original * tgt_human_mask
        else:
            tgt_parsing_map_dict = None

        out_dict = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)
        loss_dict = self.compute_loss(target_vid_original, out_dict, tgt_parsing_map_dict=tgt_parsing_map_dict)

        if self.l_w_gan > 0:
            optimizer_g, optimizer_d = self.optimizers()

            self.discriminator.requires_grad_(False)
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            if self.step_cnt >= self.add_gan_step:
                loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss

            if is_grad_step:
                optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()

            ## train discriminator
            self.discriminator.requires_grad_(True)
            real_img_pred = self.discriminator(target_vid_original)
            predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)
            recon_img_pred = self.discriminator(predicted_img.detach())
            # import pdb; pdb.set_trace()

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

            real_probs = torch.sigmoid(real_img_pred)
            fake_probs = torch.sigmoid(recon_img_pred)
            correct_real = (real_probs >= 0.5).float()
            correct_fake = (fake_probs < 0.5).float()
            total_correct = correct_real.sum() + correct_fake.sum()
            total_samples = real_probs.numel() + fake_probs.numel()
            accuracy = total_correct / total_samples
            real_acc = correct_real.sum() / total_samples
            fake_acc = correct_fake.sum() / total_samples

            optimizer_d.zero_grad()
            self.manual_backward(d_loss)
            optimizer_d.step()

            loss_dict['d_loss'] = d_loss
            # loss_dict['d_acc'] = accuracy
            loss_dict['d_real_acc'] = real_acc
            loss_dict['d_fake_acc'] = fake_acc
            self.step_cnt += 1
        else:
            optimizer_g = self.optimizers()
            lr_scheduler = self.lr_schedulers()
            self.set_module_eval_train_state(True)
            self.toggle_optimizer(optimizer_g)
            # get reconstructed image
            predicted_img = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)

            if self.l_w_face > 0 or self.l_w_face_l1 > 0:
                eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
                eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
                face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
                face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

                loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask)

            else:
                loss_dict = self.compute_loss(target_vid_original, predicted_img)

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
            if v > 0:
                self.log(k, v, prog_bar=True)

        return loss_dict

    def training_step(self, batch, batch_idx):

        loss_dict = self._step(batch, batch_idx)

        # log current learning rate
        if self.l_w_gan > 0:
            optimizer_g, optimizer_d = self.optimizers()
        else:
            optimizer_g = self.optimizers()
        current_lr = optimizer_g.param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step > 1:
            # get source-target image pair
            ref_img_original, target_vid_original, masked_ref_img, masked_target_vid = self.prepare_datapair(batch)

            if self.using_seg or self.face_parsing_en:
                # get human parsing maps
                tgt_parsing_map_dict = self.face_parser.forward(target_vid_original)

                if self.using_seg:
                    src_parsing_map_dict = self.face_parser.forward(ref_img_original)
                    src_face_body = src_parsing_map_dict['face_body']
                    src_cloth_mask = src_parsing_map_dict['cloth_mask']
                    src_human_mask = src_face_body + src_cloth_mask
                    ref_img_original = ref_img_original * src_human_mask

                    tgt_face_body = tgt_parsing_map_dict['face_body']
                    tgt_cloth_mask = tgt_parsing_map_dict['cloth_mask']
                    tgt_human_mask = tgt_face_body + tgt_cloth_mask
                    target_vid_original = target_vid_original * tgt_human_mask
            else:
                tgt_parsing_map_dict = None

            # get reconstructed image
            with torch.no_grad():
                out_dict = self.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, batch_idx)
            loss_dict = self.compute_loss(target_vid_original, out_dict, tgt_parsing_map_dict=tgt_parsing_map_dict)

            if target_vid_original.min() < 0:
                predicted_img = (out_dict['recon_img'] + 1) / 2
                target_vid_original = (target_vid_original + 1) / 2

            loss_dict['l1_loss'] = F.l1_loss(predicted_img, target_vid_original).mean()
            predicted_img = (predicted_img * 255).permute(0, 2, 3, 1).cpu().numpy()
            target_vid_original = (target_vid_original * 255).permute(0, 2, 3, 1).cpu().numpy()

            psnr_list = []
            ssim_list = []
            for tmp_i in range(len(predicted_img)):
                psnr = lpips.psnr(predicted_img[tmp_i], target_vid_original[tmp_i], peak=255.)
                ssim = structural_similarity(predicted_img[tmp_i], target_vid_original[tmp_i], data_range=255, multichannel=True, channel_axis=2)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)

            loss_dict['val_psnr'] = avg_psnr
            loss_dict['val_ssim'] = avg_ssim

            self.validation_step_outputs.append(loss_dict)

            return loss_dict

    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or len(self.validation_step_outputs) == 0:
            return

        # get all metrics
        outputs = self.validation_step_outputs
        avg_recon_loss = torch.stack([x['l1_loss'] for x in outputs]).mean()
        avg_foreground_loss = torch.stack([x['foreground_loss'] for x in outputs]).mean()
        avg_local_loss = torch.stack([x['local_loss'] for x in outputs]).mean()

        avg_psnr = np.mean([x['val_psnr'] for x in outputs])
        avg_ssim = np.mean([x['val_ssim'] for x in outputs])

        # log metrics
        self.log('val_recon_loss', avg_recon_loss, prog_bar=True)
        self.log('val_psnr', avg_psnr, prog_bar=True)
        self.log('val_ssim', avg_ssim, prog_bar=True)

        if self.face_parsing_en:
            self.log('val_foreground_loss', avg_foreground_loss, prog_bar=True)
            self.log('val_local_loss', avg_local_loss, prog_bar=True)

        if self.global_rank == 0:
            log_file = f"{self.output_dir}/validation_metrics.txt"
            current_epoch = self.current_epoch
            global_step = self.global_step
            log_content = (
                f"Epoch: {current_epoch}, "
                f"Step: {global_step}, "
                f"Recon Loss: {avg_recon_loss.item():.4f}, "
                f"PSNR: {avg_psnr:.4f}, "
                f"SSIM: {avg_ssim:.4f}"
                )
            if self.face_parsing_en:
                log_content += (
                    f", Foreground Loss: {avg_foreground_loss.item():.4f}, "
                    f"Local Loss: {avg_local_loss.item():.4f}"
                )

            with open(log_file, "a") as f:
                f.write("*" * 50 + "\n")
                f.write(log_content + "\n")

        # clear cache for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        params_to_update = list(self.motion_encoder.parameters()) + list(self.flow_estimator.parameters()) + \
                           list(self.face_encoder.parameters()) + list(self.face_generator.parameters())
        params_to_update = [p for p in params_to_update if p.requires_grad]
        params_name_to_update = [name for name, p in self.named_parameters() if p.requires_grad]

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
            if self.model_name == 'LIA':
                d_reg_ratio = self.config.optimizer.d_reg_every / (self.config.optimizer.d_reg_every + 1)
                optimizer_dis = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.config.optimizer.discriminator_lr * d_reg_ratio,
                    weight_decay=self.config.optimizer.weight_decay,
                    betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
                    eps=self.config.optimizer.adam_epsilon,
                )
            else:
                optimizer_dis = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.config.optimizer.discriminator_lr,
                    weight_decay=self.config.optimizer.weight_decay,
                    betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                    eps=self.config.optimizer.adam_epsilon,
                )

            return [optimizer, optimizer_dis], []
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
