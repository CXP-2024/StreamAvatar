import torch
from torch import nn
import sys
from pathlib import Path
from einops import rearrange
import torch.nn.functional as F
import math
import lpips
import numpy as np
from skimage.metrics import structural_similarity
from time import time
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR, SequentialLR
from utils import instantiate
import torchvision
import matplotlib.pyplot as plt
from torchvision.ops import roi_align
import random

sys.path.append(str(Path(__file__).parent.parent.parent))
from model.lightning.base_modules import BaseModule
from model.head_animation.EMOP.head_pose_regressor import HeadPoseRegressor

class HeadAnimatorModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)

        self.validation_step_outputs = []
        self.output_dir = config.model.get("output_dir", "outputs")

        self.config = config
        self.using_hybrid_mask = config.model.get("using_hybrid_mask", True)
        self.using_seg = config.model.get("using_seg", False)
        self.face_part_start_iter = config.model.get("face_part_start_iter", 0)
        self.log_grad_freq = config.model.get("log_grad_freq", 1e8)
        self.max_grad_norm = config.model.get("max_grad_norm", 5)
        self.add_gan_step = config.model.get("add_gan_step", 0)

        print(f'Using Hybird Mask: {self.using_hybrid_mask}')
        print(f'Using Segmentation: {self.using_seg}')
        print(f'Add gan loss from {self.add_gan_step} steps!')
        print(f'Training log will be saved to: {self.output_dir}')

        self.criterion_recon = nn.L1Loss()
        self.criterion_masked_face_l1 = nn.L1Loss(reduction='none')

        if self.config.model.get('sdm_loss', None) is not None and self.config.loss.get('l_w_sdm', 0) > 0:
            self.criterion_sdm = instantiate(self.config.model.sdm_loss)

        if self.config.model.get('vgg_loss', None) is not None:
            self.criterion_vgg = instantiate(self.config.model.vgg_loss)

        if self.config.get('loss', None) is not None:
            self.l_w_recon = self.config.loss.get("l_w_recon", 0)
            self.l_w_vgg = self.config.loss.get("l_w_vgg", 0)
            self.l_w_face_vgg = self.config.loss.get("l_w_face_vgg", 0)
            self.l_w_gan = self.config.loss.get("l_w_gan", 0)
            self.l_w_face_l1 = self.config.loss.get("l_w_face_l1", 0)
            self.l_w_gaze = self.config.loss.get("l_w_gaze", 0)
            self.l_w_foreground = self.config.loss.get("l_w_foreground", 0)
            self.l_w_local = self.config.loss.get("l_w_local", 0)
            self.l_w_sdm = self.config.loss.get("l_w_sdm", 0)
            self.l_w_ref_consistency = self.config.loss.get("l_w_ref_consistency", 0)
            self.l_w_facial = self.config.loss.get("l_w_facial", 0)
            self.l_w_id = self.config.loss.get("l_w_id", 0)
            self.l_w_fea_match = self.config.loss.get("l_w_fea_match", 0)
        else:
            self.l_w_recon = 1
            self.l_w_vgg = 0
            self.l_w_face = 0
            self.l_w_gan = 0
            self.l_w_face_vgg = 0
            self.l_w_face_l1 = 0
            self.l_w_gaze = 0
            self.l_w_foreground = 0
            self.l_w_local = 0
            self.l_w_sdm = 0
            self.l_w_ref_consistency = 0
            self.l_w_facial = 0
            self.l_w_id = 0
            self.l_w_fea_match = 0

        self.step_cnt = 0
        self.time = 0
        self.init_time = None

        self.face_parsing_en = self.l_w_foreground > 0 or self.l_w_local > 0

        # support GAN training & normal training
        self.automatic_optimization = False

        if 'VASA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'VASA'

        if 'LIA' in self.config.model.motion_encoder.module_name:
            self.model_name = 'LIA'

        if 'EMOP' in self.config.model.motion_encoder.module_name:
            self.model_name = 'EMOPortrait'

        print(f'Using {self.model_name} for Head Animation')

    def configure_model(self):
        config = self.config
        self.motion_encoder = instantiate(config.model.motion_encoder)
        self.flow_estimator = instantiate(config.model.flow_estimator)
        self.face_generator = instantiate(config.model.face_generator)
        self.face_encoder = instantiate(config.model.face_encoder)

        if not self.motion_encoder.use_mask_image:
            self.motion_encoder.rigid_pose_encoder.eval() ## only freeze rigid_pose_encoder
            # for param in self.motion_encoder.rigid_pose_encoder.parameters():
            #     param.requires_grad = False
            print('Use original image to predict head pose')
        else:
            self.head_pose_regrssor = HeadPoseRegressor(config.model_config.head_pose_regressor_path)
            self.head_pose_regrssor.eval()
            for param in self.head_pose_regrssor.parameters():
                param.requires_grad = False

        if self.config.get('loss', None) is not None:
            if config.loss.l_w_gan > 0:
                self.discriminator = instantiate(config.model.discriminator)

            if self.config.loss.get('l_w_facial', None) is not None and config.loss.l_w_facial > 0:
                self.facial_discriminator = instantiate(config.model.facial_component_loss)

            if self.config.loss.get('l_w_id', None) is not None and config.loss.l_w_id > 0:
                self.criterion_vggface = instantiate(config.model.vggface)
                self.criterion_vggface.eval()
                for param in self.criterion_vggface.parameters():
                    param.requires_grad = False

            if self.config.loss.get('l_w_gaze', None) is not None and config.loss.l_w_gaze > 0:
                self.gaze_estimator = instantiate(config.model.gaze_estimator)
                self.gaze_estimator.eval()
                for param in self.gaze_estimator.parameters():
                    param.requires_grad = False

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

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('motion_encoder.rigid_pose_encoder.') and not k.startswith('head_pose_regrssor.') and not k.startswith('criterion_vggface.') and not k.startswith('face_parser.')
        }
        super().load_state_dict(filtered_state_dict, strict=False)

    ### overwrite state_dict function to reduce checkpoint size
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        keys_to_remove = [k for k in state_dict.keys() if k.startswith('head_pose_regrssor.')]
        keys_to_remove += [k for k in state_dict.keys() if k.startswith('criterion_vggface.')]
        keys_to_remove += [k for k in state_dict.keys() if k.startswith('face_parser.')]
        keys_to_remove += [k for k in state_dict.keys() if k.startswith('gaze_estimator.')]

        for k in keys_to_remove:
            del state_dict[k]

        return state_dict

    def log_gradient_stats(self, loss_dict=None):
        grad_stats = {}

        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[name] = {
                    "max": grad.abs().max().item(),
                    "mean": grad.abs().mean().item(),
                    "std": grad.std().item()
                }

        max_layer = max(grad_stats, key=lambda k: grad_stats[k]["max"])

        log_file = f"{self.output_dir}/grad_info.txt"
        current_epoch = self.current_epoch
        global_step = self.global_step

        if loss_dict is not None: l1_loss = f"L1 Loss: {loss_dict['l1']:.6f}"
        else: l1_loss = ""

        log_content = (
            f"Epoch: {current_epoch}, "
            f"Step: {global_step}, "
            f"{l1_loss}, "
            f"Max Gradient Layer: {max_layer}, Value: {grad_stats[max_layer]['max']:.6f}"
            )

        if self.global_rank == 0:
            with open(log_file, "a") as f:
                f.write("*" * 50 + "\n")
                f.write(log_content + "\n")

        return grad_stats


    def motion_encode(self, source_img):
        latent_code, pyramid_feat = self.motion_encoder(source_img)
        return latent_code, pyramid_feat

    def forward(self, source_img, target_img, masked_source_img, masked_target_img, batch_idx=None):
        data_dict = {'source_img': source_img, 'target_img': target_img, 'masked_source_img': masked_source_img, 'masked_target_img': masked_target_img}

        data_dict['latent_volume'],  data_dict['idt_embed'] = self.face_encoder(data_dict['source_img'])

        if self.l_w_ref_consistency > 0:
            data_dict['ref_matching'] = self.l_w_ref_consistency
            data_dict['latent_volume_target'],  _ = self.face_encoder(data_dict['target_img'])

        data_dict = self.motion_encoder(data_dict)
        data_dict = self.flow_estimator(data_dict)

        aligned_target_volume = data_dict['target_volume']
        target_warp_embed_dict = data_dict['target_warp_embed_dict']
        data_dict = self.face_generator(data_dict, target_warp_embed_dict, aligned_target_volume)

        recon_img = data_dict['pred_target_img']

        out_dict = {}
        out_dict['recon_img'] = recon_img
        out_dict['target_img'] = target_img
        out_dict['tgt_latent'] = data_dict['target_exp_embed']
        out_dict['src_latent'] = data_dict['source_exp_embed']
        out_dict['face_mask'] = None
        out_dict['source_theta'] = data_dict['source_theta']
        out_dict['target_theta'] = data_dict['target_theta']
        out_dict['target_scale'] = data_dict['target_scale']
        out_dict['target_rotation'] = data_dict['target_rotation']
        out_dict['target_translation'] = data_dict['target_translation']
        out_dict['align_warp'] = data_dict['align_warp']

        if self.l_w_ref_consistency > 0:
            out_dict['canonical_volume_from_tgt'] = data_dict['canonical_volume_from_tgt']
            out_dict['canonical_volume_from_src'] = data_dict['canonical_volume_from_src']

        return out_dict

    def compute_base_loss(self, out_dict, tgt_parsing_map_dict=None):

        img_target, img_target_recon, face_mask = out_dict['target_img'], out_dict['recon_img'], out_dict['face_mask']

        l1_loss = self.l_w_recon * self.criterion_recon(img_target_recon, img_target)

        # Full-image Perceptual Loss
        if self.l_w_vgg > 0:
            # vgg_loss, vgg_loss_dict = self.criterion_vgg(img_target_recon, img_target)
            vgg_loss = self.criterion_vgg(img_target_recon, img_target)
            vgg_loss = self.l_w_vgg * vgg_loss.mean()
        else:
            vgg_loss = torch.zeros(1).to(self.device)

        # Facial Experssion Perceptual Loss
        if self.l_w_face_vgg > 0:
            face_vgg_loss, face_vgg_loss_dict = self.criterion_vgg(out_dict['pred_target_img_face_align'], out_dict['target_img_align_orig'])
            face_vgg_loss = self.l_w_face_vgg * face_vgg_loss.mean()
        else:
            face_vgg_loss = torch.zeros(1).to(self.device)

        if face_mask is not None and self.l_w_face_l1 > 0:
            face_l1_loss = self.criterion_masked_face_l1(img_target_recon*face_mask, img_target*face_mask)
            face_l1_loss = face_l1_loss.view(face_mask.size(0), -1).sum(-1) / (face_mask.view(face_mask.size(0), -1).sum(-1) + 1e-6)
            face_l1_loss = self.l_w_face_l1 * face_l1_loss.mean()
        else:
            face_l1_loss = torch.zeros(1).to(self.device)

        if self.l_w_gaze > 0:
            gaze_loss = self.l_w_gaze * self.gaze_estimator(out_dict['recon_img'], out_dict['target_img'])
        else:
            gaze_loss = torch.zeros(1).to(self.device)

        if self.face_parsing_en and self.step_cnt >= self.face_part_start_iter:
            assert tgt_parsing_map_dict is not None
            face_mask = tgt_parsing_map_dict['face_mask']
            face_body = tgt_parsing_map_dict['face_body']
            cloth_mask = tgt_parsing_map_dict['cloth_mask']
            mouth = tgt_parsing_map_dict['mouth']
            eye = tgt_parsing_map_dict['eye']
            ear = tgt_parsing_map_dict['ear']
            human_mask = (face_body + cloth_mask).float()
            eye_mouth_ear_mask = (eye + mouth + ear).float()

            pred_tgt_parsing_map_dict = self.face_parser.forward(img_target_recon)
            pred_mouth = pred_tgt_parsing_map_dict['mouth']
            pred_eye = pred_tgt_parsing_map_dict['eye']
            pred_ear = pred_tgt_parsing_map_dict['ear']
            pred_face_body = tgt_parsing_map_dict['face_body']
            pred_cloth_mask = tgt_parsing_map_dict['cloth_mask']
            pred_human_mask = (pred_face_body + pred_cloth_mask).float()
            pred_eye_mouth_ear_mask = (pred_eye + pred_mouth + pred_ear).float().detach()

            pred_eye_mouth_ear_mask = eye_mouth_ear_mask

            gt_mask = eye_mouth_ear_mask.reshape(eye_mouth_ear_mask.size(0), -1).sum(-1)
            pred_mask = pred_eye_mouth_ear_mask.reshape(eye_mouth_ear_mask.size(0), -1).sum(-1)
            if (gt_mask == 0).any() or (pred_mask == 0).any():
                mask_flag = ((gt_mask * pred_mask) > 0).float()
                mask_flag = mask_flag.detach()
            else:
                mask_flag = torch.ones(eye_mouth_ear_mask.size(0)).float().to(self.device)

            if self.l_w_foreground > 0:
                img_target_human = img_target * human_mask
                img_target_recon_human = img_target_recon * pred_human_mask

                foreground_loss, _ = self.criterion_vgg(img_target_recon_human, img_target_human, human_mask)
                foreground_loss = self.l_w_foreground * foreground_loss.mean()
            else:
                foreground_loss = torch.zeros(1).to(self.device)

            if self.l_w_local > 0:
                use_mask = self.config.model.get('use_mask', False)
                # if use_mask: import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                img_target_local = img_target * eye_mouth_ear_mask
                img_target_recon_local = img_target_recon * pred_eye_mouth_ear_mask

                if 'rois_face' in out_dict and True:
                    img_target_local = roi_align(img_target_local, boxes=out_dict['rois_face'], output_size=img_target_local.size(-1))
                    img_target_recon_local = roi_align(img_target_recon_local, boxes=out_dict['rois_face'], output_size=img_target_local.size(-1))

                    if self.trainer.global_step > 1 and 0:
                        import cv2
                        pred_img_align = (img_target_local[-8].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
                        gt_img_align = (img_target_recon_local[-8].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
                        align_img = np.concatenate([gt_img_align, pred_img_align], axis=1)

                        # align_img = ( data_dict['target_img'][-4].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
                        cv2.imwrite(f'{self.output_dir}/emop_src_img_align_training_{self.step_cnt}.png', align_img[:,:,::-1])
                        import pdb; pdb.set_trace()

                # local_loss, _  = self.criterion_vgg(img_target_recon_local, img_target_local, eye_mouth_ear_mask, use_mask=use_mask)
                # local_loss = self.l_w_local * (local_loss * mask_flag).mean()

                local_loss = self.l_w_local * self.criterion_vgg(img_target_recon_local, img_target_local)

                # local_loss = self.l_w_local * self.criterion_recon(img_target_recon_local, img_target_local)
            else:
                local_loss = torch.zeros(1).to(self.device)
        else:
            foreground_loss = torch.zeros(1).to(self.device)
            local_loss = torch.zeros(1).to(self.device)

        return vgg_loss, l1_loss, face_vgg_loss, face_l1_loss, gaze_loss, foreground_loss, local_loss

    def compute_head_pose(self, out_dict):
        target_img = out_dict['target_img']
        target_theta, target_scale, target_rotation, target_translation = out_dict['target_theta'], out_dict['target_scale'], out_dict['target_rotation'], out_dict['target_translation']

        with torch.no_grad():
            if not self.motion_encoder.zero_to_one:
                img_gt = (target_img + 1) / 2
            else:
                img_gt = target_img
            gt_theta, gt_scale, gt_rotation, gt_translation = self.head_pose_regrssor.forward(img_gt, return_srt=True)
            gt_theta = gt_theta.detach()

        pred_pose = torch.cat([target_scale, target_rotation, target_translation], dim=-1)
        gt_pose = torch.cat([gt_scale, gt_rotation, gt_translation], dim=-1)
        headpose_loss = self.criterion_recon(pred_pose, gt_pose)
        return headpose_loss

    def compute_loss(self, out_dict, tgt_parsing_map_dict=None):

        if self.l_w_id > 0 or self.l_w_face_vgg > 0:
            if True:
                t = out_dict['target_img'].shape[0]
                pred_align_warp = out_dict['align_warp'].float() # B x 128 x 128 x 2
                inputs_orig_face_aligned = F.grid_sample(torch.cat([out_dict['recon_img'], out_dict['target_img']]).float(), pred_align_warp)
                out_dict['pred_target_img_face_align'], out_dict['target_img_align_orig'] = inputs_orig_face_aligned.split([t, t], dim=0)

                face_size = 512
                face_bbox = out_dict['facial_info']['face_bbox']
                rois_face = []
                for b in range(face_bbox.size(0)):  # loop for batch size
                    img_inds = face_bbox.new_full((1, 1), b).to(face_bbox.device)
                    rois = torch.cat([img_inds, face_bbox[b:b + 1, :]], dim=-1)  # shape: (1, 5)
                    rois_face.append(rois)
                rois_face = torch.cat(rois_face, 0).float()
                out_dict['rois_face'] = rois_face
            else:
                face_size = 512
                face_bbox = out_dict['facial_info']['face_bbox']
                rois_face = []
                for b in range(face_bbox.size(0)):  # loop for batch size
                    img_inds = face_bbox.new_full((1, 1), b).to(face_bbox.device)
                    rois = torch.cat([img_inds, face_bbox[b:b + 1, :]], dim=-1)  # shape: (1, 5)
                    rois_face.append(rois)
                rois_face = torch.cat(rois_face, 0).float()
                out_dict['rois_face'] = rois_face

                out_dict['target_img_align_orig'] = roi_align(out_dict['target_img'], boxes=rois_face, output_size=face_size)
                out_dict['pred_target_img_face_align'] = roi_align(out_dict['recon_img'], boxes=rois_face, output_size=face_size)


        vgg_loss, l1_loss, face_vgg_loss, face_l1_loss, gaze_loss, foreground_loss, local_loss = self.compute_base_loss(out_dict, tgt_parsing_map_dict)

        loss = vgg_loss + l1_loss + face_vgg_loss + face_l1_loss + gaze_loss + foreground_loss + local_loss
        loss_dict = {'loss': loss, 'l1': l1_loss, 'face_l1': face_l1_loss, 'vgg': vgg_loss,
                     'gaze': gaze_loss, 'face_vgg': face_vgg_loss, 'foreground': foreground_loss, 'local': local_loss,}

        if self.l_w_sdm > 0:
            sdm_loss = self.l_w_sdm * self.criterion_sdm(out_dict['src_latent'], out_dict['tgt_latent'])
            loss_dict['loss'] += sdm_loss
            loss_dict['sdm'] = sdm_loss

        if self.l_w_id > 0:
            if not self.motion_encoder.zero_to_one:
                gt_img = (out_dict['target_img_align_orig'] + 1) / 2
                recon_img = (out_dict['pred_target_img_face_align'] + 1) / 2
            else:
                gt_img = out_dict['target_img_align_orig']
                recon_img = out_dict['pred_target_img_face_align']

            id_loss = self.l_w_id * self.criterion_vggface(recon_img, gt_img.detach())
            loss_dict['loss'] += id_loss
            loss_dict['id'] = id_loss

        if self.l_w_ref_consistency > 0:
            canonical_volume_from_tgt = out_dict['canonical_volume_from_tgt']
            canonical_volume_from_src = out_dict['canonical_volume_from_src'].detach()

            ref_match_loss = self.l_w_ref_consistency * self.criterion_recon(canonical_volume_from_tgt, canonical_volume_from_src)
            loss_dict['loss'] += ref_match_loss
            loss_dict['ref_match'] = ref_match_loss

        # compute head pose loss
        if self.motion_encoder.use_mask_image:
            assert self.config.loss.get("l_w_headpose", 0) > 0
            headposs_loss = self.config.loss.l_w_headpose * self.compute_head_pose(out_dict)
            loss_dict['loss'] += headposs_loss
            loss_dict['headpose'] = headposs_loss

        return loss_dict

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return real_loss.mean() + fake_loss.mean()

    def prepare_datapair(self, batch):
        masked_target_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        masked_past_frames = batch['pixel_values_past_frames']
        masked_target_vid = torch.cat([masked_past_frames, masked_target_vid], dim=1)
        masked_ref_img = batch['pixel_values_ref_img']

        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([past_frames, target_vid_original], dim=1)

        ref_img_original = ref_img_original.to(self.device)
        target_vid_original = target_vid_original.to(self.device)
        masked_ref_img = masked_ref_img.to(self.device)
        masked_target_vid = masked_target_vid.to(self.device)

        # construct ref-tgt pairs
        masked_ref_img = masked_ref_img[:,None].repeat(1, masked_target_vid.size(1), 1, 1, 1)
        masked_ref_img = rearrange(masked_ref_img, "b t c h w -> (b t) c h w")
        masked_target_vid = rearrange(masked_target_vid, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        l_eye_bbox = torch.cat([batch['past_l_eye_bbox'], batch['vid_l_eye_bbox']], dim=1)
        r_eye_bbox = torch.cat([batch['past_r_eye_bbox'], batch['vid_r_eye_bbox']], dim=1)
        mouth_bbox = torch.cat([batch['past_mouth_bbox'], batch['vid_mouth_bbox']], dim=1)
        face_bbox = torch.cat([batch['past_face_bbox'], batch['vid_face_bbox']], dim=1)

        l_eye_bbox = rearrange(l_eye_bbox, "b t c -> (b t) c")
        r_eye_bbox = rearrange(r_eye_bbox, "b t c -> (b t) c")
        mouth_bbox = rearrange(mouth_bbox, "b t c -> (b t) c")
        face_bbox = rearrange(face_bbox, "b t c -> (b t) c")
        facial_info = {'l_eye_bbox': l_eye_bbox, 'r_eye_bbox': r_eye_bbox, 'mouth_bbox': mouth_bbox, 'face_bbox': face_bbox}

        return ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, facial_info

    def _step(self, batch, batch_idx):
        # get source-target image pair
        ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, facial_info = self.prepare_datapair(batch)

        if (self.using_seg or self.face_parsing_en) and self.step_cnt >= self.face_part_start_iter:
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
        out_dict['facial_info'] = facial_info
        loss_dict = self.compute_loss(out_dict, tgt_parsing_map_dict=tgt_parsing_map_dict)

        optimizers = self.optimizers()

        if isinstance(optimizers, (list, tuple)):
            optimizer_g, optimizer_d = optimizers[0], optimizers[1]

            ################# train generator #################
            self.toggle_optimizer(optimizer_g)

            GAN_STYLE = False
            ## step:1 compute global gan loss
            self.discriminator.eval()
            # print('1:', self.discriminator.training)
            self.discriminator.requires_grad_(False)
            # for p in self.discriminator.parameters():
            #     p.requires_grad = False

            if GAN_STYLE:
                pred_label = self.discriminator(out_dict['recon_img']).reshape(-1)
                g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)
                global_loss_dict = self.discriminator.gan_forward(out_dict['recon_img'])
                g_loss = global_loss_dict['global_gan_loss']
                feature_matching_loss = global_loss_dict['style_loss']
            else:
                # Without grad as it is ground truth
                with torch.no_grad():
                    _, real_feats_gen = self.discriminator(out_dict['target_img'])
                # With grad as it is predict
                fake_score_gen, fake_feats_gen = self.discriminator(out_dict['recon_img'])
                g_loss = self.l_w_gan * self.discriminator.compute_loss(fake_score_gen, mode='gen')
                feature_matching_loss = self.l_w_fea_match * self.discriminator.feature_matching_loss(real_feats_gen, fake_feats_gen)

            # if self.step_cnt >= self.add_gan_step:
            #     loss_dict['loss'] += g_loss
            #     loss_dict['loss'] += feature_matching_loss
            # else:
            #     loss_dict['loss'] += g_loss * self.step_cnt / self.add_gan_step
            #     loss_dict['loss'] += feature_matching_loss * self.step_cnt / self.add_gan_step

            loss_dict['gan'] = g_loss
            loss_dict['feat_m'] = feature_matching_loss

            ## step:2 compute facial gan loss
            if self.l_w_facial > 0:
                self.facial_discriminator.eval()
                self.facial_discriminator.requires_grad_(False)
                facial_dict = self.facial_discriminator.get_facial_component(out_dict['recon_img'], target_vid_original.detach(), facial_info)
                facial_loss_dict = self.facial_discriminator.gan_forward(facial_dict)
                facial_loss = facial_loss_dict['facial_gan_loss'] + facial_loss_dict['style_loss']

                # if self.step_cnt >= self.add_gan_step:
                #     loss_dict['loss'] += facial_loss
                # else:
                #     loss_dict['loss'] += facial_loss * self.step_cnt / self.add_gan_step
                # loss_dict['facial_gan'] = facial_loss

                loss_dict['loss'] += facial_loss

                loss_dict['face_gan'] = facial_loss_dict['facial_gan_loss']
                loss_dict['face_sty'] = facial_loss_dict['style_loss']

            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])

            # torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            optimizer_g.step()
            if self.step_cnt % self.log_grad_freq == 0:
                self.log_gradient_stats(loss_dict)

            self.untoggle_optimizer(optimizer_g)

            GAN_DISABLE = True
            if not GAN_DISABLE:
                ################# train global discriminator #################
                self.discriminator.train()
                # print('2:', self.discriminator.training)
                self.discriminator.requires_grad_(True)
                self.toggle_optimizer(optimizer_d)

            if GAN_STYLE:
                dis_loss_dict = self.discriminator.dis_forward(target_vid_original, out_dict['recon_img'].detach())
                d_loss = dis_loss_dict['d_loss']
            else:
                real_score_dis, _ = self.discriminator(out_dict['target_img'])
                fake_score_dis, _ = self.discriminator(out_dict['recon_img'].detach())
                real_score, fake_score = real_score_dis[0][0], fake_score_dis[0][0]

                d_loss = self.discriminator.compute_loss(fake_scores=fake_score, real_scores=real_score, mode='dis')
            loss_dict['d_loss'] = d_loss

            if True:
                # import pdb; pdb.set_trace()
                real_probs = torch.sigmoid(real_score)
                fake_probs = torch.sigmoid(fake_score)
                correct_real = (real_probs >= 0.5).float()
                correct_fake = (fake_probs < 0.5).float()
                total_correct = correct_real.sum() + correct_fake.sum()
                total_samples = real_probs.numel() + fake_probs.numel()
                accuracy = total_correct / total_samples
                real_acc = correct_real.sum() / real_probs.numel()
                fake_acc = correct_fake.sum() / fake_probs.numel()
                # loss_dict['d_acc'] = accuracy
                loss_dict['real_acc'] = real_acc
                loss_dict['fake_acc'] = fake_acc


            if not GAN_DISABLE:
                optimizer_d.zero_grad()
                self.manual_backward(d_loss)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            ################# train facial discriminator #################
            if self.l_w_facial > 0:
                optimizer_d_facial = optimizers[2]

                if not GAN_DISABLE:
                    self.facial_discriminator.train()
                    self.facial_discriminator.requires_grad_(True)
                    self.toggle_optimizer(optimizer_d_facial)

                facial_dis_loss_dict = self.facial_discriminator.dis_forward(facial_dict)
                loss_dict['facial_d_loss'] = facial_dis_loss_dict['d_loss']

                if not GAN_DISABLE:
                    optimizer_d_facial.zero_grad()
                    self.manual_backward(facial_dis_loss_dict['d_loss'])
                    optimizer_d_facial.step()
                    self.untoggle_optimizer(optimizer_d_facial)

                if True:
                    real_probs = torch.sigmoid(facial_dis_loss_dict['real_d_pred_mouth'])
                    fake_probs = torch.sigmoid(facial_dis_loss_dict['fake_d_pred_mouth'])
                    correct_real = (real_probs >= 0.5).float()
                    correct_fake = (fake_probs < 0.5).float()
                    total_correct = correct_real.sum() + correct_fake.sum()
                    total_samples = real_probs.numel() + fake_probs.numel()
                    accuracy = total_correct / total_samples
                    real_acc = correct_real.sum() / real_probs.numel()
                    fake_acc = correct_fake.sum() / fake_probs.numel()
                    # loss_dict['d_acc'] = accuracy
                    loss_dict['mouth_real_acc'] = real_acc
                    loss_dict['mouth_fake_acc'] = fake_acc

        else:
            optimizer_g = optimizers
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

        for k, v in loss_dict.items():
            if v > 0:
                self.log(k, v, prog_bar=True)

        if self.global_rank == 0 and self.global_step % 100 == 0:
            log_file = f"{self.output_dir}/train_log.txt"
            current_epoch = self.current_epoch
            global_step = self.global_step
            log_content =  f"Epoch: {current_epoch}, Step: {global_step}, "
            for k, v in loss_dict.items():
                log_content += f"{k}: {v.item():.4f}, "

            with open(log_file, "a") as f:
                f.write("*" * 50 + "\n")
                f.write(log_content + "\n")

        return loss_dict



    def training_step(self, batch, batch_idx):
        if self.init_time is None:
            self.init_time = time()

        loss_dict = self._step(batch, batch_idx)

        # log current learning rate
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            optimizer_g = optimizers[0]
        else:
            optimizer_g = optimizers
        current_lr = optimizer_g.param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True)

        # Step the scheduler after every training step
        if self.config.get('scheduler', None) is not None:
            if current_lr > self.config.scheduler.min_lr:
                if self.l_w_gan > 0:
                    schedulers_list = self.lr_schedulers()
                    scheduler_g, scheduler_d = schedulers_list[0], schedulers_list[1]
                    scheduler_g.step()
                    scheduler_d.step()

                    if self.l_w_facial > 0:
                        scheduler_facial = schedulers_list[2]
                        scheduler_facial.step()

                        # print(optimizers[1].param_groups[0]['lr'], optimizers[2].param_groups[0]['lr'])
                        # import pdb; pdb.set_trace()
                else:
                    scheduler = self.lr_schedulers()
                    scheduler.step()


        self.step_cnt += 1
        self.time = time() - self.init_time
        self.log('avg_time', self.time / self.step_cnt, on_step=True, on_epoch=False, prog_bar=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step > 1:
            # get source-target image pair
            ref_img_original, target_vid_original, masked_ref_img, masked_target_vid, facial_info = self.prepare_datapair(batch)

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
                out_dict['facial_info'] = facial_info
                loss_dict = self.compute_loss(out_dict, tgt_parsing_map_dict=tgt_parsing_map_dict)

                if target_vid_original.min() < 0:
                    predicted_img = (out_dict['recon_img'] + 1) / 2
                    target_vid_original = (target_vid_original + 1) / 2
                else:
                    predicted_img = out_dict['recon_img']

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
        avg_recon_loss = torch.stack([x['l1'] for x in outputs]).mean()
        avg_foreground_loss = torch.stack([x['foreground'] for x in outputs]).mean()
        avg_local_loss = torch.stack([x['local'] for x in outputs]).mean()

        avg_psnr = np.mean([x['val_psnr'] for x in outputs])
        avg_ssim = np.mean([x['val_ssim'] for x in outputs])

        # log metrics
        self.log('val_recon', avg_recon_loss, prog_bar=True)
        self.log('val_psnr', avg_psnr, prog_bar=True)
        self.log('val_ssim', avg_ssim, prog_bar=True)

        if self.face_parsing_en:
            if self.l_w_foreground > 0:
                avg_foreground_loss /= self.l_w_foreground
                self.log('val_foreground_loss', avg_foreground_loss, prog_bar=True)

            if self.l_w_local > 0:
                avg_local_loss /= self.l_w_local
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
        optimizer_list = [optimizer]

        # Scheduler with warm-up and cosine annealing
        if self.config.get('scheduler', None) is not None:
            total_steps = self.config.scheduler.total_steps

            # Cosine scheduler
            cos_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.config.scheduler.min_lr)
            scheduler = {
                'scheduler': cos_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
            scheduler_cfg = [scheduler]
        else:
            scheduler_cfg = []


        if self.l_w_gan > 0:
            optimizer_dis = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.optimizer.discriminator_lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                eps=self.config.optimizer.adam_epsilon,
            )

            # Scheduler with warm-up and cosine annealing
            if self.config.get('scheduler', None) is not None:
                # Cosine scheduler
                dis_cos_scheduler = CosineAnnealingLR(optimizer_dis, T_max=total_steps, eta_min=self.config.scheduler.min_lr)
                dis_scheduler = {
                    'scheduler': dis_cos_scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }

                optimizer_list += [optimizer_dis]
                scheduler_cfg += [dis_scheduler]

            if self.l_w_facial > 0:
                optimizer_facial = torch.optim.AdamW(
                    self.facial_discriminator.parameters(),
                    lr=self.config.optimizer.discriminator_lr,
                    weight_decay=self.config.optimizer.weight_decay,
                    betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
                    eps=self.config.optimizer.adam_epsilon,
                )

                # Scheduler with warm-up and cosine annealing
                if self.config.get('scheduler', None) is not None:
                    # Cosine scheduler
                    facial_cos_scheduler = CosineAnnealingLR(optimizer_facial, T_max=total_steps, eta_min=self.config.scheduler.min_lr)
                    facial_scheduler = {
                        'scheduler': facial_cos_scheduler,
                        'interval': 'step',
                        'frequency': 1,
                    }

                    optimizer_list += [optimizer_facial]
                    scheduler_cfg += [facial_scheduler]

        return optimizer_list, scheduler_cfg
