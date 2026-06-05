import torch
from torch import nn
from diffusers import DDIMScheduler
from omegaconf import OmegaConf

from model.head_animation.head_animator import HeadAnimatorModule
from utils import instantiate
import time

class AudioHeadAnimatorModule(HeadAnimatorModule):
    def __init__(self, config):
        super().__init__(config)
        self._get_scheduler()

    def configure_model(self):
        super().configure_model()
        # self.motion_generator = instantiate(self.config.model.motion_generator, True)
        # if self.config.model.motion_gen_ckpt is not None:
        #     checkpoint = torch.load(self.config.model.motion_gen_ckpt)
        #     # print(checkpoint.keys())
        #     self.motion_generator.load_state_dict(checkpoint, strict=False)
        # self.motion_generator.to(dtype=eval(self.config.model.dtype))

    def _get_scheduler(self):
        pass
        # self.motion_gen_cfg = OmegaConf.load(self.config.model.motion_generator.class_cfg)
        # ddim_sched_kwargs = OmegaConf.to_container(self.motion_gen_cfg.noise_scheduler_kwargs)
        # if self.motion_gen_cfg.enable_zero_snr:
        #     ddim_sched_kwargs.update(
        #         rescale_betas_zero_snr=True,
        #         timestep_spacing="trailing",
        #         # prediction_type="v_prediction",
        #     )
        # self.train_noise_scheduler = DDIMScheduler(
        #     **ddim_sched_kwargs
        # )

    def forward(self, source_img, masked_source_img, masked_past_frames, audio_self, audio_other, video_length, style_ref_frames=None):
        timings = {}
        device = source_img.device
        if self.using_hybrid_mask:
            t_start = time.time()
            style_ref_frames = style_ref_frames.squeeze(0)
            # print(f"masked_source_img shape: {masked_source_img.shape}, style_ref_frames shape: {style_ref_frames.shape}")
            src_latent, _ = self.motion_encoder(masked_source_img[0:1])
            src_latent = src_latent.repeat(video_length, 1)
            ref_latent, _ = self.motion_encoder(style_ref_frames)
            # print(f"src_latent shape: {src_latent.shape}, ref_latent shape: {ref_latent.shape}")
            style_motion = ref_latent.unsqueeze(0)
            timings['motion_encoder'] = time.time() - t_start
            print(f"Motion encoder time: {timings['motion_encoder']:.4f}s")

            t_start = time.time()
            # tgt_motion_latent = self.motion_generate(masked_past_frames, audio_self, audio_other, video_length) # project target image to reference latent space
            speaker_id = torch.zeros(1,1).to(device).long()
            audio_self = audio_self.unsqueeze(0)
            motion_latent_in = src_latent.unsqueeze(0)
            # print(f"motion_latent_in shape: {motion_latent_in.shape}, audio_self shape: {audio_self.shape}, speaker_id shape: {speaker_id.shape}")
            tgt_motion_latent = self.motion_generator.inference(audio_self, speaker_id, masked_motion=motion_latent_in,
                                                    noise_scheduler=self.train_noise_scheduler, style_motion=style_motion)
            # print(f"tgt_motion_latent shape: {tgt_motion_latent.shape}", src_latent.shape)
            timings['motion_generate'] = time.time() - t_start
            print(f"Motion generate time: {timings['motion_generate']:.4f}s")
            tgt_motion_latent = tgt_motion_latent.view(-1, tgt_motion_latent.size(-1))



            t_start = time.time()
            tgt_latent = self.flow_estimator(src_latent, tgt_motion_latent) # navigate source to target in reference latent space
            timings['flow_estimator'] = time.time() - t_start
            print(f"Flow estimator time: {timings['flow_estimator']:.4f}s")

            t_start = time.time()
            face_feat = self.face_encoder(source_img)
            timings['face_encoder'] = time.time() - t_start
            print(f"Face encoder time: {timings['face_encoder']:.4f}s")

            t_start = time.time()
            recon_imgs = self.face_generator(tgt_latent, face_feat)
            timings['face_generator'] = time.time() - t_start
            print(f"Face generator time: {timings['face_generator']:.4f}s")
        else:
            t_start = time.time()
            tgt_latent = self.motion_generate(masked_past_frames, audio_self, audio_other, video_length) # project target image to reference latent space
            timings['motion_generate'] = time.time() - t_start
            print(f"Motion generate time: {timings['motion_generate']:.4f}s")

            t_start = time.time()
            src_latent = self.motion_encoder(source_img) # project source image to reference latent space
            timings['motion_encoder'] = time.time() - t_start
            print(f"Motion encoder time: {timings['motion_encoder']:.4f}s")

            t_start = time.time()
            tgt_latent = self.flow_estimator(src_latent, tgt_latent) # navigate source to target in reference latent space
            timings['flow_estimator'] = time.time() - t_start
            print(f"Flow estimator time: {timings['flow_estimator']:.4f}s")

            t_start = time.time()
            face_feat = self.face_encoder(source_img)
            timings['face_encoder'] = time.time() - t_start
            print(f"Face encoder time: {timings['face_encoder']:.4f}s")

            t_start = time.time()
            recon_imgs = self.face_generator(tgt_latent, face_feat)
            timings['face_generator'] = time.time() - t_start
            print(f"Face generator time: {timings['face_generator']:.4f}s")

        total_time = sum(timings.values())
        print("\nModel Component Timings:")
        for component, t in timings.items():
            print(f"{component}: {t:.4f}s ({(t/total_time)*100:.1f}%)")
        print(f"Total time: {total_time:.4f}s")

        return recon_imgs

    def _step(self, batch):

        optimizer_g, optimizer_d = self.optimizers()

        ## train generator
        self.toggle_optimizer(optimizer_g)
        masked_target_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        masked_past_frames = batch['pixel_values_past_frames']
        masked_target_vid = torch.cat([masked_target_vid, masked_past_frames], dim=1)
        masked_ref_img = batch['pixel_values_ref_img']

        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']
        past_frames = batch['pixel_values_past_frames_original']
        target_vid_original = torch.cat([target_vid_original, past_frames], dim=1)

        # construct ref-tgt pairs
        masked_ref_img = masked_ref_img[:,None].repeat(1, masked_target_vid.size(1), 1, 1, 1)
        masked_ref_img = rearrange(masked_ref_img, "b t c h w -> (b t) c h w")
        masked_target_vid = rearrange(masked_target_vid, "b t c h w -> (b t) c h w")
        masked_past_frames = rearrange(masked_past_frames, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        audio_self = batch['target_wav_fea'].to(self.device)
        # TODO(wei): use audio other
        audio_other = torch.zeros_like(audio_self)
        # get reconstructed image
        predicted_img = self.forward(ref_img_original, masked_ref_img, masked_past_frames, audio_self, audio_other)

        if self.l_w_face > 0:
            eye_mouth_mask_vid = batch['eye_mouth_mask_vid']
            eye_mouth_mask_past_frames = batch['eye_mouth_mask_past_frames']
            face_mask = torch.cat([eye_mouth_mask_vid, eye_mouth_mask_past_frames], dim=1)
            face_mask = rearrange(face_mask, "b t c h w -> (b t) c h w")

            loss_dict = self.compute_loss(target_vid_original, predicted_img, face_mask)

        else:
            loss_dict = self.compute_loss(target_vid_original, predicted_img)

        if self.l_w_gan > 0:
            # adversarial loss
            pred_label = self.discriminator(predicted_img).reshape(-1)
            g_loss = self.l_w_gan * self.g_nonsaturating_loss(pred_label)

            loss_dict['loss'] += g_loss
            loss_dict['g_loss'] = g_loss

            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

            # import pdb; pdb.set_trace()

            ## train discriminator
            self.toggle_optimizer(optimizer_d)

            real_img_pred = self.discriminator(target_vid_original)
            recon_img_pred = self.discriminator(predicted_img.detach())

            d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)

            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)

            self.log("d_loss", d_loss, prog_bar=True)

        else:
            self.manual_backward(loss_dict['loss'])
            optimizer_g.step()
            optimizer_g.zero_grad()
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
