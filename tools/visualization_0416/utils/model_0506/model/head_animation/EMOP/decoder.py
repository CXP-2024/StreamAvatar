import torch
from torch import nn
from torch import optim
from torch.cuda import amp
import torch.nn.functional as F
import math
import numpy as np
import itertools
import copy
from torch.cuda import amp
from scipy import linalg
from . import utils
from .utils import ProjectorConv, ProjectorNorm, ProjectorNormLinear, assign_adaptive_conv_params, \
    assign_adaptive_norm_params, Upsample_sg2
from dataclasses import dataclass
# from .sg3_generator import Generator


class Decoder(nn.Module):

    @dataclass
    class Config:
        eps : float
        image_size : int
        gen_embed_size : int
        gen_adaptive_kernel : bool
        gen_adaptive_conv_type: str
        gen_latent_texture_size: int
        in_channels: int
        gen_num_channels: int
        dec_max_channels: int
        gen_use_adanorm: bool
        gen_activation_type: str
        gen_use_adaconv: bool
        dec_channel_mult: float
        dec_num_blocks: int
        dec_up_block_type: str
        dec_pred_seg: bool
        dec_seg_channel_mult: float
        num_gpus: int
        norm_layer_type: str
        bigger: bool = False
        vol_render: bool = False
        im_dec_num_lrs_per_resolution: int = 1
        im_dec_ch_div_factor: float = 2.0
        emb_v_exp: bool = False
        dec_use_sg3_img_dec: bool = False
        no_detach_frec: int = 10
        dec_key_emb: str = 'orig'

    def __init__(self, cfg:Config):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.adaptive_conv_type = self.cfg.gen_adaptive_conv_type
        num_blocks = self.cfg.dec_num_blocks
        num_up_blocks = int(math.log(self.cfg.image_size // self.cfg.gen_latent_texture_size, 2))
        self.in_channels = self.cfg.in_channels
        out_channels = min(int(self.cfg.gen_num_channels * self.cfg.dec_channel_mult * 2**num_up_blocks), self.cfg.dec_max_channels)
        # print(num_up_blocks, out_channels)
        self.gen_max_channels = self.cfg.dec_max_channels
        # self.num_gpus = self.cfg.num_gpus
        self.norm_layer_type = self.cfg.norm_layer_type
        norm_layer_type = self.cfg.norm_layer_type

        # if norm_layer_type == 'bn':
        #     if self.num_gpus > 1:
        #         norm_layer_type = 'sync_' + norm_layer_type
        # if self.cfg.gen_use_adanorm:
        #     norm_layer_type = 'ada_' + norm_layer_type

        # print(norm_layer_type)
        if self.cfg.vol_render:
            layers = []
        else:
            layers = [
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    bias=False)
            ]

        for i in range(num_blocks):
            layers += [
                utils.blocks['res'](
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_layer_type=norm_layer_type,
                    activation_type=self.cfg.gen_activation_type,
                    conv_layer_type=('ada_' if self.cfg.gen_use_adaconv else '') + 'conv')]


        self.res_decoder = nn.Sequential(*layers)

        if self.cfg.dec_use_sg3_img_dec:
            self.img_decoder = Generator(
                512,
                512,
                512,
                3,
                num_layers=14,  # Total number of layers, excluding Fourier features and ToRGB. # NOTE Original 6
                num_critical=2,  # Number of critically sampled layers at the end.
                first_cutoff=10.079,  # Cutoff frequency of the first layer (f_{c,0}). # NOTE Original 2.0
                first_stopband=17.959,  # Minimum stopband of the first layer (f_{t,0}). # NOTE Original 2**3.1
                last_stopband_rel=2**0.3,  # Minimum stopband of the last layer, expressed relative to the cutoff.
                margin_size=16,  # Number of additional pixels outside the image.
                num_fp16_res=False,  # Use FP16 for the N highest resolutions.
            )
        else:
            self.img_decoder = ImageDecoder(
                self.cfg.image_size,
                self.cfg.gen_latent_texture_size,
                self.cfg.gen_use_adanorm,
                self.cfg.gen_num_channels,
                self.cfg.dec_up_block_type,
                self.cfg.gen_activation_type,
                self.cfg.gen_use_adaconv,
                self.cfg.dec_pred_seg,
                self.cfg.dec_seg_channel_mult,
                out_channels,
                # self.num_gpus,
                norm_layer_type=norm_layer_type,
                bigger=self.cfg.bigger,
                im_dec_num_lrs_per_resolution = self.cfg.im_dec_num_lrs_per_resolution,
                im_dec_ch_div_factor = self.cfg.im_dec_ch_div_factor
                )


        self.gen_use_adanorm = self.cfg.gen_use_adanorm
        if self.cfg.gen_use_adanorm:
            # self.projector = ProjectorNormLinear(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
            #                                gen_embed_size=32,
            #                                gen_max_channels=64)

            # self.projector = ProjectorNorm(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
            #                                 gen_embed_size=self.cfg.gen_embed_size, gen_max_channels=self.gen_max_channels,)
            self.projector = ProjectorNormLinear(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                           gen_embed_size=self.cfg.gen_embed_size,
                                           gen_max_channels=self.gen_max_channels, emb_v_exp = self.cfg.emb_v_exp, no_detach_frec=self.cfg.no_detach_frec, key_emb=self.cfg.dec_key_emb)
        else:
            self.projector = ProjectorNorm(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                           gen_embed_size=self.cfg.gen_embed_size,
                                           gen_max_channels=self.gen_max_channels)

        # print(sum(p.numel() for p in self.res_decoder.parameters() if p.requires_grad), sum(p.numel() for p in self.img_decoder.parameters() if p.requires_grad))
        if self.cfg.gen_use_adaconv:
            self.projector_conv = ProjectorConv(net_or_nets=[self.res_decoder, self.img_decoder], eps=self.cfg.eps,
                                                gen_adaptive_kernel=self.cfg.gen_adaptive_kernel,
                                                gen_max_channels=self.gen_max_channels)

    def forward(self, data_dict, embed_dict, feat_2d, input_flip_feat=False, annealing_alpha=0.0, embed=None, stage_two=False, iteration=0):
        if self.gen_use_adanorm:
            # b, c, es, _ = data_dict['ada_v'].shape
            # params_norm = self.projector(data_dict['ada_v'].view(b, c, es ** 2))

            params_norm = self.projector(embed_dict, iter=iteration)
            annealing_alpha = 1
            # print('aaaa')
        else:
            params_norm = self.projector(embed_dict, iter=iteration)

        if input_flip_feat:
            # Repeat params for flipped feat
            params_norm_ = []
            for param in params_norm:
                if isinstance(param, tuple):
                    params_norm_.append((torch.cat([p] * 2) for p in param))
                else:
                    params_norm_.append(torch.cat([param] * 2))
        else:
            params_norm_ = params_norm

        assign_adaptive_norm_params([self.res_decoder, self.img_decoder], params_norm_, annealing_alpha)

        if hasattr(self, 'projector_conv'):
            params_conv = self.projector_conv(embed_dict)

            if input_flip_feat:
                # Repeat params for flipped feat
                params_conv_ = []
                for param in params_conv:
                    if isinstance(param, tuple):
                        params_conv_.append((torch.cat([p] * 2) for p in param))
                    else:
                        params_conv_.append(torch.cat([param] * 2))
            else:
                params_conv_ = params_conv

            assign_adaptive_conv_params([self.res_decoder, self.img_decoder], params_conv, self.adaptive_conv_type, annealing_alpha)

        feat_2d = self.res_decoder(feat_2d)
        img, seg, img_f = self.img_decoder(feat_2d, stage_two=stage_two)

        # Predict conf
        if hasattr(self, 'conf_decoder') and self.training and input_flip_feat:
            feat, feat_flip = feat_2d.split(feat_2d.shape[0] // 2)

            conf_ms, conf_ms_flip, conf, conf_flip = self.conf_decoder(feat, feat_flip)

            for conf_ms_k, conf_ms_flip_k, conf_name in zip(conf_ms, conf_ms_flip, self.conf_ms_names):
                data_dict[f'{conf_name}_ms'] = conf_ms_k
                data_dict[f'{conf_name}_flip_ms'] = conf_ms_flip_k

                data_dict[conf_name] = conf_ms_k[0]
                data_dict[f'{conf_name}_flip'] = conf_ms_flip_k[0]

            for conf_k, conf_flip_k, conf_name in zip(conf, conf_flip, self.conf_names):
                data_dict[f'{conf_name}'] = conf_k
                data_dict[f'{conf_name}_flip'] = conf_flip_k

        if stage_two:
            return img, seg, feat_2d, img_f
        else:
            return img, seg, None, None

class ImageDecoder(nn.Module):
    def __init__(self,
                 image_size,
                 gen_latent_texture_size,
                 gen_use_adanorm,
                 gen_num_channels,
                 dec_up_block_type,
                 gen_activation_type,
                 gen_use_adaconv,
                 dec_pred_seg,
                 dec_seg_channel_mult,
                 shared_in_channels,
                #  num_gpus,
                 norm_layer_type,
                 bigger=False,
                 im_dec_num_lrs_per_resolution=1,
                 im_dec_ch_div_factor = 2
                 ):
        super(ImageDecoder, self).__init__()
        num_up_blocks = int(math.log(image_size // gen_latent_texture_size, 2))
        out_channels = shared_in_channels
        self.bigger = bigger
        self.im_dec_num_lrs_per_resolution = im_dec_num_lrs_per_resolution

        layers = []

        if self.bigger:
            num_up_blocks = num_up_blocks - 1

        for i in range(num_up_blocks):
            in_channels = out_channels
            # out_channels = max(out_channels // 2, gen_num_channels)
            out_channels = max(int(out_channels / im_dec_ch_div_factor/32)*32, gen_num_channels)


            if self.bigger:
                out_channels = max(out_channels, 256)
            # out_channels = max(out_channels, gen_num_channels)

            # if out_channels%32!=0:
            #     c_norm_layer_type = 'gn_24'
            # else:
            #     c_norm_layer_type = norm_layer_type
            k=0
            for _ in range(self.im_dec_num_lrs_per_resolution):
                layers += [
                    utils.blocks[dec_up_block_type](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2 if k==0 else 1,
                        norm_layer_type=norm_layer_type,
                        activation_type=gen_activation_type,
                        conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                        resize_layer_type='nearest' if k==0 else 'none'),
                ]
                in_channels = out_channels
                k+=1

        if self.bigger:
            layers += [
                utils.blocks[dec_up_block_type](
                    in_channels=out_channels,
                    out_channels=out_channels//2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

                utils.blocks[dec_up_block_type](
                    in_channels=out_channels//2,
                    out_channels=out_channels//2,
                    stride=2,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',

                    resize_layer_type='nearest'),
                utils.blocks[dec_up_block_type](
                    in_channels=out_channels // 2,
                    out_channels=out_channels // 4,
                    norm_layer_type=norm_layer_type,
                    activation_type=gen_activation_type,
                    conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv'),

            ]
            out_channels = out_channels // 4

        self.dec_img_blocks = nn.Sequential(*layers)

        layers = [
            utils.norm_layers[norm_layer_type](out_channels),
            utils.activations[gen_activation_type](inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=3,
                kernel_size=1),
            nn.Sigmoid()]

        self.dec_img_head = nn.Sequential(*layers)

        if dec_pred_seg:
            in_channels = shared_in_channels
            out_channels = int(gen_num_channels * dec_seg_channel_mult * 2**num_up_blocks)

            layers = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False)]

            for i in range(num_up_blocks):
                in_channels = out_channels
                out_channels = max(out_channels // 2, int(gen_num_channels * dec_seg_channel_mult))
                layers += [
                    utils.blocks[dec_up_block_type](
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=2,
                        norm_layer_type=norm_layer_type,
                        activation_type=gen_activation_type,
                        conv_layer_type=('ada_' if gen_use_adaconv else '') + 'conv',
                        resize_layer_type='nearest')]

    def forward(self, feat, stage_two=False):
        img_feat = self.dec_img_blocks(feat)
        img = self.dec_img_head(img_feat.float())

        seg = None
        if hasattr(self, 'dec_seg_blocks'):
            seg_feat = self.dec_seg_blocks(feat)
            seg = self.dec_seg_head(seg_feat.float())

        if stage_two:
            return img, None, img_feat
        else:
            return img, None, None

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img
