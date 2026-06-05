import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lightning.base_modules import BaseModule
from omegaconf import DictConfig
from typing import Any, Dict, Tuple
from utils import instantiate
import cv2
from PIL import Image
import numpy as np


class VQAutoEncoder(BaseModule):
    """ VQ-VAE model """
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__(config)

        self.config = config

        self.l_w_recon = config.loss.l_w_recon
        self.l_w_embedding = config.loss.l_w_embedding
        self.l_w_commitment = config.loss.l_w_commitment
        self.mse_loss = nn.MSELoss()

    def configure_model(self):
        config = self.config
        self.encoder = instantiate(config.model.encoder)

        self.decoder = instantiate(config.model.decoder)
        # self.quantizer = instantiate(config.model.quantizer)

        # VQ Embedding (Vector Quantization) layer
        self.vq_embedding = nn.Embedding(config.model.n_embedding, config.model.latent_dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / config.model.latent_dim, 1.0 / config.model.latent_dim)  # Random initialization

    def configure_optimizers(self) -> Dict[str, Any]:
        params_to_update = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        return {"optimizer": optimizer}

    def encode(self, image):
        ze = self.encoder(image)

        # Vector Quantization
        embedding = self.vq_embedding.weight.data
        B, C, H, W = ze.shape
        K, _ = embedding.shape
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        ze_broadcast = ze.reshape(B, 1, C, H, W)
        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)
        nearest_neighbor = torch.argmin(distance, 1)

        # Quantized features
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        return ze, zq

    def decode(self, quantized_fea):
        x_hat = self.decoder(quantized_fea)
        return x_hat

    def _step(self, batch, return_loss=True):
        pixel_values_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        pixel_values_vid = pixel_values_vid.view(-1, 3,  pixel_values_vid.size(-2),  pixel_values_vid.size(-1)) # [B, T, C, H, W] -> [B*T, C, H, W]

        # import cv2
        # cv2.imwrite('debug_img.png', 255*pixel_values_vid[-1].permute(1,2,0).cpu().numpy()[:,:,::-1])
        # import pdb; pdb.set_trace()

        # test on single image
        # pixel_values_vid = Image.open('debug_img.png')
        # pixel_values_vid = np.array(pixel_values_vid) / 255.0
        # pixel_values_vid = torch.from_numpy(pixel_values_vid).float().to(self.device)[None].permute(0, 3, 1, 2)

        # Encoding
        hidden_fea, quantized_fea = self.encode(self, pixel_values_vid)

        # Stop gradient
        decoder_input = hidden_fea + (quantized_fea - hidden_fea).detach()

        # Decoding
        x_hat = self.decode(decoder_input)

        if return_loss:
            # Reconstruction Loss
            l_reconstruct = self.mse_loss(x_hat, pixel_values_vid)

            # Embedding Loss
            l_embedding = self.mse_loss(hidden_fea.detach(), quantized_fea)

            # Commitment Loss
            l_commitment = self.mse_loss(hidden_fea, quantized_fea.detach())

            # Total Loss
            total_loss = l_reconstruct + self.l_w_embedding * l_embedding + self.l_w_commitment * l_commitment

            self.log('recon_loss', l_reconstruct, on_step=True, on_epoch=True, prog_bar=True)
            self.log('emb_loss', l_embedding, on_step=True, on_epoch=True, prog_bar=True)
            self.log('commit_loss', l_commitment, on_step=True, on_epoch=True, prog_bar=True)

            return total_loss
        else:
            return x_hat, pixel_values_vid

    def training_step(self, batch):
        total_loss = self._step(batch)
        return total_loss

    def validation_step(self, batch):
        total_loss = self._step(batch)
        return total_loss

    def forward(self, batch):
        x_pred, x_gt = self._step(batch, return_loss=False)

        return x_pred, x_gt
