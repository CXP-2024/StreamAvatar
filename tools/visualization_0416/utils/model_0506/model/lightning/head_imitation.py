import torch
from torch import nn
from model.lightning.base_modules import BaseModule
from torch.utils.data import DataLoader, Dataset
from models.volumetric_avatar.img2vol_enc import LocalEncoder
from models.volumetric_avatar.warp_generator import WarpGenerator
from models.volumetric_avatar.warped_vol_dec import Decoder_stage2

class HeadImitationModule(BaseModule):
    def __init__(self, encoder, warp_generator, decoder, config):
        super().__init__(config)
        self.encoder = encoder
        self.warp_generator = warp_generator
        self.decoder = decoder
        self.config = config
        self.criterion = nn.MSELoss()  # TODO:loss
        self.learning_rate = config.get("learning_rate", 1e-4)

    def forward(self, source_img):
        latent_volume = self.encoder(source_img)
        warped_volume, deltas = self.warp_generator({"orig": latent_volume})
        output_img, _, _, _ = self.decoder({}, {}, warped_volume)
        return output_img

    def _step(self, batch):
        source_img, target_img = batch
        predicted_img = self.forward(source_img)
        loss = self.criterion(predicted_img, target_img)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class CustomDataset(Dataset):
    def __init__(self, source_images, target_images):
        self.source_images = source_images
        self.target_images = target_images

    def __len__(self):
        return len(self.source_images)

    def __getitem__(self, idx):
        return self.source_images[idx], self.target_images[idx]


def create_data_loaders(source_images, target_images, batch_size=16):
    dataset = CustomDataset(source_images, target_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    # TODO:config
    config = {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "num_epochs": 10,
    }

    encoder = LocalEncoder(
        use_amp_autocast=True,
        gen_upsampling_type="nearest",
        gen_downsampling_type="bilinear",
        gen_input_image_size=256,
        gen_latent_texture_size=64,
        gen_latent_texture_depth=8,
        gen_latent_texture_channels=64,
        warp_norm_grad=True,
        gen_num_channels=32,
        enc_channel_mult=1,
        norm_layer_type="bn",
        num_gpus=1,
        gen_max_channels=256,
        enc_block_type="res",
        gen_activation_type="relu",
        in_channels=3,
    )
    warp_generator = WarpGenerator(WarpGenerator.Config(
        eps=1e-8,
        num_gpus=1,
        gen_adaptive_conv_type="conv",
        gen_activation_type="relu",
        gen_upsampling_type="nearest",
        gen_downsampling_type="bilinear",
        gen_dummy_input_size=64,
        gen_latent_texture_depth=8,
        gen_latent_texture_size=64,
        gen_max_channels=256,
        gen_num_channels=32,
        gen_use_adaconv=False,
        gen_adaptive_kernel=False,
        gen_embed_size=32,
        warp_output_size=64,
        warp_channel_mult=1,
        warp_block_type="res",
        norm_layer_type="bn",
        input_channels=64,
    ))
    decoder = Decoder_stage2(
        eps=1e-8,
        image_size=256,
        use_amp_autocast=True,
        gen_embed_size=32,
        gen_adaptive_kernel=False,
        gen_adaptive_conv_type="conv",
        gen_latent_texture_size=64,
        in_channels=64,
        gen_num_channels=32,
        dec_max_channels=256,
        gen_use_adanorm=False,
        gen_activation_type="relu",
        gen_use_adaconv=False,
        dec_channel_mult=1,
        dec_num_blocks=4,
        dec_up_block_type="res",
        dec_pred_seg=False,
        dec_seg_channel_mult=1,
        dec_pred_conf=False,
        dec_conf_ms_names="",
        dec_conf_names="",
        dec_conf_ms_scales=4,
        dec_conf_channel_mult=1,
        gen_downsampling_type="bilinear",
        num_gpus=1,
        norm_layer_type="bn",
    )

    # TODO:data
    source_images = torch.randn(100, 3, 256, 256)
    target_images = torch.randn(100, 3, 256, 256)
    train_loader = create_data_loaders(source_images, target_images, batch_size=config["batch_size"])

    model = LightningModel(encoder, warp_generator, decoder, config)

    # training
    from lightning.pytorch import Trainer
    trainer = Trainer(max_epochs=config["num_epochs"], devices="auto", accelerator="gpu")
    trainer.fit(model, train_loader)
