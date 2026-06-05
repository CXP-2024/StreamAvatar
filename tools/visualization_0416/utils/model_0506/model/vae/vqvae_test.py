import unittest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from model.vae.vqvae import VQAutoEncoder

class TestVQAutoEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        config = {
            'model': {
                'encoder': {
                    'module_name': 'model.vae.cnn',
                    'class_name': 'Encoder2D',
                    'output_channels': 512
                },
                'decoder': {
                    'module_name': 'model.vae.cnn',
                    'class_name': 'Decoder2D',
                    'input_dim': 512
                },
                'latent_dim': 512
            },
            'optimizer': {
                'lr': 1e-4,
                'weight_decay': 0.0,
                'adam_beta1': 0.9,
                'adam_beta2': 0.999,
                'adam_epsilon': 1e-8
            },
            'loss': {
                'l_w_recon': 1.0,
                'l_w_embedding': 1.0,
                'l_w_recon': 1.0
            }
        }
        cls.config = OmegaConf.create(config)
        seed_everything(42)
        cls.model = VQAutoEncoder(cls.config)
        cls.model.configure_model()

    def test_model_initialization(self):
        """Test that the model and its components are initialized correctly."""
        self.assertIsInstance(self.model, VQAutoEncoder)
        self.assertIsInstance(self.model.encoder, nn.Module)
        self.assertIsInstance(self.model.decoder, nn.Module)
        self.assertTrue(hasattr(self.model, 'quantizer'))

    def test_encode_decode(self):
        """Test the encode and decode functions of the model."""
        batch_size = 2
        channels = 3
        height = 512  # Use 512x512 input to match the model architecture
        width = 512

        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)

        # Test encode
        quant, emb_loss, info = self.model.encode(x)
        self.assertEqual(quant.shape, (batch_size, 1, self.model.config.model.latent_dim))
        self.assertIsInstance(emb_loss, torch.Tensor)
        self.assertIsInstance(info, tuple)  # VectorQuantizer returns a tuple, not a dict

        # Test decode
        dec = self.model.decode(quant)
        self.assertEqual(dec.shape, (batch_size, channels, height, width))

    def test_forward(self):
        """Test the forward pass of the model."""
        batch_size = 2
        channels = 3
        height = 512  # Use 512x512 input to match the model architecture
        width = 512

        # Create dummy input
        x = torch.randn(batch_size, channels, height, width)

        # Test forward pass
        dec, emb_loss, info = self.model.forward(x)

        # Check output shapes and types
        self.assertEqual(dec.shape, (batch_size, channels, height, width))
        self.assertIsInstance(emb_loss, torch.Tensor)
        self.assertIsInstance(info, tuple)  # VectorQuantizer returns a tuple, not a dict

    def test_training_step(self):
        """Test the training step of the model."""
        batch_size = 2
        channels = 3
        height = 512  # Use 512x512 input to match the model architecture
        width = 512

        # Create dummy batch
        batch = {
            'pixel_values_vid': torch.randn(batch_size, channels, height, width)
        }

        # Test training step
        loss = self.model.training_step(batch)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)

    def test_validation_step(self):
        """Test the validation step of the model."""
        batch_size = 2
        channels = 3
        height = 512  # Use 512x512 input to match the model architecture
        width = 512

        # Create dummy batch
        batch = {
            'pixel_values_vid': torch.randn(batch_size, channels, height, width)
        }

        # Test validation step
        loss = self.model.validation_step(batch)
        self.assertIsInstance(loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main()
