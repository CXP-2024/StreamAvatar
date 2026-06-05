from omegaconf import DictConfig
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from utils import instantiate


class BaseModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model = self.configure_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
