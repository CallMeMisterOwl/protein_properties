from dataclasses import dataclass
from typing import Any, Literal
import pytorch_lightning as pl
import torch.functional as F
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score
    

class SASABaseline(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[2,3,10] = 3,
                 class_weights: Any = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 loss_fn: Any = None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn
        self.model = nn.Sequential(
            nn.Linear(1024, self.num_classes),
        )

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss)
        self.log("train_f1", F1(y_hat[mask], y[mask]), on_epoch=True)
        self.log("train_acc", Accuracy(y_hat[mask], y[mask]), on_epoch=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss)
        self.log("val_f1", F1(y_hat[mask], y[mask]), on_epoch=True)
        self.log("val_acc", Accuracy(y_hat[mask], y[mask]), on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss)
        self.log("test_f1", F1(y_hat[mask], y[mask]), on_epoch=True)
        self.log("test_acc", Accuracy(y_hat[mask], y[mask]), on_epoch=True)
        return loss
    
    def _configure_optimizer(self, optim_config):
        if optim_config.optimize == "Adam":
            return torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        raise ValueError(f"Invalid optimizer {optim_config.optimize}. See --help")

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer]#, [{"schduler": self._configure_scheduler(optimizer), "interval": "epoch"}]
    
    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
    
    