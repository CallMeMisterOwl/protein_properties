from dataclasses import dataclass
from typing import Any, Literal
import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.functional.classification import f1_score, matthews_corrcoef, accuracy
from typing import Optional


class BfactorSASACNN(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 1,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 wing: int = 15,
                 num_lin_layers: int = 2,
                 size_lin_layers: list = [64, 16],
                 dropout: float = 0.1,
                 dilation: int = 1,
                 input_features: int = 2304,
                 **kwargs):
        super().__init__()
        assert num_lin_layers == len(size_lin_layers)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None

        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        
        # since we handle each sample in the batch separately and the samples are already padded to 31 we don't need padding
        # it might be smart though pad here instead of in the dataloader, then we would be able to use dilation 
        self.cnn = nn.Conv1d(
            input_features,
            input_features,
            wing * 2 + 1,
            #padding=wing * dilation,
            groups=input_features,
            #dilation=dilation
        )
        # create linear layers the first one has to have 1024 input features and the last one has to have num_classes output features
        # subsequent layers are defined by size_lin_layers
        # add relu and dropout after each layer
        self.linear_layers = nn.Sequential(
            nn.Linear(input_features, size_lin_layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(size_lin_layers[i], size_lin_layers[i+1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_lin_layers - 1)],
            nn.Linear(size_lin_layers[-1], num_classes if num_classes > 2 else 1)
        )

        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hparams["Modeltype"] = kwargs.get("task_type", 'BfactorSASACNN')
        self.save_hyperparameters()
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.cnn(x.permute(0, 2, 1))
        return self.linear_layers(out.permute(0, 2, 1))
        
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y > self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected {_[0]}")
            return
            
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        print('Val step yas')
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y > self.mask_value)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y > self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def predict_step(self, batch, batch_idx):
        x, y, pids = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()    
        return y_hat.detach().numpy(), y.detach().numpy(), pids
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3, factor=0.5, verbose=True, threshold=0.001)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer], [{"scheduler": self._configure_scheduler(optimizer), "interval": "epoch", "monitor": "val_loss"}]

    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
    
class BFactorBaseline(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 1,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 input_features: int = 2304,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None
        self.model = nn.Sequential(
            nn.Linear(input_features, self.num_classes if self.num_classes > 2 else 1),
        )
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        if self.num_classes == 1:
            torch.nn.init.kaiming_uniform_(self.model[0].weight)
            self.model[0].bias.data.fill_(0.01)

        self.hparams["Modeltype"] = "BfactorBaseline"
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.save_hyperparameters()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Identity() 
        

    def forward(self, x):
        x_out = self.model(x)
        return self.activation(x_out)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=1)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        #raise ValueError(f"Invalid optimizer {optim_config.optimize}. See --help")

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer]#, [{"schduler": self._configure_scheduler(optimizer), "interval": "epoch"}]
    
    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)

class SASABaseline(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 3,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None
        self.model = nn.Sequential(
            nn.Linear(1024, self.num_classes if self.num_classes > 2 else 1),
        )
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        if self.num_classes == 1:
            torch.nn.init.xavier_uniform_(self.model[0].weight)
            self.model[0].bias.data.fill_(0.01)

        self.hparams["Modeltype"] = "SASABaseline"
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.save_hyperparameters()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Identity() if self.num_classes != 1 else nn.Sigmoid()
        

    def forward(self, x):
        x_out = self.model(x)
        return self.activation(x_out)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        #raise ValueError(f"Invalid optimizer {optim_config.optimize}. See --help")

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer]#, [{"schduler": self._configure_scheduler(optimizer), "interval": "epoch"}]
    
    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
    
class SASALSTM(pl.LightningModule):

    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 3,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 hidden: int = 20,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 input_features: int = 1024,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None
        self.num_layers = num_layers

        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        
        self.lstm_layer = nn.LSTM(input_size=input_features,
                               hidden_size=hidden,
                               bidirectional=True,
                               num_layers=num_layers,
                               dropout=dropout)

        # HIDDEN times 2 because of bidirectionality
        self.out = nn.Linear(hidden*2, self.num_classes if self.num_classes > 2 else 1)

        
        self.dropout = nn.Dropout(dropout) if self.num_layers == 1 else nn.Identity() 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hparams["Modeltype"] = "SASALSTM"
        self.save_hyperparameters()
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_h, (_, _) = self.lstm_layer(x)
        lstm_h = self.dropout(lstm_h)
        # applies the linear layer to every LSTM time step
        return torch.stack([self.out(pre) for pre in lstm_h])

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
        

    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer], [{"scheduler": self._configure_scheduler(optimizer), "interval": "epoch", "monitor": "val_loss"}]

    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)

class SASACNN(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 1,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-5,
                 weight_decay: float = 0.0,
                 wing: int = 15,
                 num_lin_layers: int = 3,
                 size_lin_layers: list = [128, 64, 16],
                 dropout: float = 0.1,
                 dilation: int = 1,
                 input_features: int = 2304,
                 **kwargs):
        super().__init__()
        assert num_lin_layers == len(size_lin_layers)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None

        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        
        self.cnn = nn.Conv1d(
            input_features,
            input_features,
            wing * 2 + 1,
            padding=wing * dilation,
            groups=input_features,
            dilation=dilation
        )
        # create linear layers the first one has to have 1024 input features and the last one has to have num_classes output features
        # subsequent layers are defined by size_lin_layers
        # add relu and dropout after each layer
        self.linear_layers = nn.Sequential(
            nn.Linear(input_features, size_lin_layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(size_lin_layers[i], size_lin_layers[i+1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_lin_layers - 1)],
            nn.Linear(size_lin_layers[-1], num_classes if num_classes > 2 else 1)
        )

        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hparams["Modeltype"] = "SASACNN"
        self.save_hyperparameters()
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # TODO remove this
        out = self.cnn(x.permute(0, 2, 1))
        #out = nn.ReLU()(out)
        
        # applies the linear layer to every CNN step
        return self.linear_layers(out.permute(0, 2, 1))
        
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y > self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected {_[0]}")
            return
            
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y > self.mask_value)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y > self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()    
        mask = (y > self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])    
        metrics = self._accuracy(y_hat[mask], y[mask])
        
        return y_hat[mask].squeeze().cpu().numpy().flatten(), loss.cpu().item(), metrics[0][1], metrics[1][1]
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3, factor=0.5, verbose=True, threshold=0.001)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer], [{"scheduler": self._configure_scheduler(optimizer), "interval": "epoch", "monitor": "val_loss"}]

    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)

class BfactorCNN(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[1, 2, 3, 10] = 1,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-5,
                 weight_decay: float = 0.0,
                 wing: int = 15,
                 num_lin_layers: int = 3,
                 size_lin_layers: list = [128, 64, 16],
                 dropout: float = 0.1,
                 dilation: int = 1,
                 input_features: int = 2304,
                 **kwargs):
        super().__init__()
        assert num_lin_layers == len(size_lin_layers)
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None

        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")
        
        self.cnn = nn.Conv1d(
            input_features,
            input_features,
            wing * 2 + 1,
            padding=wing * dilation,
            groups=input_features,
            dilation=dilation
        )
        # create linear layers the first one has to have 1024 input features and the last one has to have num_classes output features
        # subsequent layers are defined by size_lin_layers
        # add relu and dropout after each layer
        self.linear_layers = nn.Sequential(
            nn.Linear(input_features, size_lin_layers[0]),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(size_lin_layers[i], size_lin_layers[i+1]),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_lin_layers - 1)],
            nn.Linear(size_lin_layers[-1], num_classes if num_classes > 2 else 1)
        )

        self.dropout = dropout
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hparams["Modeltype"] = "BfactorCNN"
        self.save_hyperparameters()
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # TODO remove this
        out = self.cnn(x.permute(0, 2, 1))
        #out = nn.ReLU()(out)
        
        # applies the linear layer to every CNN step
        return self.linear_layers(out.permute(0, 2, 1))
        
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected {_[0]}")
            return
            
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected {_[0]}")
            return
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()    
        mask = (y != self.mask_value)
        
        loss = self._loss(y_hat[mask], y[mask])    
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf loss detected {_[0]}")
            print(y_hat, y)
            return 
        metrics = self._accuracy(y_hat[mask], y[mask])
        
        return y_hat[mask].squeeze().cpu().numpy().flatten(), loss.cpu().item(), metrics[0][1], metrics[1][1]
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3, factor=0.5, verbose=True, threshold=0.001)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer], [{"scheduler": self._configure_scheduler(optimizer), "interval": "epoch", "monitor": "val_loss"}]

    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
            ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
    
    
class EnsembleModel(pl.LightningModule):
    def __init__(self, models, num_classes=1, device="cuda"):
        super().__init__()
        self.models = models
        self.models = [model.to(device) for model in self.models]
        self.num_classes = num_classes
        self.hparams["Modeltype"] = "EnsembleModel"
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        
        # mean of all predictions
        return torch.mean(torch.stack([model(x).squeeze() if self.num_classes < 3 else self.softmax(model(x)) for model in self.models]), dim=0)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != self.mask_value)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes == 1:
            return y_hat[mask].squeeze().cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(y_hat[mask].squeeze()).cpu().numpy().flatten(), y[mask].cpu().numpy().squeeze()
        elif self.num_classes > 2:
            # For multiclass predictions don't
            return self.softmax(y_hat[mask].squeeze()).cpu().numpy(), y[mask].cpu().numpy().squeeze()
    
    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
        ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y)
        return F.cross_entropy(y_hat, y)

class SASADummyModel(pl.LightningModule):
    def __init__(self, num_classes, label_distribution: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.label_distribution = label_distribution if label_distribution is not None else None
        self.mask_value = -1 if self.num_classes > 1 else -1.0
        self.hparams["Modeltype"] = "SASADummyModel"

    def forward(self, x):
        num_samples = x.shape[1]
        if self.num_classes == 1:
            # randomly pull from label distribution
            return self.label_distribution[torch.randint(0, len(self.label_distribution) - 1, (num_samples,))]
        
        if self.label_distribution is not None:
            if self.num_classes > 2:
         # Randomly draw indices based on label distribution
                torch.use_deterministic_algorithms(False)
                label_indices = torch.multinomial(self.label_distribution, num_samples, replacement=True)
                torch.use_deterministic_algorithms(True)
                
                # Convert label indices to one-hot representation
                logits = torch.zeros(num_samples, self.num_classes)
                logits[:, label_indices - 1] = 1.0
            
            else:
                # Binary classification
                p = self.label_distribution.item()  # Probability of positive class
                logits = torch.empty(num_samples, 1).uniform_(0, 1)
                logits = torch.where(logits >= p, 1.0, 0.0)
        else:
            # Generate random logits
            logits = torch.randn(num_samples, self.num_classes if self.num_classes != 2 else 1)

        return logits.to(x.device)

    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()    
        mask = (y > self.mask_value).cpu()
        loss = self._loss(y_hat[mask], y[mask])    
        metrics = self._accuracy(y_hat[mask], y[mask])
        
        return y_hat[mask].squeeze().cpu().numpy().flatten(), loss.cpu().item(), metrics[0][1], metrics[1][1]
    
    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        if self.num_classes != 1:
            return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
            ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
        ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        else:
            return [("MAE", F.l1_loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]

    def _loss(self, y_hat, y):
        
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y)
        return F.cross_entropy(y_hat, y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
        )

class GlycoModel(pl.LightningModule):
    def __init__(self,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-4,
                 weight_decay: float = 0.0,
                 num_classes: int = 3,
                 num_layers: int = 1,
                 input_dim: int = 1024,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None
        if num_layers == 1:
            self.model = nn.Linear(input_dim, self.num_classes if self.num_classes > 2 else 1)
        else:
            num_hidden = kwargs.get("num_hidden", [54])
            dropout = kwargs.get("dropout", 0.5)
            self.model = nn.Sequential(
                nn.Linear(input_dim, num_hidden[0]),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                *[nn.Sequential(
                    nn.Linear(num_hidden[i], num_hidden[i+1] if i < len(num_hidden) - 1 else self.num_classes if self.num_classes > 2 else 1),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout)
                ) for i in range(num_layers - 2)],
                nn.Linear(num_hidden[-1], self.num_classes if self.num_classes > 2 else 1)
            )

        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")

        self.hparams["Modeltype"] = "GlycoModel"
        self.save_hyperparameters()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.val_preds = []
        self.val_ys = []

        self.test_preds = []
        self.test_ys = []
        

    def forward(self, x):

        return self.sigmoid(self.model(x)) if self.num_classes == 2 else self.softmax(self.model(x))
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        if x.shape[1] == 1:
            x = x.squeeze(0)
        y = y.squeeze()
        y_hat = self(x).squeeze()   
        loss = self._loss(y_hat, y)
        if len(y_hat.shape) == 1 or y_hat.shape == torch.Size([]):
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat, y):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False, batch_size=y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        if x.shape[1] == 1:
            x = x.squeeze(0)
        y = y.squeeze()
        y_hat = self(x).squeeze()
        loss = self._loss_test(y_hat, y)
        if len(y_hat.shape) == 1 or y_hat.shape == torch.Size([]):
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_preds.append(y_hat)
        self.val_ys.append(y)
        return loss

    def on_validation_epoch_end(self):
        y_hat = torch.cat(self.val_preds, dim=0)
        y = torch.cat(self.val_ys, dim=0)
        for t in self._accuracy(y_hat, y):
            self.log(f"val_{t[0]}", t[1], on_epoch=True)
        self.val_preds.clear()
        self.val_ys.clear()
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        if x.shape[1] == 1:
            x = x.squeeze(0)
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        loss = self._loss_test(y_hat, y)
        if len(y_hat.shape) == 1 or y_hat.shape == torch.Size([]):
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        self.test_preds.append(y_hat)
        self.test_ys.append(y)

        if self.num_classes == 2:
            if len(y_hat.shape) == 1 or y_hat.shape == torch.Size([]):
                y_hat = y_hat.unsqueeze(0)
                y = y.unsqueeze(0)
            return y_hat.cpu().numpy().squeeze(0), y.cpu().numpy().squeeze(0)
        
        return y_hat.cpu().numpy(), y.cpu().numpy()
    
    def predict_step(self, batch, batch_idx):
        x, y, _ = batch
        if x.shape[1] == 1:
            x = x.squeeze(0)
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        loss = self._loss_test(y_hat, y)
        y_hat = y_hat > 0.5 if self.num_classes == 2 else y_hat.argmax()
        return y_hat.cpu().item(), loss.cpu().item()
    
    def on_test_epoch_end(self):
        y_hat = torch.cat(self.test_preds, dim=0)
        y = torch.cat(self.test_ys, dim=0)
        for t in self._accuracy(y_hat, y):
            self.log(f"test_{t[0]}", t[1], on_epoch=True)
        self.test_preds.clear()
        self.test_ys.clear()
        
    
    def _configure_optimizer(self, optim_config = None):
        
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        #raise ValueError(f"Invalid optimizer {optim_config.optimize}. See --help")

    def _configure_scheduler(self, optimizer: torch.optim.Optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", patience=3)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        return [optimizer]#, [{"schduler": self._configure_scheduler(optimizer), "interval": "epoch"}]
    
    def _accuracy(self, y_hat, y):
        metrics = []
        task = "binary" if self.num_classes == 2 else "multiclass"
        return [("MCC", matthews_corrcoef(y_hat, y, task=task, num_classes=self.num_classes)), 
        ("ACC", accuracy(y_hat, y, task=task, num_classes=self.num_classes, average="macro")),
        ("F1", f1_score(y_hat, y, task=task, num_classes=self.num_classes, average="macro"))]
        
    def _loss_test(self, y_hat, y):
        if len(y_hat.shape) != 2:
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y)
        return F.cross_entropy(y_hat, y)
        
    def _loss(self, y_hat, y):
        if len(y_hat.shape) != 2:
            y_hat = y_hat.unsqueeze(0)
            y = y.unsqueeze(0)
        if self.loss_fn is not None: 
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
