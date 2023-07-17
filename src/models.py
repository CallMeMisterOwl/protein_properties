from dataclasses import dataclass
from typing import Any, Literal
import lightning.pytorch as pl
import torch.nn.functional as F
import torch
from torch import nn
from torchmetrics.functional.regression import pearson_corrcoef
from torchmetrics.functional.classification import binary_f1_score, multiclass_f1_score, binary_accuracy, multiclass_accuracy, matthews_corrcoef
from typing import Optional
    

class SASABaseline(pl.LightningModule):
    def __init__(self, 
                 num_classes: Literal[2, 3, 10] = 3,
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

        self.hparams["Modeltype"] = "SASABaseline"
        self.save_hyperparameters()
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
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
        if self.num_classes == 2:
            return [("MCC", matthews_corrcoef(y_hat, y, task="binary", num_classes=self.num_classes)), ("ACC", binary_accuracy(y_hat, y))]
        if self.num_classes == 1:
            return [("MAE", nn.L1Loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]
        return [("MCC", matthews_corrcoef(y_hat, y, task="multiclass", num_classes=self.num_classes)), 
        ("ACC", multiclass_accuracy(y_hat, y, num_classes=self.num_classes))]

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
                 num_classes: Literal[2, 3, 10] = 3,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 hidden: int = 20,
                 num_layers: int = 1,
                 dropout: float = 0.0,
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
        
        self.lstm_layer = nn.LSTM(input_size=1024,
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


    def forward(self, x):
        lstm_h, (_, _) = self.lstm_layer(x)
        lstm_h = self.dropout(lstm_h)
        # applies the linear layer to every LSTM time step
        return torch.stack([self.out(pre) for pre in lstm_h])#, torch.stack([self.uncertainty(pre) for pre in lstm_h])

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

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
        if self.num_classes == 2:
            return [("MCC", matthews_corrcoef(y_hat, y, task="binary", num_classes=self.num_classes)), ("ACC", binary_accuracy(y_hat, y))]
        if self.num_classes == 1:
            return [("MAE", nn.L1Loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]
        return [("MCC", matthews_corrcoef(y_hat, y, task="multiclass", num_classes=self.num_classes)), 
        ("ACC", multiclass_accuracy(y_hat, y, num_classes=self.num_classes))]

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
                 num_classes: Literal[2, 3, 10] = 3,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 wing: int = 15,
                 num_lin_layers: int = 2,
                 size_lin_layers: list = [256, 152],
                 dropout: float = 0.5,
                 dilation: int = 1,
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
            1024,
            1024,
            wing * 2 + 1,
            padding=wing * dilation,
            groups=1024,
            dilation=dilation
        )
        # create linear layers the first one has to have 1024 input features and the last one has to have num_classes output features
        # subsequent layers are defined by size_lin_layers
        # add relu and dropout after each layer
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, size_lin_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(size_lin_layers[i], size_lin_layers[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_lin_layers - 1)],
            nn.Linear(size_lin_layers[-1], num_classes if num_classes > 2 else 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.hparams["Modeltype"] = "SASACNN"
        self.save_hyperparameters()

    def forward(self, x):
        # TODO remove this
        out = self.cnn(x.permute(0, 2, 1))
        #out = nn.ReLU()(out)
        out = F.dropout(out, p=0.2, training=self.training)
        # applies the linear layer to every CNN step
        return self.linear_layers(out.permute(0, 2, 1))
        
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        if self.num_classes < 3:
            # For binary predictions flatten the array
            return self.sigmoid(pred.squeeze()).cpu().numpy().flatten(), y.cpu().numpy().squeeze()
        else:
            # For multiclass predictions don't
            return self.softmax(pred.squeeze()).cpu().numpy(), y.cpu().numpy().squeeze()
        
    
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
        if self.num_classes == 2:
            return [("MCC", matthews_corrcoef(y_hat, y, task="binary", num_classes=self.num_classes)), ("ACC", binary_accuracy(y_hat, y))]
        if self.num_classes == 1:
            return [("MAE", nn.L1Loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]
        return [("MCC", matthews_corrcoef(y_hat, y, task="multiclass", num_classes=self.num_classes)), 
        ("ACC", multiclass_accuracy(y_hat, y, num_classes=self.num_classes))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)

class SASADummyModel(pl.LightningModule):
    def __init__(self, num_classes, label_distribution: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.label_distribution = label_distribution if label_distribution is not None else None
        

    def forward(self, x):
        num_samples = x.shape[1]

        if self.num_classes == 1:
            # return zeros for regression
            return torch.zeros(num_samples, 1)
        if self.label_distribution is not None:
            if self.num_classes > 2:
         # Randomly draw indices based on label distribution
                torch.use_deterministic_algorithms(False)
                label_indices = torch.multinomial(self.label_distribution, num_samples, replacement=True)
                torch.use_deterministic_algorithms(True)
                
                # Convert label indices to one-hot representation
                logits = torch.zeros(num_samples, self.num_classes if self.num_classes != 2 else 1)
                logits[torch.arange(num_samples), label_indices] = 1.0
            
            else:
                
                # Binary classification
                p = self.label_distribution.item()  # Probability of positive class
                logits = torch.empty(num_samples, 1).uniform_(0, 1)
                logits = torch.where(logits >= p, 1.0, 0.0)
        else:
            # Generate random logits
            logits = torch.randn(num_samples, self.num_classes if self.num_classes != 2 else 1)

        return logits.to(x.device)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
        
        loss = self._loss(y_hat[mask], y[mask])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        mask = (y != -1)
        loss = self._loss(y_hat[mask], y[mask])
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def _accuracy(self, y_hat, y):
        metrics = []
        if self.num_classes == 2:
            return [("MCC", matthews_corrcoef(y_hat, y, task="binary", num_classes=self.num_classes)), ("ACC", binary_accuracy(y_hat, y))]
        if self.num_classes == 1:
            return [("MAE", nn.L1Loss(y_hat, y)), ("PCC", pearson_corrcoef(y_hat, y))]
        return [("MCC", matthews_corrcoef(y_hat, y, task="multiclass", num_classes=self.num_classes)), 
        ("ACC", multiclass_accuracy(y_hat, y, num_classes=self.num_classes))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
        )
    

class GlycoModel(pl.LightningModule):
    def __init__(self,
                 class_weights: torch.Tensor = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 **kwargs):
        super().__init__()
        self.num_classes = 3
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = None
        self.model = nn.Sequential(
            nn.Linear(1024, self.num_classes if self.num_classes > 2 else 1),
        )
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.output_path = kwargs.get("output_path", ".")

        self.hparams["Modeltype"] = "GlycoModel"
        self.save_hyperparameters()
        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        loss = self._loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat, y):
            self.log(f"train_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()
        mask = (y != -1)
    
        loss = self._loss(y_hat[mask], y[mask])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat[mask], y[mask]):
            self.log(f"val_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze()
        y_hat = self(x).squeeze()        
        loss = self._loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        for t in self._accuracy(y_hat, y):
            self.log(f"test_{t[0]}", t[1], on_epoch=True, on_step=False)
        return loss
    
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
        if self.num_classes == 2:
            return [("F1", binary_f1_score(y_hat, y))]
        if self.num_classes == 1:
            return [("MAE", nn.L1Loss(y_hat, y))]
        return [("F1", multiclass_f1_score(y_hat, y, num_classes=self.num_classes))]

    def _loss(self, y_hat, y):
        if self.loss_fn is not None:
            return self.loss_fn(y_hat, y, self.class_weights)
        if self.num_classes == 1:
            return F.mse_loss(y_hat, y)
        elif self.num_classes == 2:
            return F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.class_weights)
        return F.cross_entropy(y_hat, y, weight=self.class_weights)