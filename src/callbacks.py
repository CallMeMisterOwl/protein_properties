import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
from lightning.pytorch.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import wandb


def create_conf_matrix(ys, preds, num_classes, model_name):
    if num_classes == 2:
        ranges = "RSA ranges for classes: \n\n0 : 0% < 16%    1 = 16% - 100%"
    elif num_classes == 3:
        ranges = "RSA ranges for classes: \n\n0 : 0% < 16%    1 = 16% < 36%    2 = 36% - 100%"
    else:
        # forumla for calculating ranges for n classes -> sqrt of 100 * RSA
        """
                Ten-state model RSA ranges

                0: [0, 0.01)          1: [0.01, 0.04)
                2: [0.04, 0.09)     3: [0.09, 0.16)
                4: [0.16, 0.25)     5: [0.25, 0.36) 
                6: [0.36, 0.49)     7: [0.49, 0.64)
                8: [0.64, 0.81)     9: [0.81, 1]
        """
        ranges = ""

    fig, axes = plt.subplots()
    """Plot confusion matrix for predictions vs. ground truth"""
    conf_matrix = confusion_matrix(ys, preds)
    pal = sns.color_palette("blend:#053b38,#3df2d4", as_cmap=True)
    sns.heatmap(conf_matrix, annot=True, cmap=pal, fmt="d", ax=axes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix test set {model_name} classes {num_classes}")
    
    return fig, axes
    

class LogPredictionCallback(Callback):
    
    def __init__(self, out_path):
        super().__init__()
        self.out_path = Path(out_path)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.test_cache = []

        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_cache.append(outputs)
        

    def on_test_epoch_end(self, trainer, pl_module):
        # Unpack outputs
        if pl_module.hparams["Modeltype"] == "SASADummyModel" or pl_module.hparams["Modeltype"] == "GlycoDummy":
            return
        outputs = list(map(list, zip(*self.test_cache)))
        
        
        preds = np.concatenate(outputs[0])
        ys = np.concatenate(outputs[1])
        if pl_module.num_classes == 1:
            self.test_preds = pd.DataFrame(zip(preds, ys), columns=["Pred_value", "Real_value"])
            self.test_preds.to_csv(self.out_path / f"{pl_module.hparams['Modeltype']}_{pl_module.num_classes}_test_preds.tsv", sep='\t', index=False)
            return
        if pl_module.num_classes < 3:
            # For binary predictions use threshold of 0.5
            pred_classes = (preds >= 0.5).astype(int)
        else:
            # For multiclass predictions take index of max
            
            pred_classes = np.argmax(preds, axis=1)
        
        # Save test predictions to csv
        self.test_preds = pd.DataFrame(zip(preds, pred_classes, ys), columns=["Score", "Pred_class", "Real_class"])
        self.test_preds.to_csv(self.out_path / f"{pl_module.hparams['Modeltype']}_{pl_module.num_classes}_test_preds.tsv", sep='\t', index=False)
        fig, ax = create_conf_matrix(ys, pred_classes, pl_module.num_classes, pl_module.hparams["Modeltype"])
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log({f"confmatrix_{pl_module.hparams['Modeltype']}_{pl_module.num_classes}": wandb.Image(ax)})