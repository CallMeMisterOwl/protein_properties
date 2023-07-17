from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
from pathlib import Path

from torch import nn
from src.utils import seed_all, kaiming_init
from pytorch_lightning.loggers import WandbLogger

import wandb


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--output_path", type=Path)
        parser.add_argument("--HP", type=bool, default=False)
        parser.link_arguments("model.init_args.num_classes", "data.init_args.config.num_classes")


def main():
    seed_all(13)
    cli = MyLightningCLI(
        save_config_callback=None,
        run=False
    )
    cli.config.output_path.mkdir(parents=True, exist_ok=True)

    cli.trainer.logger = WandbLogger(project="protein_properties", log_model=True)
    cli.model.class_weights = cli.datamodule.class_weights
    kaiming_init(cli.model)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    # get validation scores for the best model
    cli.trainer.validate(ckpt_path="best", dataloaders=cli.datamodule.val_dataloader())
    if not cli.config.HP:
        cli.trainer.test(ckpt_path="best", dataloaders=cli.datamodule.test_dataloader())


if __name__ == "__main__":
    main()
