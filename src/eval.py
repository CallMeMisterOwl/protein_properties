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
        parser.add_argument("--ckpt_path", type=Path)


def main():
    seed_all(13)
    cli = MyLightningCLI(
        save_config_callback=None,
        run=False
    )
    cli.config.output_path.mkdir(parents=True, exist_ok=True)
    model = cli.model.load_from_checkpoint(cli.config.ckpt_path)
    cli.trainer.test(model, dataloaders=cli.datamodule.test_dataloader())


if __name__ == "__main__":
    main()
