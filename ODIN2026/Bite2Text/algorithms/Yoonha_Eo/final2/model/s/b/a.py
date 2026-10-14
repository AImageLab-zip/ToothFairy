import argparse
from pathlib import Path

import os, sys
sys.path.append(os.getcwd())

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml

from teethland.datamodules import (
    TeethInstFullDataModule,
    TeethMixedFullDataModule,
)
from teethland.models import FullNet


def predict(stage: str, mixed: bool, devices: int, config: str):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config['seed'], workers=True)

    config['datamodule']['batch_size'] = 1
    if mixed:
        dm = TeethMixedFullDataModule(
            seed=config['seed'], **config['datamodule'],
        )
        config['model']['instseg'] = config['model']['mixedseg']
    else:
        dm = TeethInstFullDataModule(
            seed=config['seed'], **config['datamodule'],
        )
    
    model = FullNet(
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        **config['model'],
        out_dir=Path('dentalnetPr' if stage == 'instances' else config['out_dir']),
    )

    logger = TensorBoardLogger(
        save_dir=config['work_dir'],
        name='',
        version=f'landmarks_{config["version"]}',
        default_hp_metric=False,
    )
    logger.log_hyperparams(config)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=devices,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.predict(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['instances', 'landmarks'])
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--devices', required=False, default=1, type=int)
    parser.add_argument('--config', required=False, default='teethland/config/config.yaml', type=str)
    args = parser.parse_args()

    predict(args.stage, args.mixed, args.devices, args.config)
