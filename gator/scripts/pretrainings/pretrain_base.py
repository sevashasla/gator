import numpy as np
import os

from typing import Literal

import torch
import torch.backends.cudnn as cudnn
from abc import abstractmethod

import yaml
import tyro
from dataclasses import dataclass
from pathlib import Path
from lightning import Trainer, seed_everything
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from gator.scripts.serialize import to_serializable
import gator.utils.misc as misc
from gator import logger


@dataclass(kw_only=True)
class TrainingArgumentsBase:
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """
    
    # dataset 
    dataset: str = 'habitat_release_shards'
    """training set"""
    transforms: str = 'crop224+acolor'
    """
    transforms to apply. in the paper, we also use some homography and
    rotation, but find later that they were not useful or even harmful
    """ 
    # training 
    seed: int = 0
    """Random seed"""

    precision: Literal['16', '32', 'bf16-mixed', '16-mixed'] = '16-mixed'
    """Use Automatic Mixed Precision for pretraining"""
    num_workers: int = 8
    """number of distributed processes"""
    world_size: int = 1
    local_rank: int = -1
    num_nodes: int = 1

    dist_url: str = 'env://'
    """url used to set up distributed training"""
    # paths 
    exp_name: str | None = None

    output_dir: Path = Path('/scratch/izar/skorokho/gator/')
    """path where to save the output"""
    data_dir: Path = Path('/scratch/izar/skorokho/croco-dataset/')
    """path where data are stored"""

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = "exp"
            idx = 0
            while (self.output_dir / f"{self.exp_name}-{idx:03}").exists():
                idx += 1
            self.exp_name = f"{self.exp_name}-{idx:03}"
        self.output_dir = self.output_dir / self.exp_name

    @abstractmethod
    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        raise NotImplementedError("`get_dataloaders` method should be implemented by subclasses")
        
    @abstractmethod
    def get_wrapper(self) -> L.LightningModule:
        raise NotImplementedError("`get_wrapper` method should be implemented by subclasses")
    
    @abstractmethod
    def get_max_epochs(self) -> int:
        raise NotImplementedError("`get_max_epochs` method should be implemented by subclasses")

        
def pretrain_model(args: TrainingArgumentsBase):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()
    
    logger.info("output_dir: " + str(args.output_dir))
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # auto resume 
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    latest_ckpt_path = ckpt_dir / "last.ckpt"
    latest_ckpt_path = latest_ckpt_path if latest_ckpt_path.exists() else None

    save_config_path = args.output_dir / "training_config.yml"
    with open(save_config_path, "w") as f:
        yaml.safe_dump(to_serializable(args), f, sort_keys=False)

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_eval = args.get_dataloaders()
    model_wrapped = args.get_wrapper()
    max_epochs = args.get_max_epochs()

    logger.info(f"Model = {str(model_wrapped)}")
    logger.info(f"Start training until {max_epochs} epochs")
    
    wandb_logger = WandbLogger(
        save_dir=args.output_dir,
        project="gator",
        name=args.exp_name,
    )

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        monitor="epoch",
        mode="max",
        save_top_k=3,
        every_n_epochs=1,
        enable_version_counter=False,
        save_on_exception=True,
    )

    learning_rate_logger = LearningRateMonitor(
        logging_interval='step',
        log_momentum=False,
    )

    # trainer = Trainer(
    #     logger=wandb_logger,
    #     precision=args.precision,
    #     max_epochs=max_epochs,
    #     accelerator="gpu",
    #     strategy="ddp",
    #     devices=args.world_size,
    #     num_nodes=args.num_nodes,
    #     callbacks=[model_checkpoint_callback],
    # )
    
    trainer = Trainer(
        logger=wandb_logger,
        precision=args.precision,
        max_epochs=max_epochs,
        accelerator="gpu",
        callbacks=[model_checkpoint_callback, learning_rate_logger],
    )

    trainer.fit(
        model=model_wrapped, 
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_eval,
        ckpt_path=latest_ckpt_path,
    )
