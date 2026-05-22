import numpy as np
import os

from typing import Literal

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from abc import abstractmethod

import yaml
from dataclasses import dataclass
from pathlib import Path
from lightning import Trainer, seed_everything
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from gator.relpose.loss import L21Loss, RelativeCameraPoseRegression
from gator.relpose.models.relpose_wrapper import RelposeOptimizationParameters, RelposeWrapper
from gator.scripts.relpose.load_dataset import LoadDatasetArguments
from gator.scripts.serialize import to_serializable
import gator.utils.misc as misc
from gator import logger


@dataclass(kw_only=True)
class FinetuningArgumentsBase:
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """
    
    # dataset 
    dataset: str = '7scenes'
    """training set"""
    # training 
    seed: int = 0
    """Random seed"""

    inner_model_ckpt_path: Path
    dataset_loader: LoadDatasetArguments
    opt_params: RelposeOptimizationParameters

    precision: Literal['16', '32', 'bf16-mixed', '16-mixed'] = '16-mixed'
    """Use Automatic Mixed Precision for pretraining"""
    num_workers: int = 8
    """number of distributed processes"""
    world_size: int = 1
    local_rank: int = -1
    num_nodes: int = 1
    freeze_encdec: bool = True

    dist_url: str = 'env://'
    """url used to set up distributed training"""
    # paths 
    exp_name: str | None = None

    output_dir: Path = Path('/scratch/izar/skorokho/gator/relpose/')
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

    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = self.dataset_loader.get_train_dataset()
        eval_dataset = self.dataset_loader.get_test_dataset()
        
        logger.info(f"train dataset size: {len(train_dataset)}")
        logger.info(f"eval dataset size: {len(eval_dataset)}")

        data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.opt_params.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        self.opt_params.update_steps_per_epoch(len(train_dataset))

        data_loader_eval = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.opt_params.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        return data_loader_train, data_loader_eval
        
    @abstractmethod
    def _get_model(self) -> nn.Module:
        raise NotImplementedError("`_get_model` method should be implemented by subclasses")
    
    def get_wrapper(self) -> L.LightningModule:
        criterion = RelativeCameraPoseRegression(L21Loss())
        model = self._get_model()
        
        # wrapper
        model_wrapped = RelposeWrapper(
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
        )
        return model_wrapped
    
    def get_max_epochs(self) -> int:
        return self.opt_params.max_epoch

def finetune_model_relpose(args: FinetuningArgumentsBase):
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
        project="relpose",
        name=args.exp_name,
    )
    # wandb_logger = CSVLogger(
    #     save_dir=args.output_dir,
    #     name=args.exp_name,
    # )

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
