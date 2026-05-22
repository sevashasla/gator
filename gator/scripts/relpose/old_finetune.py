import numpy as np
import os

from typing import Literal

import torch
import torch.backends.cudnn as cudnn

import tyro
from dataclasses import dataclass
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gator.models.gator_2view.model_gator import GatorConfig
from gator.relpose.loss import L21Loss, RelativeCameraPoseRegression
from gator.relpose.models.gator_relpose import GatorRelpose
from gator.scripts.relpose.load_dataset import LoadDatasetArguments
import gator.utils.misc as misc
from gator import logger
from gator.relpose.models.relpose_wrapper import RelposeWrapper, RelposeOptimizationParameters

@dataclass(kw_only=True)
class TrainingArguments:
    """
    On the cluster each node has two V100 32GB GPUs.
    """
    
    # dataset 
    dataset: str = '7scenes'
    """training set"""
    # training 
    seed: int = 0
    """Random seed"""

    gator_ckpt_path: Path
    dataset_loader: LoadDatasetArguments
    model_config: GatorConfig

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

    output_dir: Path = Path('/scratch/izar/skorokho/gator/relpose')
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
        
def main(args: TrainingArguments):
    logger.info("output_dir: " + str(args.output_dir))
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # auto resume 
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    latest_ckpt_path = ckpt_dir / "last.ckpt"
    latest_ckpt_path = latest_ckpt_path if latest_ckpt_path.exists() else None

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    ## training dataset and loader 
    
    train_dataset = args.dataset_loader.get_train_dataset()
    eval_dataset = args.dataset_loader.get_test_dataset()

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.opt_params.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )
    args.opt_params.update_steps_per_epoch(len(train_dataset))

    data_loader_eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.opt_params.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
   
    ## model
    model = GatorRelpose(
        args.model_config,
        gator_ckpt_path=args.gator_ckpt_path,
        freeze=args.freeze_encdec,
    )
    model.to(device)
    logger.info(f"Successfully loaded model from {args.gator_ckpt_path}!")

    criterion = RelativeCameraPoseRegression(L21Loss())
    
    model_wrapped = RelposeWrapper(
        model=model, 
        loss_fn=criterion,
        optimization_config=args.opt_params,
    )

    logger.info(f"Model = {str(model_wrapped)}")
    logger.info(f"Start training until {args.opt_params.max_epoch} epochs")
    
    # wandb_logger = WandbLogger(
    #     save_dir=args.output_dir,
    #     project="relpose",
    #     name=args.exp_name,
    # )
    wandb_logger = CSVLogger(
        save_dir=args.output_dir,
        name=args.exp_name,
    )

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        monitor="epoch",
        save_top_k=3,
        every_n_epochs=1,
        enable_version_counter=False,
        save_on_exception=True,
    )
    
    trainer = Trainer(
        logger=wandb_logger,
        precision=args.precision,
        max_epochs=args.opt_params.max_epoch,
        accelerator="gpu",
        callbacks=[model_checkpoint_callback],
    )

    trainer.fit(
        model=model_wrapped, 
        train_dataloaders=data_loader_train,
        val_dataloaders=data_loader_eval,
        ckpt_path=latest_ckpt_path,
    )

if __name__ == '__main__':
    args = tyro.cli(TrainingArguments)
    main(args)
