import numpy as np
import os

from typing import Literal

import torch
import torch.backends.cudnn as cudnn

import tyro
from dataclasses import dataclass
from pathlib import Path

from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from gator.models.model_gator import GatorConfig, Gator
from gator.models.gator_losses import GatorLossConfig
import gator.utils.misc as misc
from gator import logger
from gator.models.gator_wrapper import GatorWrapper, OptimizationParameters
import webdataset as wds
from gator.datasets.shard.transforms import get_pair_transforms

@dataclass(kw_only=True)
class TrainingArguments:
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    gator_loss_config: GatorLossConfig  
    gator_config: GatorConfig
    
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

    opt_params: OptimizationParameters

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

        
def main(args: TrainingArguments):
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

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    transform = get_pair_transforms(args.transforms, totensor=True, normalize=True)

    ## training dataset and loader 
    logger.info('Building dataset for {:s} with transforms {:s}'.format(args.dataset, args.transforms))

    dataset = wds.WebDataset(
        urls=str(args.data_dir / args.dataset / "train-{000000..000180}.tar"),
        shardshuffle=True,
    )\
        .shuffle(512)\
        .decode("pil")\
        .rename(im1="im1.jpg", im2="im2.jpg")\
        .to_tuple("im1", "im2")\
        .map(lambda x: transform(x[0], x[1]))\
        .batched(args.batch_size, partial=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=None,
    )

    # learning rates
    args.opt_params.update_lr(args.batch_size)
   
    ## model 
    loss_cls = args.gator_loss_config.get_loss()
    criterion = loss_cls(
        grid_size=(
            args.gator_config.image_size // args.gator_config.patch_size, 
            args.gator_config.image_size // args.gator_config.patch_size
        ),
        patch_size=args.gator_config.patch_size,
    )

    model = Gator(args.gator_config, args.gator_loss_config)
    model.to(device)
    
    model_wrapped = GatorWrapper(
        model=model, 
        loss_fn=criterion,
        optimization_config=args.opt_params,
    )

    logger.info(f"Model = {str(model_wrapped)}")
    logger.info(f"Start training until {args.opt_params.max_epoch} epochs")
    
    wandb_logger = WandbLogger(
        save_dir=args.output_dir,
        project="gator",
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

    # trainer = Trainer(
    #     logger=wandb_logger,
    #     precision=args.precision,
    #     max_epochs=args.max_epoch,
    #     accelerator="gpu",
    #     strategy="ddp",
    #     devices=args.world_size,
    #     num_nodes=args.num_nodes,
    #     callbacks=[model_checkpoint_callback],
    # )
    
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
        ckpt_path=latest_ckpt_path,
    )

if __name__ == '__main__':
    args = tyro.cli(TrainingArguments)
    main(args)
