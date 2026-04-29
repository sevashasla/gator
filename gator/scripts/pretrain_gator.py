import numpy as np
import os

from typing import Literal

import torch
import torch.backends.cudnn as cudnn

import yaml
import tyro
from dataclasses import dataclass
from pathlib import Path
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from gator.scripts.serialize import to_serializable
from gator.models.gator_visualizer import GatorVisConfig
from gator.models.model_gator import GatorConfig, Gator
from gator.models.gator_losses import GatorLossConfig
import gator.utils.misc as misc
from gator import logger
from gator.models.gator_wrapper import GatorWrapper, OptimizationParameters
import webdataset as wds
from gator.datasets.shard.transforms import get_pair_transforms_gator

@dataclass(kw_only=True)
class TrainingArguments:
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    loss_config: GatorLossConfig  
    gator_config: GatorConfig
    visualizer_config: GatorVisConfig
    
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

    # normalization is done in the gator model itself
    transform = get_pair_transforms_gator(args.transforms, normalize=False)

    ## training dataset and loader 
    logger.info('Building dataset for {:s} with transforms {:s}'.format(args.dataset, args.transforms))

    all_shards = list((args.data_dir / args.dataset).glob("train-*.tar"))
    all_shards = sorted(all_shards)

    rng = np.random.default_rng(0)
    rng.shuffle(all_shards)

    n_eval = int(args.opt_params.tt_split_ratio * len(all_shards))
    eval_shards = all_shards[:n_eval]
    train_shards = all_shards[n_eval:]

    logger.info(f"Training ({len(train_shards)} shards): {[s.name for s in train_shards]}")
    logger.info(f"Evaluation ({len(eval_shards)} shards): {[s.name for s in eval_shards]}")

    train_dataset = wds.WebDataset(
        urls=[str(el) for el in train_shards],
        shardshuffle=True,
    )\
        .shuffle(512)\
        .decode("torchrgb8")\
        .rename(im1="im1.jpg", im2="im2.jpg")\
        .to_tuple("im1", "im2")\
        .map(lambda x: transform(x[0], x[1]))\
        .batched(args.opt_params.batch_size, partial=False)
    
    eval_dataset = wds.WebDataset(
        urls=[str(el) for el in eval_shards],
        shardshuffle=False,
    )\
        .decode("torchrgb8")\
        .rename(im1="im1.jpg", im2="im2.jpg")\
        .to_tuple("im1", "im2")\
        .map(lambda x: transform(x[0], x[1]))\
        .batched(args.opt_params.batch_size, partial=False)


    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=None,
    )

    data_loader_eval = torch.utils.data.DataLoader(
        eval_dataset,
        num_workers=min(args.num_workers, len(eval_shards)),
        batch_size=None,
    )

    # learning rates
    args.opt_params.update_lr()
   
    ## model 
    loss_cls = args.loss_config.get_loss()
    criterion = loss_cls(
        grid_size=(
            args.gator_config.image_size // args.gator_config.patch_size, 
            args.gator_config.image_size // args.gator_config.patch_size
        ),
        patch_size=args.gator_config.patch_size,
    )

    visualizer_cls = args.visualizer_config.get_visualizer()
    visualizer = visualizer_cls(
        grid_size=(
            args.gator_config.image_size // args.gator_config.patch_size, 
            args.gator_config.image_size // args.gator_config.patch_size
        ),
        patch_size=args.gator_config.patch_size,
    )

    model = Gator(args.gator_config)
    model.to(device)
    
    model_wrapped = GatorWrapper(
        model=model, 
        loss_fn=criterion,
        optimization_config=args.opt_params,
        visualizer=visualizer,
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
        callbacks=[model_checkpoint_callback, learning_rate_logger],
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
