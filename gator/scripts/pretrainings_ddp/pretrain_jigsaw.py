import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from typing import Literal
import tyro
from dataclasses import dataclass
import webdataset as wds
import torchvision.transforms.v2 as T

from gator.models.gator_losses.base import GatorBaseLoss
from gator.models.gator_visualizer.base import GatorBaseVis
from gator.models.jigsaw_1view.jigsaw_wrapper import Jigsaw1ViewWrapper
from gator.models.jigsaw_1view.model_jigsaw import Jigsaw1View, Jigsaw1ViewConfig
from gator import logger
from gator.datasets.shard.transforms import get_pair_transforms_gator
from gator.scripts.pretrainings_ddp.pretrain_base import TrainingArgumentsBase, pretrain_model
import gator.utils.misc as misc

@dataclass(kw_only=True)
class TrainingArgumentsJigsaw(TrainingArgumentsBase):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: Jigsaw1ViewConfig

    def _get_transforms(self) -> T.Transform:
        return get_pair_transforms_gator(self.transforms)
    
    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        ## training dataset and loader 
        # normalization is done in the gator model itself
        transform = self._get_transforms()

        logger.info('Building dataset for {:s} with transforms {:s}'.format(self.dataset, self.transforms))
        
        world_size_env = int(os.environ.get("WORLD_SIZE", 1))
        global_rank_env = int(os.environ.get("RANK", 0))
        logger.info(f"World_size (from env) {world_size_env}")
        logger.info(f"Global_rank (from env) {global_rank_env}")

        all_shards = list((self.data_dir / self.dataset).glob("train-*.tar"))
        all_shards = sorted(all_shards)

        n_eval = int(self.opt_params.tt_split_ratio * len(all_shards))
        shuffle_seed = 0
        eval_ids = None
        while eval_ids is None or len(all_shards) - 1 in eval_ids:
            rng = np.random.default_rng(shuffle_seed)
            random_ids = rng.permutation(len(all_shards))
            eval_ids = random_ids[:n_eval]
            train_ids = random_ids[n_eval:]

        # get train/eval split
        train_shards = [all_shards[i] for i in train_ids]
        eval_shards = [all_shards[i] for i in eval_ids]

        logger.info(f"Training ({len(train_shards)} shards): {[s.name for s in train_shards]}")
        logger.info(f"Evaluation ({len(eval_shards)} shards): {[s.name for s in eval_shards]}")

        # prepare datasets
        train_batches_per_rank = int(
            self.opt_params.dataset_size * (1 - self.opt_params.tt_split_ratio)
        ) // (self.opt_params.batch_size * world_size_env)
        train_dataset = wds.WebDataset(
            urls=[str(el) for el in train_shards],
            shardshuffle=True,
            resampled=True,
            nodesplitter=wds.shardlists.split_by_node,
        )\
            .shuffle(512)\
            .decode("torchrgb8")\
            .rename(im1="im1.jpg", im2="im2.jpg")\
            .to_tuple("im1", "im2")\
            .map(lambda x: transform(x[0], x[1]))\
            .batched(self.opt_params.batch_size, partial=False)\
            .with_epoch(train_batches_per_rank)
        
        eval_batches_per_rank = int(
            self.opt_params.dataset_size * self.opt_params.tt_split_ratio
        ) // (self.opt_params.batch_size * world_size_env)
        eval_dataset = wds.WebDataset(
            urls=[str(el) for el in eval_shards],
            shardshuffle=False,
            nodesplitter=wds.shardlists.split_by_node,
        )\
            .decode("torchrgb8")\
            .rename(im1="im1.jpg", im2="im2.jpg")\
            .to_tuple("im1", "im2")\
            .map(lambda x: transform(x[0], x[1]))\
            .batched(self.opt_params.batch_size, partial=False)\
            .with_epoch(eval_batches_per_rank)

        # prepare dataloaders
        data_loader_train = wds.WebLoader(
            train_dataset,
            num_workers=self.num_workers,
            batch_size=None,
        )

        eval_num_workers = min(self.num_workers, max(1, n_eval // world_size_env))
        logger.info(f"Want to use {eval_num_workers} workers for evaluation, but instead use only 1 :(")
        eval_num_workers = 1
        data_loader_eval = wds.WebLoader(
            eval_dataset,
            num_workers=eval_num_workers,
            batch_size=None,
        )

        return data_loader_train, data_loader_eval
    
    def _get_loss(self) -> GatorBaseLoss:
        # loss
        loss_cls = self.loss_config.get_loss()
        criterion = loss_cls(
            grid_size=(
                self.model_config.image_size // self.model_config.patch_size, 
                self.model_config.image_size // self.model_config.patch_size
            ),
            patch_size=self.model_config.patch_size,
        )
        return criterion
    
    def _get_visualizer(self) -> GatorBaseVis:
        # visualizer
        visualizer_cls = self.visualizer_config.get_visualizer()
        visualizer = visualizer_cls(
            grid_size=(
                self.model_config.image_size // self.model_config.patch_size, 
                self.model_config.image_size // self.model_config.patch_size
            ),
            patch_size=self.model_config.patch_size,
        )
        return visualizer
    
    def _get_model(self) -> torch.nn.Module:
        # model
        model = Jigsaw1View(self.model_config)
        return model
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = Jigsaw1ViewWrapper(
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped
    
    def get_max_epochs(self) -> int:
        return self.opt_params.max_epoch

if __name__ == '__main__':
    args = tyro.cli(TrainingArgumentsJigsaw)
    pretrain_model(args)
