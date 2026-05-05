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
from gator.models.gator_visualizer import GatorVisConfig
from gator.models.gator_losses import GatorLossConfig
from gator.models.jigsaw_1view.jigsaw_wrapper import Jigsaw1ViewWrapper, OptimizationParameters
from gator.models.jigsaw_1view.model_jigsaw import Jigsaw1View, Jigsaw1ViewConfig
import gator.utils.misc as misc
from gator import logger
from gator.datasets.shard.transforms import get_pair_transforms_gator
from gator.scripts.pretrainings.pretrain_base import TrainingArgumentsBase, pretrain_model


@dataclass(kw_only=True)
class TrainingArgumentsJigsaw(TrainingArgumentsBase):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    loss_config: GatorLossConfig  
    model_config: Jigsaw1ViewConfig
    visualizer_config: GatorVisConfig
    opt_params: OptimizationParameters

    def _get_transforms(self) -> T.Transform:
        return get_pair_transforms_gator(self.transforms)
    
    def get_dataloaders(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        ## training dataset and loader 
        # normalization is done in the gator model itself
        transform = self._get_transforms()

        logger.info('Building dataset for {:s} with transforms {:s}'.format(self.dataset, self.transforms))

        all_shards = list((self.data_dir / self.dataset).glob("train-*.tar"))
        all_shards = sorted(all_shards)

        rng = np.random.default_rng(0)
        rng.shuffle(all_shards)

        n_eval = int(self.opt_params.tt_split_ratio * len(all_shards))
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
            .batched(self.opt_params.batch_size, partial=False)
        
        eval_dataset = wds.WebDataset(
            urls=[str(el) for el in eval_shards],
            shardshuffle=False,
        )\
            .decode("torchrgb8")\
            .rename(im1="im1.jpg", im2="im2.jpg")\
            .to_tuple("im1", "im2")\
            .map(lambda x: transform(x[0], x[1]))\
            .batched(self.opt_params.batch_size, partial=False)


        data_loader_train = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.num_workers,
            batch_size=None,
        )

        data_loader_eval = torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=min(self.num_workers, len(eval_shards)),
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Jigsaw1View(self.model_config)
        model.to(device)
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
