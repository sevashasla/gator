import os
import numpy as np
from typing import ClassVar, Literal
import torch
import torch.backends.cudnn as cudnn
import tyro
from dataclasses import dataclass

from gator.models.criterion import MaskedMSE
from gator.models.gator_visualizer.mae import MAEVis
from gator.models.mae_1view.wrapper_mae import MAE1ViewWrapper
from gator.models.mae_1view.model_mae import MAEModel, MAEConfig
from gator.scripts.pretrainings.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator.scripts.pretrainings.pretrain_base import pretrain_model

@dataclass(kw_only=True)
class TrainingArgumentsMAE(TrainingArgumentsJigsaw):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: MAEConfig

    norm_pix_loss: Literal[0, 1] = 1
    """apply per-patch mean/std normalization before applying the loss"""

    loss_config: ClassVar[None] = None
    visualizer_config: ClassVar[None] = None
    
    def _get_model(self) -> torch.nn.Module:
        # model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MAEModel(self.model_config)
        model.to(device)
        return model
    
    def _get_visualizer(self):
        visualizer = MAEVis(
            grid_size=(
                self.model_config.image_size // self.model_config.patch_size, 
                self.model_config.image_size // self.model_config.patch_size
            ),
            patch_size=self.model_config.patch_size,
        )
        return visualizer
    
    def _get_loss(self):
        loss = MaskedMSE(norm_pix_loss=bool(self.norm_pix_loss))
        return loss
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = MAE1ViewWrapper(
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped

if __name__ == '__main__':
    args = tyro.cli(TrainingArgumentsMAE)
    pretrain_model(args)
