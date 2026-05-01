import os
import numpy as np
from typing import Literal
import torch
import torch.backends.cudnn as cudnn
import tyro
from dataclasses import dataclass

from gator.models.gator_2view.model_gator import GatorConfig, Gator
from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.scripts.pretrainings.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator.scripts.pretrainings.pretrain_base import pretrain_model

@dataclass(kw_only=True)
class TrainingArgumentsGator(TrainingArgumentsJigsaw):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: GatorConfig
    
    def _get_model(self) -> torch.nn.Module:
        # model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Gator(self.model_config)
        model.to(device)
        return model
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = GatorWrapper(
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped

if __name__ == '__main__':
    args = tyro.cli(TrainingArgumentsGator)
    pretrain_model(args)
