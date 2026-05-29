import os
import numpy as np
from typing import Literal
import torch
import torch.backends.cudnn as cudnn
import tyro
from dataclasses import dataclass, field

from gator.models.gator_2view.model_gator import GatorConfig, Gator
from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.scripts.pretrainings_ddp.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator.scripts.pretrainings_ddp.pretrain_base import pretrain_model

@dataclass(kw_only=True)
class TrainingArgumentsGator(TrainingArgumentsJigsaw):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: GatorConfig
    opt_params: OptimizationParameters = field(
        default_factory=lambda: OptimizationParameters(
            blr=1e-4,
            tt_split_ratio=0.045, # 181 * 0.045 ~= 8 => if training on 4 gpus then each one gets 2 shards
        )
    )
    
    def _get_model(self) -> torch.nn.Module:
        # model
        model = Gator(self.model_config)
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
