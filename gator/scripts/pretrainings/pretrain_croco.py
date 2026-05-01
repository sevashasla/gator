import numpy as np
import os

from typing import Literal

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import tyro
from dataclasses import dataclass, field

from gator.models.criterion import MaskedMSE
from gator.models.croco import CroCoNet
import gator.utils.misc as misc
from gator import logger
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.models.croco_wrapper import CroCoWrapper
from gator.scripts.pretrainings.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator.scripts.pretrainings.pretrain_base import pretrain_model
from typing import ClassVar

@dataclass(kw_only=True)
class TrainingArgumentsCroco(TrainingArgumentsJigsaw):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """
    norm_pix_loss: Literal[0, 1] = 1
    """apply per-patch mean/std normalization before applying the loss"""

    transforms: str = "crop224+acolor+norm"
    """
    transforms to apply. in the paper, we also use some homography and
    rotation, but find later that they were not useful or even harmful
    """ 

    # remove them from the arguments
    loss_config: ClassVar[None] = None
    model_config: ClassVar[None] = None
    visualizer_config: ClassVar[None] = None
    opt_params = field(
        default_factory=lambda: OptimizationParameters(
            weight_decay=0.05,
            blr=1.5e-4 / 2, # because we use 128 but not 256 batch size
            warmup_epochs=40,
            batch_size=64,
            epochs=800,
            max_epoch=400,
        )
    )

    model_configuration: str = "CroCoNet()"
    
    def _get_loss(self) -> MaskedMSE:
        # loss
        logger.info(f'Loading criterion: MaskedMSE(norm_pix_loss={str(bool(self.norm_pix_loss))})')
        criterion = MaskedMSE(norm_pix_loss=bool(self.norm_pix_loss))
        return criterion
    
    def _get_model(self) -> CroCoNet:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model: CroCoNet = eval(self.model_configuration)
        model.to(device)
        return model
    
    def get_wrapper(self):
        criterion = self._get_loss()
        # It is built-it in the Wrapper
        # visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = CroCoWrapper(
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
        )
        return model_wrapped
    

if __name__ == '__main__':
    args = tyro.cli(TrainingArgumentsCroco)
    pretrain_model(args)
