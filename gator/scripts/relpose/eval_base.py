import numpy as np
import os

from typing import Literal

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from lightning import Trainer, seed_everything

from gator.scripts.relpose.finetune_base import FinetuningArgumentsBase
import gator.utils.misc as misc
from gator import logger

def eval_model_relpose(args: FinetuningArgumentsBase):
    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    _, data_loader_eval = args.get_dataloaders()
    model_wrapped = args.get_wrapper()
    max_epochs = args.get_max_epochs()

    logger.info(f"Model = {str(model_wrapped)}")
    logger.info(f"Start training until {max_epochs} epochs")

    
    trainer = Trainer(
        precision=args.precision,
        max_epochs=max_epochs,
        accelerator="gpu",
    )

    trainer.validate(
        model=model_wrapped, 
        dataloaders=data_loader_eval,
        ckpt_path=args.ckpt_path, # TODO: Fix with proper inheritance someday...
    )
