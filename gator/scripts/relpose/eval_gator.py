from pathlib import Path

import tyro
from dataclasses import dataclass

from gator.relpose.loss import L21Loss, RelativeCameraPoseRegression
import lightning as L
from gator.relpose.models.relpose_wrapper import RelposeWrapper
from gator.scripts.relpose.eval_base import eval_model_relpose
from gator.scripts.relpose.finetune_gator import FinetuningArgumentsGator

@dataclass(kw_only=True)
class EvalArgumentsGator(FinetuningArgumentsGator):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    ckpt_path: Path

    def get_wrapper(self) -> L.LightningModule:
        criterion = RelativeCameraPoseRegression(L21Loss())
        model = self._get_model()
        
        # wrapper
        model_wrapped = RelposeWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model,
            loss_fn=criterion,
            optimization_config=self.opt_params,
        )
        return model_wrapped

if __name__ == '__main__':
    args = tyro.cli(EvalArgumentsGator)
    eval_model_relpose(args)
