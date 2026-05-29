import torch
import tyro
from dataclasses import dataclass

from gator.models.gator_2view.model_gator import GatorConfig
from gator.relpose.models.gator_relpose import GatorRelpose
from gator.scripts.relpose.finetune_base import FinetuningArgumentsBase, finetune_model_relpose

@dataclass(kw_only=True)
class FinetuningArgumentsGator(FinetuningArgumentsBase):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: GatorConfig

    def _get_model(self) -> torch.nn.Module:
        # model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GatorRelpose(
            config=self.model_config,
            gator_ckpt_path=self.inner_model_ckpt_path,
            freeze=self.freeze_encdec,
        )
        model.to(device)
        return model

if __name__ == '__main__':
    args = tyro.cli(FinetuningArgumentsGator)
    finetune_model_relpose(args)
