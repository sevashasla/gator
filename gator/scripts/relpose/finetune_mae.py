import torch
import tyro
from dataclasses import dataclass
from pathlib import Path

from gator.models.mae_1view.model_mae import MAEConfig
from gator.relpose.models.mae_relpose import MAERelpose
from gator.scripts.relpose.finetune_base import FinetuningArgumentsBase, finetune_model_relpose

@dataclass(kw_only=True)
class FinetuningArgumentsMAE(FinetuningArgumentsBase):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_config: MAEConfig
    output_dir: Path = Path('/scratch/izar/bosi/gator/relpose/')

    def _get_model(self) -> torch.nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MAERelpose(
            config=self.model_config,
            mae_ckpt_path=self.inner_model_ckpt_path,
            freeze=self.freeze_encdec,
        )
        model.to(device)
        return model

if __name__ == '__main__':
    args = tyro.cli(FinetuningArgumentsMAE)
    finetune_model_relpose(args)
