import torch
import tyro
from dataclasses import dataclass

from gator.relpose.models.croco_relpose import CroCoNetRelpose
from gator import logger
from gator.scripts.relpose.finetune_base import FinetuningArgumentsBase, finetune_model_relpose

@dataclass(kw_only=True)
class FinetuningArgumentsCroco(FinetuningArgumentsBase):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    model_configuration: str = "CroCoNet()"
    
    def _get_model(self) -> CroCoNetRelpose:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TODO: This is a bit hacky, we should ideally refactor 
        # the code to avoid eval and string manipulation...
        model_configuration2 = "CroCoNetRelpose(" + \
        f"croco_ckpt_path='{self.inner_model_ckpt_path}', freeze={self.freeze_encdec}, " + \
        f"{self.model_configuration[9:-1]}" + \
        ")"
        logger.info(f"Instantiating model with configuration: {model_configuration2}")
        model: CroCoNetRelpose = eval(model_configuration2)

        model.to(device)
        return model

if __name__ == '__main__':
    args = tyro.cli(FinetuningArgumentsCroco)
    finetune_model_relpose(args)
