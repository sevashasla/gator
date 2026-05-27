from pathlib import Path
from lightning import Trainer, seed_everything
import torch
import torch.backends.cudnn as cudnn
import tyro
from dataclasses import dataclass

from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.scripts.pretrainings.pretrain_gator import TrainingArgumentsGator
import yaml

@dataclass(kw_only=True)
class TestArgumentsGator(TrainingArgumentsGator):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    ckpt_path: Path
    output_path: Path | None = None

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.ckpt_path.parent.parent / "test_metrics.yaml"
        return super().__post_init__()
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = GatorWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model,
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped

def test_gator(args: TestArgumentsGator):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed_everything(args.seed)
    cudnn.benchmark = True

    _, data_loader_eval = args.get_dataloaders()
    model_wrapped = args.get_wrapper()
    max_epochs = args.get_max_epochs()
    
    trainer = Trainer(
        precision=args.precision,
        max_epochs=max_epochs,
        accelerator="gpu",
    )

    test_metrics = trainer.test(
        model=model_wrapped, 
        dataloaders=data_loader_eval,
        ckpt_path=args.ckpt_path,
    )

    out_path = args.output_path
    with out_path.open("w") as f:
        yaml.dump(test_metrics, f)


if __name__ == '__main__':
    args = tyro.cli(TestArgumentsGator)
    test_gator(args)
