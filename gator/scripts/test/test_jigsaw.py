from pathlib import Path
from dataclasses import dataclass

import tyro

from gator.models.jigsaw_1view.jigsaw_wrapper import Jigsaw1ViewWrapper
from gator.scripts.pretrainings.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator.scripts.test.test_gator import test_gator

@dataclass(kw_only=True)
class TestArgumentsJigsaw(TrainingArgumentsJigsaw):
    """
    On the cluster each node has two V100 32GB GPUs.
    With num_workers=4 it already achieves around 90% GPU utilization.
    """

    ckpt_path: Path
    output_path: Path | None

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.ckpt_path.parent.parent / "test_metrics.yaml"
        return super().__post_init__()
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = Jigsaw1ViewWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model,
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped

if __name__ == '__main__':
    args = tyro.cli(TestArgumentsJigsaw)
    test_gator(args)
