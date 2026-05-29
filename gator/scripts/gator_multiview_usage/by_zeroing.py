from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, ClassVar
from lightning import seed_everything
from matplotlib.pylab import indices
import torch
from torchmetrics import Accuracy, MeanSquaredError, MeanAbsoluteError
import torchvision
import torchvision.transforms.v2.functional as TF
import torch.nn.functional as F
from tqdm import tqdm
import tyro

from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.models.gator_visualizer.base import GatorBaseVis
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.scripts.pretrainings.pretrain_gator import TrainingArgumentsGator
import gator.utils.misc as misc
import torch.backends.cudnn as cudnn
from gator import logger
import pandas as pd


@dataclass(kw_only=True)
class ArgumentsGator(TrainingArgumentsGator):
    ckpt_path: Path
    output_dir: ClassVar[Path] = None
    output_path: Path = Path("output.csv")
    
    shuffle_ratios: list[float] = field(default_factory=lambda: [1.0])
    """
    List of shuffle ratios to evaluate
    """
    
    num_repeat: int = 3
    """
    How many times to repeat each validation
    """

    def __post_init__(self):
        pass
    
    def get_wrapper(self):
        criterion = self._get_loss()
        # It is built-it in the Wrapper
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = GatorWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model,
            loss_fn=criterion,
            visualizer=visualizer,
            optimization_config=OptimizationParameters(),
        )
        return model_wrapped

class MetricsCalculator:
    def __init__(self, num_patches: int, device: torch.device):
        self._num_patches = num_patches
        self._accuracy = Accuracy(
            task="multiclass", num_classes=self._num_patches ** 2
        ).to(device)
        self._mse = MeanSquaredError().to(device)
        self._mse_weighted = MeanSquaredError().to(device)
        self._mae = MeanAbsoluteError().to(device)
        self._mae_weighted = MeanAbsoluteError().to(device)

        self._positions = torch.stack(torch.meshgrid(
            torch.arange(self._num_patches),
            torch.arange(self._num_patches),
        ), dim=-1).to(device).float() # (N, 2)
        self._positions = self._positions.view(-1, 2) # (N^2, 2)
        print("positions: ", self._positions.shape)

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        pred: (B, N1, N) - prediction (logits) for each patch (N1 patches, N possible positions)
        gt: (B, N1, 2) - (y, x) coordinates of the ground truth patches
        """

        if pred.ndim != 3:
            raise ValueError(f"pred should have 3 dimensions but got {pred.ndim}")
        if gt.ndim != 3:
            raise ValueError(f"gt should have 3 dimensions but got {gt.ndim}")
        if pred.shape[:2] != gt.shape[:2]:
            raise ValueError(f"pred and gt should have the same batch size and number of patches but got {pred.shape[:2]} and {gt.shape[:2]}")

        if pred.dtype != torch.float32:
            pred = pred.float()

        gt_flat = gt[:, :, 0] * self._num_patches + gt[:, :, 1]
        pred_pos = pred.argmax(dim=-1) # (B, N1)

        pred_y = pred_pos.div(self._num_patches)
        pred_x = pred_pos.remainder(self._num_patches)
        pred_yx = torch.stack([pred_y, pred_x], dim=-1) # (B, N1, 2)
        weights = F.softmax(pred, dim=-1) # (B, N1, N)

        pred_yx_weighted = torch.einsum("bni,ij->bnj", weights, self._positions) # (B, N1, 2)

        self._accuracy.update(pred_pos, gt_flat)
        self._mse.update(pred_yx, gt.float())
        self._mae.update(pred_yx, gt.float())

        self._mse_weighted.update(pred_yx_weighted, gt.float())
        self._mae_weighted.update(pred_yx_weighted, gt.float())

    def compute(self, suffix: str = ""):
        acc = self._accuracy.compute()
        mse = self._mse.compute()
        mae = self._mae.compute()

        return {
            f"acc{suffix}": acc.item(),
            f"mse{suffix}": mse.item(),
            f"mae{suffix}": mae.item(),
            f"mse_weighted{suffix}": self._mse_weighted.compute().item(),
            f"mae_weighted{suffix}": self._mae_weighted.compute().item(),
        }

    def reset(self):
        self._accuracy.reset()
        self._mse.reset()
        self._mae.reset()
        self._mse_weighted.reset()
        self._mae_weighted.reset()

def main(args: ArgumentsGator):
    logger.info("output_path: " + str(args.output_path))
    args.output_path.parent.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    # generate seeds for validation
    rng = torch.Generator()
    rng.manual_seed(seed)
    validation_seeds = torch.randint(
        low=0, high=1000000, size=(args.num_repeat * len(args.shuffle_ratios),), generator=rng
    ).tolist()
    logger.info("Using seeds for validation: " + str(validation_seeds))

    cudnn.benchmark = True

    data_loader_train, data_loader_eval = args.get_dataloaders()
    model_wrapped: GatorWrapper = args.get_wrapper()
    model_wrapped.to(device)
    model_wrapped.eval()
    num_patches = params.model_config.image_size // params.model_config.patch_size

    result = []

    metrics_nozero = MetricsCalculator(num_patches, device)
    metrics_zero = MetricsCalculator(num_patches, device)
    
    val_idx = 0
    for shuffle_ratio in args.shuffle_ratios:

        metrics_nozero.reset()
        metrics_zero.reset()

        for _ in range(args.num_repeat):
            val_seed = validation_seeds[val_idx]
            val_idx += 1

            for i, batch in enumerate(tqdm(data_loader_eval)):
                batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)
                
                batch_nozero = batch
                batch_zero = torch.stack([
                    batch[:, 0, :, :, :], 
                    torch.zeros_like(batch[:, 1, :, :, :]),
                ], dim=1) # (B, 2, C, H, W)

                with torch.inference_mode():
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        torch.manual_seed(val_seed)
                        out_nozero, gt_ids_nozero, num_register_tokens = model_wrapped.forward(
                            batch_nozero.to(torch.float16), 
                            shuffle_ratio=shuffle_ratio,
                        )

                        torch.manual_seed(val_seed)
                        out_zero, gt_ids_zero, num_register_tokens = model_wrapped.forward(
                            batch_zero.to(torch.float16), 
                            shuffle_ratio=shuffle_ratio,
                        )

                if not torch.all(gt_ids_nozero == gt_ids_zero):
                    raise RuntimeError("The chosen ids must be the same for correct evaluation!")
                
                metrics_nozero.update(out_nozero[:, num_register_tokens:, :], gt_ids_nozero)
                metrics_zero.update(out_zero[:, num_register_tokens:, :], gt_ids_zero)

        
        result.append({
            "shuffle_ratio": shuffle_ratio,
            **metrics_zero.compute(),
            "zero": True,
        })
        result.append({
            "shuffle_ratio": shuffle_ratio,
            **metrics_nozero.compute(),
            "zero": False,
        })

        result_pd = pd.DataFrame(result)
        result_pd.to_csv(args.output_path, index=False)
        logger.info("Saved results to " + str(args.output_path))


if __name__ == '__main__':
    params = tyro.cli(ArgumentsGator)
    main(params)
