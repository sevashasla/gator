from dataclasses import dataclass
from pathlib import Path
from typing import Literal, ClassVar
from lightning import seed_everything
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
import tyro

from gator.models.croco_wrapper import CroCoWrapper
from gator.models.gator_visualizer.denormalize import Denormalize
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.models.jigsaw_1view.model_jigsaw import _IMAGE_MEAN, _IMAGE_STD
from gator.scripts.pretrainings.pretrain_croco import TrainingArgumentsCroco
import gator.utils.misc as misc
import torch.backends.cudnn as cudnn
from gator import logger


@dataclass(kw_only=True)
class ShowArgumentsCroco(TrainingArgumentsCroco):
    ckpt_path: Path
    output_dir: ClassVar[Path] = None
    output_path: Path = Path("output.png")

    def __post_init__(self):
        pass
    
    def get_wrapper(self):
        criterion = self._get_loss()
        # It is built-it in the Wrapper
        # visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = CroCoWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model,
            loss_fn=criterion,
            optimization_config=OptimizationParameters(),
        )
        return model_wrapped

def main(args: ShowArgumentsCroco):
    logger.info("output_path: " + str(args.output_path))
    args.output_path.parent.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_eval = args.get_dataloaders()
    model_wrapped = args.get_wrapper()
    model_wrapped.to(device)
    model_wrapped.eval()

    denormalize = Denormalize(
        denormalize_imagenet="norm" in args.transforms,
        denormalize_target=bool(args.norm_pix_loss),
    )

    for _, batch in zip(range(1), data_loader_eval):
        batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out, _, _ = model_wrapped.forward(batch.to(torch.float16))

        out = out.float() # (B, N, D)
        
        # randomly select 8 images from the batch
        indices = torch.randperm(batch.size(0))[:8]

        images_gt_part = batch[indices, 0, :, :, :]
        images_ref = batch[indices, 1, :, :, :]

        images_pred = out[indices]

        patches_pred = denormalize.denormalize_pred(
            pred=images_pred,
            target=model_wrapped._model.patchify(images_gt_part),
        )
        
        images_pred = model_wrapped._model.unpatchify(patches_pred)

        images_gt_part, images_ref, images_pred = denormalize.denormalize_images(
            [images_gt_part, images_ref, images_pred]
        )

        # (8, C, H, W) -> (C, H, 8*W)
        images_ref_cat = torch.cat(images_ref.unbind(0), dim=-1)
        images_gt_cat = torch.cat(images_gt_part.unbind(0), dim=-1)
        images_pred_cat = torch.cat(images_pred.unbind(0), dim=-1)

        # concatenate them into one image
        # (C, H, 8*W)x3 -> (C, 3*H, 8*W)
        images_cat = torch.cat([
            images_ref_cat, 
            images_gt_cat, 
            images_pred_cat
        ], dim=1)
        images_cat = TF.resize(images_cat, (112 * 3, 112 * 8))
        images_cat = images_cat.clamp(0, 1)
        torchvision.utils.save_image(
            images_cat.cpu(), args.output_path
        )


if __name__ == '__main__':
    params = tyro.cli(ShowArgumentsCroco)
    main(params)
