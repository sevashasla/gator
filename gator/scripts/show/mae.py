from dataclasses import dataclass
from pathlib import Path
from typing import Literal, ClassVar
from lightning import seed_everything
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
import tyro

from gator.models.gator_visualizer.denormalize import Denormalize
from gator.models.mae_1view.wrapper_mae import MAE1ViewWrapper
from gator.scripts.pretrainings.pretrain_mae import TrainingArgumentsMAE
import gator.utils.misc as misc
import torch.backends.cudnn as cudnn
from gator import logger


@dataclass(kw_only=True)
class ShowArgumentsMAE(TrainingArgumentsMAE):
    ckpt_path: Path
    output_dir: ClassVar[Path] = None
    output_path: Path = Path("output.png")

    def __post_init__(self):
        pass
    
    def get_wrapper(self):
        criterion = self._get_loss()
        visualizer = self._get_visualizer()
        model = self._get_model()
        
        # wrapper
        model_wrapped = MAE1ViewWrapper.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            model=model, 
            loss_fn=criterion,
            optimization_config=self.opt_params,
            visualizer=visualizer,
        )
        return model_wrapped

def main(args: ShowArgumentsMAE):
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

    denormalize = Denormalize(
        denormalize_imagenet="norm" in args.transforms,
        denormalize_target=bool(args.norm_pix_loss),
    )

    model_wrapped.eval()
    for _, batch in zip(range(1), data_loader_eval):
        batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out, mask_taken, num_register_tokens = model_wrapped.forward(batch.to(torch.float16))

        out = out.float() # (B, N, D)
        
        # randomly select 8 images from the batch
        indices = torch.randperm(batch.size(0))[:8]

        images_gt_part = batch[indices, 0, :, :, :]
        images_ref = batch[indices, 1, :, :, :]

        # (B, C, H, W) -> (B, N, D)
        images_pred = model_wrapped._visualizer.forward(
            pred=out[indices],
            gt_pos=mask_taken[indices],
            gt_image=images_gt_part,
            num_register_tokens=num_register_tokens,
        )
        
        patches_pred = denormalize.denormalize_pred(
            pred=model_wrapped._visualizer.patchify(images_pred),
            target=model_wrapped._visualizer.patchify(images_gt_part),
        )
        images_pred = model_wrapped._visualizer.unpatchify(patches_pred)
        
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
    params = tyro.cli(ShowArgumentsMAE)
    main(params)
