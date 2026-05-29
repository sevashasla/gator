from dataclasses import dataclass
from pathlib import Path
from typing import Literal, ClassVar
from lightning import seed_everything
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
import tyro

from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.models.gator_visualizer.base import GatorBaseVis
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.scripts.pretrainings.pretrain_gator import TrainingArgumentsGator
import gator.utils.misc as misc
import torch.backends.cudnn as cudnn
from gator import logger


@dataclass(kw_only=True)
class ShowArgumentsGator(TrainingArgumentsGator):
    ckpt_path: Path
    sample_ratio: float = 1.0
    zero_reference: bool = False
    num_batches: int = 3

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

def shuffle_patches(
        images: torch.Tensor, 
        visualizer: GatorBaseVis,
    ) -> torch.Tensor:
    """
    images: (B, C, H, W)
    patch_size: int
    """

    patches = visualizer.patchify(images) # (B, N, D)
    B, N, D = patches.shape
    random_numbers = torch.rand(B, N, device=patches.device)
    random_order = torch.argsort(random_numbers, dim=-1) # (B, N)
    patches_reshuffled = torch.gather(
        patches, dim=1, index=random_order[:, :, None].expand(-1, -1, D)
    )
    images_reshuffled = visualizer.unpatchify(patches_reshuffled)
    return images_reshuffled

def main(args: ShowArgumentsGator):
    args.output_dir.parent.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    _, data_loader_eval = args.get_dataloaders()
    model_wrapped: GatorWrapper = args.get_wrapper()
    model_wrapped.to(device)
    model_wrapped.eval()

    eval_length = int(args.opt_params.dataset_size * args.opt_params.tt_split_ratio) // args.opt_params.batch_size
    random_indices = torch.randperm(eval_length)[:args.num_batches].tolist()

    for i, batch in enumerate(data_loader_eval):
        if i not in random_indices:
            continue

        output_path = args.output_dir / f"pred_batch_{i}.png"
        batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)

        real_images = batch[:, 0, :, :, :]
        images1 = shuffle_patches(real_images, model_wrapped._visualizer)
        batch = torch.stack([
            images1, 
            batch[:, 1, :, :, :] * (1.0 - args.zero_reference),
        ], dim=1) # (B, 2, C, H, W)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out, chosen_ids, num_register_tokens = model_wrapped.forward(
                batch.to(torch.float16), 
                shuffle_ratio=args.sample_ratio,
            )
        
        # randomly select 8 images from the batch
        indices = torch.randperm(batch.size(0))[:8]

        real_images_selected = real_images[indices, :, :, :]
        images_input = batch[indices, 0, :, :, :]
        images_ref = batch[indices, 1, :, :, :]

        # out = out.float() # (B, N_reg+N, N)
        # _, _, N = out.shape
        # pred_ids = out[indices, num_register_tokens:, :].argmax(dim=-1) # (8, N)
        # patches_input = model_wrapped._visualizer.patchify(images_input) # (8, N1, D)
        # B1, _, D = patches_input.shape
        # images_pred = torch.zeros((B1, N, D)).to(patches_input)
        # images_pred.scatter_reduce_(
        #     dim=1,
        #     index=pred_ids.unsqueeze(-1).expand(-1, -1, patches_input.size(-1)),
        #     src=patches_input,
        #     reduce="mean",
        #     include_self=False,
        # )
        # images_pred = model_wrapped._visualizer.unpatchify(images_pred) # (8, C, H, W)
        images_pred = model_wrapped._visualizer.forward(
            pred=out[indices].float(),
            gt_pos=chosen_ids[indices],
            gt_image=images1[indices],
            num_register_tokens=num_register_tokens,
        )


        # (8, C, H, W) -> (C, H, 8*W)
        images_ref_cat = torch.cat(images_ref.unbind(0), dim=-1)
        real_images_selected_cat = torch.cat(real_images_selected.unbind(0), dim=-1)
        images_input_cat = torch.cat(images_input.unbind(0), dim=-1)
        images_pred_cat = torch.cat(images_pred.unbind(0), dim=-1)

        # concatenate them into one image
        # (C, H, 8*W)x3 -> (C, 3*H, 8*W)
        images_cat = torch.cat([
            images_input_cat, 
            images_ref_cat, 
            images_pred_cat,
            real_images_selected_cat,
        ], dim=1)
        images_cat = TF.resize(images_cat, (112 * 4, 112 * 8))
        images_cat = images_cat.clamp(0, 1)
        print(output_path)
        torchvision.utils.save_image(
            images_cat.cpu(), output_path
        )


if __name__ == '__main__':
    params = tyro.cli(ShowArgumentsGator)
    main(params)
