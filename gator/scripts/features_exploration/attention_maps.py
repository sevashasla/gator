"""
python3 gator/scripts/features_exploration/attention_maps.py \
    --ckpt-path /scratch/izar/skorokho/gator/gator-small-classification-000/checkpoints/last.ckpt \
    --model-config.enc-emb-dim 384 \
    --model-config.dec-emb-dim 384  \
    --model-config.enc-num-heads 6 \
    --model-config.dec-num-heads 6 \
    --output-dir  /scratch/izar/skorokho/gator/more-attn-vis/ \
    --ca-only \
    --num-batches 10     
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, ClassVar
from lightning import seed_everything
import torch
import torchvision
import torchvision.transforms.v2.functional as TF
import tyro

from gator.models.blocks import Attention, CrossAttention
from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.models.gator_2view.model_gator import Gator
from gator.models.gator_visualizer.base import GatorBaseVis
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.scripts.features_exploration.attention_block import SavedAttention, SavedCrossAttention
from gator.scripts.pretrainings.pretrain_gator import TrainingArgumentsGator
import gator.utils.misc as misc
import torch.backends.cudnn as cudnn
from gator import logger


@dataclass(kw_only=True)
class ShowArgumentsGator(TrainingArgumentsGator):
    ckpt_path: Path
    num_batches: int = 3
    ca_only: bool = False

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info("output_dir: " + str(self.output_dir))

    @staticmethod
    def _get_parent_module(root, module_name: str):
        """
        For 'encoder.blocks.0.attn', returns:
            parent = root.encoder.blocks[0]
            child_name = 'attn'
        """
        parts = module_name.split(".")
        parent = root

        for p in parts[:-1]:
            if p.isdigit():
                parent = parent[int(p)]
            else:
                parent = getattr(parent, p)

        return parent, parts[-1]
    
    def _get_model(self) -> torch.nn.Module:
        # model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Gator(self.model_config)

        for name, module in list(model.named_modules()):
            if isinstance(module, Attention) and (not self.ca_only):
                parent, child_name = self._get_parent_module(model, name)

                new_module = SavedAttention(
                    # default params
                    dim=module.qkv.weight.shape[1],
                    rope=module.rope,
                    num_heads=module.num_heads,
                    qkv_bias=module.qkv.bias is not None,
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    fused_attn=module.fused_attn,

                    # save path
                    save_path=self.output_dir / f"{name}-000.pth",
                )

                setattr(parent, child_name, new_module)

            if isinstance(module, CrossAttention):
                parent, child_name = self._get_parent_module(model, name)

                new_module = SavedCrossAttention(
                    # default params
                    dim=module.projq.weight.shape[1],
                    rope=module.rope,
                    num_heads=module.num_heads,
                    qkv_bias=module.projq.bias is not None,
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    fused_attn=module.fused_attn,

                    # save path
                    save_path=self.output_dir / f"{name}-000.pth",
                )

                setattr(parent, child_name, new_module)

        model.to(device)
        return model
    
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


def change_output_dir(model: Gator, new_output_dir: Path):
    for name, module in list(model.named_modules()):
        if isinstance(module, SavedAttention) or isinstance(module, SavedCrossAttention):
            module._save_path = new_output_dir / f"{name}-000.pth"


def main(args: ShowArgumentsGator):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    seed_everything(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_eval = args.get_dataloaders()
    model_wrapped: GatorWrapper = args.get_wrapper()
    model_wrapped.to(device)
    model_wrapped.eval()

    eval_length = int(args.opt_params.dataset_size * args.opt_params.tt_split_ratio) // args.opt_params.batch_size
    random_indices = torch.randperm(eval_length)[:args.num_batches].tolist()

    for i, batch in enumerate(data_loader_eval):
        if i not in random_indices:
            continue
        
        curr_output_dir = args.output_dir / f"batch_{i:03}"
        curr_output_dir.mkdir(exist_ok=True, parents=True)
        change_output_dir(
            model_wrapped._model, curr_output_dir,
        )

        batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model_wrapped.forward(
                batch.to(torch.float16), 
                shuffle_ratio=1.0,
            )

        torch.save(batch.cpu().flatten(0, 1), curr_output_dir / "input_batch.pth")


if __name__ == '__main__':
    params = tyro.cli(ShowArgumentsGator)
    main(params)
