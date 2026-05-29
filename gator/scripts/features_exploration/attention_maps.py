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

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info("output_dir: " + str(self.output_dir))

    def _get_model(self) -> torch.nn.Module:
        # model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Gator(self.model_config)

        # swap attention modules into SavedAttention to save the qkv tensors
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


        for name, module in list(model.named_modules()):
            if isinstance(module, Attention):
                parent, child_name = _get_parent_module(model, name)

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
                parent, child_name = _get_parent_module(model, name)

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

    for _, batch in zip(range(1), data_loader_eval):
        batch = torch.stack(batch, dim=1).to(device) # (B, 2, C, H, W)
        
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model_wrapped.forward(
                batch.to(torch.float16), 
                shuffle_ratio=1.0,
            )

        torch.save(batch.cpu().flatten(0, 1), args.output_dir / "input_batch.pth")


if __name__ == '__main__':
    params = tyro.cli(ShowArgumentsGator)
    main(params)
