from __future__ import annotations

import PIL
import gradio as gr
from pathlib import Path

import torch
import torchvision

from gator.datasets.shard.transforms import get_pair_transforms_gator
from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.models.gator_2view.model_gator import Gator, GatorConfig
from gator.models.gator_losses.classification import GatorClassificationLoss
from gator.models.gator_visualizer.classification import ClassificationVis
from gator.models.jigsaw_1view.jigsaw_wrapper import OptimizationParameters
from gator.scripts.show.show_gator import shuffle_patches

_current_dir = Path(__file__).parent

class GatorGradioDemo:
    def __init__(self):
        self._config = GatorConfig(
            enc_emb_dim=384,
            enc_num_heads=6,
            dec_emb_dim=384,
            dec_num_heads=6,
        )

        self._checkpoint_path = _current_dir / "./checkpoints/gator-small-classification.ckpt"

        visualizer = ClassificationVis(grid_size=(14, 14), patch_size=16)
        loss_fn = GatorClassificationLoss(grid_size=(14, 14), patch_size=16)

        model = Gator(config=self._config)
        self._model_wrapper = GatorWrapper.load_from_checkpoint(
            checkpoint_path=self._checkpoint_path,
            model=model,
            visualizer=visualizer,
            loss_fn=loss_fn,
            optimization_config=OptimizationParameters(),
        )
        self._model_wrapper.eval()
        print(f"Loaded model from {self._checkpoint_path}!")

        self._transformation = get_pair_transforms_gator("resize224+crop224")
        self._to_pil = torchvision.transforms.ToPILImage()

    def forward(self, image1_pil: PIL.Image, image2_pil: PIL.Image) -> list[PIL.Image, PIL.Image]:
        img1, img2 = self._transformation(image1_pil, image2_pil)
        img1 = img1[None]
        img2 = img2[None]
        img1 = shuffle_patches(img1, self._model_wrapper._visualizer)

        with torch.inference_mode():
            out, chosen_ids, num_register_tokens = self._model_wrapper.forward(
                torch.stack([img1, img2], dim=1), shuffle_ratio=1.0,
            )

        images_pred = self._model_wrapper._visualizer.forward(
            pred=out.float(),
            gt_pos=chosen_ids,
            gt_image=img1,
            num_register_tokens=num_register_tokens,
        )

        return (
            self._to_pil(img1[0]),
            self._to_pil(images_pred[0]), 
        )

def main():
    print("Launching Gradio demo...")
    print("If running locally, open http://localhost:7860 in your browser.")
    
    title = "Gator Demo"
    description = "Upload two images and see how Gator unshuffles patches from one of them to match the other!"

    demo = GatorGradioDemo()
    gr.Interface(
        fn=demo.forward, 
        inputs=[
            gr.Image(type="pil", label="To-be shuffled"), 
            gr.Image(type="pil", label="Reference")
        ], 
        outputs=[
            gr.Image(type="pil", label="Shuffled"),
            gr.Image(type="pil", label="Unshuffled"), 
        ], 
        title=title,
        description=description,
        examples=[
            [
                PIL.Image.open(_current_dir / f"./gradio_examples/img{i}_1.jpg"),
                PIL.Image.open(_current_dir / f"./gradio_examples/img{i}_2.jpg"),
            ] for i in range(5)
        ]
    ).launch()

if __name__ == "__main__":
    main()