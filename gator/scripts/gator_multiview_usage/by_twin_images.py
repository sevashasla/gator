import numpy as np
from dataclasses import dataclass
import torch
import tyro
from pathlib import Path
import torch.backends.cudnn as cudnn

from lightning import seed_everything
import yaml
from gator.models.gator_2view.gator_wrapper import GatorWrapper
from gator.scripts.gator_multiview_usage.swap_rectangles import swap_two_rectangles
from gator.scripts.pretrainings.pretrain_gator import TrainingArgumentsGator
from gator import logger
import cv2
import torchvision

from torchmetrics import Accuracy

@dataclass(kw_only=True)
class ArgumentsGator(TrainingArgumentsGator):
    ckpt_path: Path
    dataset_config_path: Path = Path("./docs/diffusion_editing/twoview_check.yml")

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

class TwinDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_config_path: Path,
            transforms_pair,
        ):
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        self._dataset_config = dataset_config
        self._transforms_pair = transforms_pair

        if "edited_root" not in self._dataset_config:
            raise RuntimeError("edited_root not found in dataset config")
        if "images" not in self._dataset_config:
            raise RuntimeError("images not found in dataset config")

    def _get_ids_of_interest(self, rect: tuple[int, int, int, int]):
        y_begin, x_begin, height, width = rect
        y_end = y_begin + height
        x_end = x_begin + width
        return np.stack(np.meshgrid(
            np.arange(y_begin, y_end),
            np.arange(x_begin, x_end),
        )).T.reshape(-1, 2)

    def __len__(self):
        return len(self._dataset_config["images"])
    
    def __getitem__(self, idx):
        image_info = self._dataset_config["images"][idx]
        image1 = cv2.imread(str(Path(self._dataset_config["edited_root"]) / image_info["name"]))
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1, (224, 224), cv2.INTER_AREA)

        # swap rectangles based
        image2 = swap_two_rectangles(
            image1,
            rectangle_1=image_info["rectangle1"],
            rectangle_2=image_info["rectangle2"],
        )

        image1 = torch.from_numpy(image1).permute(2, 0, 1) # (C, H, W)
        image2 = torch.from_numpy(image2).permute(2, 0, 1) # (C, H, W)

        # get indices of interest for the first patch
        ids_of_interest1 = self._get_ids_of_interest(image_info["rectangle1"])
        ids_of_interest2 = self._get_ids_of_interest(image_info["rectangle2"])

        return image1, image2, ids_of_interest1, ids_of_interest2
    
    def collate_fn(self, batch):
        """
        batch - list of tuples (image1, image2, ids_of_interest1, ids_of_interest2)
        """

        # shape? I hope it is (B, 2, C, H, W) 
        images = torch.stack([
            torch.stack(self._transforms_pair(image1, image2)) for image1, image2, _, _ in batch
        ], dim=0)

        ids_of_interest1 = [torch.from_numpy(ids1) for _, _, ids1, _ in batch]
        ids_of_interest2 = [torch.from_numpy(ids2) for _, _, _, ids2 in batch]

        return images, ids_of_interest1, ids_of_interest2

def check_multiview_gator(params: ArgumentsGator):
    # logger.info("output_path: " + str(args.output_path))
    # args.output_path.parent.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed
    seed_everything(seed)

    cudnn.benchmark = True

    # get model
    model_wrapped = args.get_wrapper()
    model_wrapped.to(device)

    # read dataset
    dataset = TwinDataset(
        args.dataset_config_path,
        transforms_pair=args._get_eval_transform(),
    )
    print("Length:", len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.opt_params.batch_size, 
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    num_patches = params.model_config.image_size // params.model_config.patch_size
    accuracy_calculator = Accuracy(
        task="multiclass", 
        num_classes=num_patches ** 2
    ).to(device)

    for i, b in enumerate(dataloader):
        images, ids_of_interest1, ids_of_interest2 = b
        images = images.to(device)

        images12 = images
        images21 = torch.stack([images[:, 1, :, :, :], images[:, 0, :, :, :]], dim=1)
        
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out12, gt_ids, num_register_tokens = model_wrapped.forward(
                    images12.to(torch.float16), 
                    shuffle_ratio=1.0
                )

                out21, gt_ids, num_register_tokens = model_wrapped.forward(
                    images21.to(torch.float16), 
                    shuffle_ratio=1.0
                )

        for j in range(b[0].shape[0]):
            ids_of_interest1_flat = ids_of_interest1[j][:, 0] * num_patches + ids_of_interest1[j][:, 1] # (B, N)
            ids_of_interest1_flat = ids_of_interest1_flat.to(device)
            ids_of_interest2_flat = ids_of_interest2[j][:, 0] * num_patches + ids_of_interest2[j][:, 1] # (B, N)
            ids_of_interest2_flat = ids_of_interest2_flat.to(device)

            pred12 = out12[j, num_register_tokens:, :].argmax(dim=-1) # (B, N)
            pred12 = pred12[ids_of_interest1_flat] # (B, N1)
            pred21 = out21[j, num_register_tokens:, :].argmax(dim=-1) # (B, N)
            pred21 = pred21[ids_of_interest2_flat] # (B, N2)

            # pick the ids of interest
            accuracy_calculator.update(pred12.flatten(), ids_of_interest2_flat.flatten())
            accuracy_calculator.update(pred21.flatten(), ids_of_interest1_flat.flatten())

        images_pred_1 = model_wrapped._visualizer.forward(
            pred=out12.float(),
            gt_pos=gt_ids,
            gt_image=images12[:, 0, :, :, :],
            num_register_tokens=num_register_tokens,
        )
        print("images_pred_1.shape:", images_pred_1.shape)
        # (B, 3, C, H, W)
        images_exp_12 = torch.cat([
            images12,
            images_pred_1[:, None, :, :, :],
        ], dim=1)
        B, _, C, H, W = images_exp_12.shape
        print("images_exp_12.shape:", images_exp_12.shape)

        # (B, 3, C, H, W) -> (C, 3, H, B, W) -> (C, 3*H, B*W)
        images_exp_12 = images_exp_12.permute(2, 1, 3, 0, 4).flatten(3, 4).flatten(1, 2)
        print("images_exp_12.shape:", images_exp_12.shape)

        images_pred_2 = model_wrapped._visualizer.forward(
            pred=out21.float(),
            gt_pos=gt_ids,
            gt_image=images21[:, 0, :, :, :],
            num_register_tokens=num_register_tokens,
        ) # (B, C, H, W)
        print("images_exp_2.shape:", images_pred_2.shape)

        images_exp_21 = torch.cat([
            images21,
            images_pred_2[:, None, :, :, :],
        ], dim=1)
        print("images_exp_21.shape:", images_exp_21.shape)
        images_exp_21 = images_exp_21.permute(2, 1, 3, 0, 4).flatten(3, 4).flatten(1, 2)
        print("images_exp_21.shape:", images_exp_21.shape)

        images_exp = torch.cat([images_exp_12, images_exp_21], dim=-1) # (C, 3*H, 2*B*W)
        print("images_exp.shape:", images_exp.shape)

        images_exp[:, :, B*W] = 0.0 # make a black separating line
        torchvision.utils.save_image(images_exp, args.output_dir / f"twin_images_{i:03}.png")

    accuracy = accuracy_calculator.compute()
    logger.info(f"Accuracy: {accuracy:.4f}")

    with open(args.output_dir / "twin_images_accuracy.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")


if __name__ == '__main__':
    args = tyro.cli(ArgumentsGator)
    check_multiview_gator(args)
