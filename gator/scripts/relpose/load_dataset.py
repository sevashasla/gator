from copy import deepcopy
from pathlib import Path
from typing import Literal

import torch
import tyro

from gator.relpose.datasets.base.easy_dataset import CatDataset
from gator.relpose.datasets.sevenscenes import SevenScenesRelpose
from gator.relpose.datasets.sevenscenes_retrieval import SevenScenesRetrieval
import numpy as np
import os
from dataclasses import dataclass, field

from gator import logger
from gator.relpose.image_retrieval.topk_retrieval import DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK, PREPROCESS_FOLDER, TopkRetrieval
import torchvision.transforms.v2 as transforms

@dataclass
class LoadDatasetArguments:
    resolution: tuple[int, int] = (224, 224)  # by default (224,224) for Reloc3r-224
    db_step: int = 1
    topk_train: int = 20
    """
    Number of closest neighbors to retrieve for each query image. Following
    CroCo we use topk=20 for training
    """
    
    topk_test: int = 1
    """Following CroCo we use topk=1 for testing"""

    cache_folder: str = PREPROCESS_FOLDER
    db_descs_file_mask: str = DB_DESCS_FILE_MASK
    pair_info_file_mask: str = PAIR_INFO_FILE_MASK

    device: str = "cpu"

    scenes: list[str] = field(
        default_factory=lambda: [
            "chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"
        ]
    )

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(self.cache_folder):
            os.mkdir(self.cache_folder)

        self._train_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=(0.6, 1.4), 
                contrast=(0.6, 1.4), 
                saturation=(0.6, 1.4), 
                hue=0.0
            ),
            transforms.ToTensor(),
        ])

        self._test_transform = transforms.ToTensor()

    def _prepare_train_dataset(self, scene_name: str):
        self_copy = deepcopy(self)
        pair_info_path = '{}/{}'.format(
            self_copy.cache_folder, 
            self_copy.pair_info_file_mask
        ).format(f"{scene_name}_train", self_copy.db_step, self_copy.topk_train)

        if os.path.exists(pair_info_path):
            logger.info(f'Found cached database-query pairs at {pair_info_path}.')
            return

        self_copy.topk = self.topk_train
        runner = TopkRetrieval(self_copy)
        dataset_db = SevenScenesRetrieval(scene=scene_name, split="train")
        # update or use cache
        self_copy.load_bd_desc = os.path.exists(
            runner.db_descs_path_mask.format(f"{dataset_db.scene}_train", self_copy.db_step)
        )
        runner.build_database(dataset_db)

        dataset_q = SevenScenesRetrieval(scene=scene_name, split="train")
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        logger.info('Database-query pairs saved to {}.'.format(pair_info_path)) 

    def _prepare_test_dataset(self, scene_name: str):
        self_copy = deepcopy(self)
        pair_info_path = '{}/{}'.format(
            self_copy.cache_folder, 
            self_copy.pair_info_file_mask
        ).format(f"{scene_name}_test", self_copy.db_step, self_copy.topk_test)

        if os.path.exists(pair_info_path):
            logger.info(f'Found cached database-query pairs at {pair_info_path}.')
            return

        self_copy.topk = self.topk_test
        runner = TopkRetrieval(self_copy)
        dataset_db = SevenScenesRetrieval(scene=scene_name, split="train")
        # update or use cache
        self_copy.load_bd_desc = os.path.exists(
            runner.db_descs_path_mask.format(f"{dataset_db.scene}_train", self_copy.db_step)
        )
        runner.build_database(dataset_db)
        
        dataset_q = SevenScenesRetrieval(scene=scene_name, split="test")
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        logger.info('Database-query pairs saved to {}.'.format(pair_info_path)) 


    def get_train_dataset(self):
        datasets = []
        for scene in self.scenes:
            self._prepare_train_dataset(scene)
            logger.info(f"Prepared train dataset for scene {scene}.")
            for pair_id in range(self.topk_train):
                datasets.append(
                    SevenScenesRelpose(
                        scene=scene, 
                        pair_id=pair_id, 
                        split="train",
                        resolution=self.resolution,
                        transform=self._train_transform,
                        topk=self.topk_train,
                    )
                )
        return CatDataset(datasets)


    def get_test_dataset(self):
        datasets = []
        for scene in self.scenes:
            self._prepare_test_dataset(scene)
            logger.info(f"Prepared test dataset for scene {scene}.")
            for pair_id in range(self.topk_test):
                datasets.append(
                    SevenScenesRelpose(
                        scene=scene, 
                        pair_id=pair_id, 
                        split="test",
                        resolution=self.resolution,
                        transform=self._test_transform,
                        topk=self.topk_test,
                    )
                )
        return CatDataset(datasets)

if __name__ == "__main__":
    params = tyro.cli(LoadDatasetArguments)
    train_dataset = params.get_train_dataset()
    logger.info(f"Len train dataset: {len(train_dataset)}")

    test_dataset = params.get_test_dataset()
    logger.info(f"Len test dataset: {len(test_dataset)}")
