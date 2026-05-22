from pathlib import Path
from typing import Literal

import torch
import tyro

from gator.relpose.datasets import get_data_loader
from gator.relpose.datasets.base.easy_dataset import CatDataset
from gator.relpose.datasets.sevenscenes import SevenScenesRelpose
from gator.relpose.datasets.sevenscenes_retrieval import SevenScenesRetrieval
import numpy as np
import os
from dataclasses import dataclass

from gator import logger
from gator.relpose.image_retrieval.topk_retrieval import DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK, PREPROCESS_FOLDER, TopkRetrieval

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    logger.info('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    logger.info(f'Dataset length: {len(loader)}')
    return loader

@dataclass
class Arguments:
    resolution: tuple[int, int] = (224, 224)  # by default (224,224) for Reloc3r-224
    db_step: int = 1
    topk: int = 10
    cache_folder: str = PREPROCESS_FOLDER
    db_descs_file_mask: str = DB_DESCS_FILE_MASK
    pair_info_file_mask: str = PAIR_INFO_FILE_MASK
    batch_size: int = 10
    num_workers: int = 10
    scene: str = 'chess',
    amp: int = 0
    output_folder: Path = Path('./output')

    split: str = "train"
    scene: Literal['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'] = 'chess'

    device: torch.device = torch.device("cpu")
    load_bd_desc: bool = False

    def __post_init__(self):
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args: Arguments):
    if not os.path.exists(args.cache_folder):
        os.mkdir(args.cache_folder)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    runner = TopkRetrieval(args)
    dataset_db = SevenScenesRetrieval(scene=args.scene, split=args.split)

    # update or use cache
    args.load_bd_desc = os.path.exists(
        runner.db_descs_path_mask.format(dataset_db.scene, args.db_step)
    )
    runner.build_database(dataset_db)

    dataset_q = SevenScenesRetrieval(scene=args.scene, split=args.split)
    pair_info_path = '{}/{}'.format(
        args.cache_folder, 
        args.pair_info_file_mask
    ).format(dataset_q.scene, args.db_step, args.topk)

    if not os.path.exists(pair_info_path):
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)


        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        logger.info('Database-query pairs saved to {}.'.format(pair_info_path)) 
    else:
        all_retrieved = np.load(pair_info_path, allow_pickle=True)
        logger.info('Database-query pairs loaded from {}.'.format(pair_info_path))


    # data_loader_test = {
    #     pair_id: \
    #         build_dataset(
    #             SevenScenesRelpose(
    #                 scene=args.scene,
    #                 pair_id=pair_id,
    #                 resolution=args.resolution
    #             ),
    #             args.batch_size, 
    #             args.num_workers, 
    #             test='test' in args.split.lower()
    #         )
    #         for pair_id in range(args.topk)
    # }

    dataset = CatDataset([
        SevenScenesRelpose(
            scene=args.scene,
            pair_id=pair_id,
            resolution=args.resolution
        ) for pair_id in range(args.topk)
    ])

    logger.info(f"Len dataset: {len(dataset)}")

    dataloder = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        shuffle=False
    )

    logger.info(f"Len dataloader: {len(dataloder)}")
    for item in dataloder:
        logger.info(f"Batch img shape: {item[0]['img'].shape}")
        break



    # for test_name, testset in data_loader_test.items():
    #     logger.info('Testing {:s}'.format(test_name))
    #     pose_folder = '{}/poses_{}_pair-id={}'.format(args.cache_folder, testset.dataset.scene, testset.dataset.pair_id)
    #     logger.info(pose_folder)
    #     logger.info(f"Len testset: {len(testset)}")

        


if __name__ == '__main__':
    # parser = get_args_parser()
    # args = parser.parse_args()
    # run(args)
    args = tyro.cli(Arguments)
    main(args)
