from pathlib import Path
from typing import Literal

import torch

from gator.relpose.datasets import get_data_loader
from gator.relpose.datasets.sevenscenes import SevenScenesRelpose
from gator.relpose.datasets.sevenscenes_retrieval import SevenScenesRetrieval
import argparse
import numpy as np
import os
from dataclasses import dataclass

from gator.relpose.image_retrieval.topk_retrieval import DB_DESCS_FILE_MASK, PAIR_INFO_FILE_MASK, PREPROCESS_FOLDER, TopkRetrieval

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    print('Dataset length: ', len(loader))
    return loader

@dataclass
class Arguments:
    resolution: tuple[int, int] = (224, 224)  # by default (224,224) for Reloc3r-224
    dataset_db_str: str = "SevenScenesRetrieval(scene='{}', split='train')"
    dataset_q_str: str = "SevenScenesRetrieval(scene='{}', split='train')"
    db_step: int = 1
    topk: int = 10
    cache_folder: str = PREPROCESS_FOLDER
    db_descs_file_mask: str = DB_DESCS_FILE_MASK
    pair_info_file_mask: str = PAIR_INFO_FILE_MASK
    dataset_relpose: str = "SevenScenesRelpose(scene='{}', pair_id={}, resolution={})"
    batch_size: int = 10
    num_workers: int = 10
    scene: str = 'chess',
    amp: int = 0
    output_folder: Path = Path('./output')

    split: str = "train"
    scene: Literal['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'] = 'chess'

    def __post_init__(self):
        self.output_folder.mkdir(exist_ok=True, parents=True)
        self.cache_folder = Path(self.cache_folder)

def main(args: Arguments):
    if not os.path.exists(args.cache_folder):
        os.mkdir(args.cache_folder)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    args.device = device
    args.load_bd_desc = False 

    runner = TopkRetrieval(args)
    # dataset_db = eval(args.dataset_db)
    dataset_db = SevenScenesRetrieval(
        scene=args.scene,
        split=args.split,
    )
    runner.build_database(dataset_db)

    dataset_q = SevenScenesRetrieval(
        scene=args.scene,
        split=args.split,
    )
    all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)

    pair_info_path = '{}/{}'.format(
        args.cache_folder, 
        args.pair_info_file_mask
    ).format(dataset_q.scene, args.db_step, args.topk)

    np.save(pair_info_path, all_retrieved, allow_pickle=True)
    print('Database-query pairs saved to {}.'.format(pair_info_path)) 

    data_loader_test = {
        '{} pair_id={}'.format(args.dataset_relpose.split('(')[0], pair_id): \
            build_dataset(
                args.dataset_relpose.format(
                    args.scene, pair_id, args.resolution
                ), 
                args.batch_size, 
                args.num_workers, 
                test='test' in args.split.lower()
            )
            for pair_id in range(args.topk)
    }

    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        pose_folder = '{}/poses_{}_pair-id={}'.format(args.cache_folder, testset.dataset.scene, testset.dataset.pair_id)
        print(pose_folder)


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for visual localization')

    parser.add_argument('--resolution', 
        default=(224,224))  # by default (224,224) for Reloc3r-224

    # test set: process the database
    parser.add_argument('--dataset_db', type=str, 
        default="SevenScenesRetrieval(scene='{}', split='train')")
    parser.add_argument('--dataset_q', type=str, 
        default="SevenScenesRetrieval(scene='{}', split='train')")
    parser.add_argument('--db_step', type=int, 
        default=1, help='process all database images or skip every db_step images') 
    parser.add_argument('--topk', type=int, 
        default=10, help='topk similar images for motion averaging')
    parser.add_argument('--cache_folder', type=str, default=PREPROCESS_FOLDER)
    parser.add_argument('--db_descs_file_mask', type=str, default=DB_DESCS_FILE_MASK)
    parser.add_argument('--pair_info_file_mask', type=str, default=PAIR_INFO_FILE_MASK)

    # test set: relpose
    parser.add_argument('--dataset_relpose', type=str, 
        default="SevenScenesRelpose(scene='{}', pair_id={}, resolution={})")
    parser.add_argument('--batch_size', type=int,
        default=10)
    parser.add_argument('--num_workers', type=int,
        default=10)

    parser.add_argument('--scene', type=str, 
        default='chess', choices=['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])  
    parser.add_argument('--amp', type=int, 
        default=0,
        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")

    # parser.add_argument('--output_dir', type=str, 
    #     default='./output', help='path where to save the output') 

    return parser


def run(args):
    if not os.path.exists(args.cache_folder):
        os.mkdir(args.cache_folder)
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    args.device = device

    # set up the evaluation 
    args.pair_info_available = False
    if not args.pair_info_available:
        args.load_bd_desc = False 
        args.dataset_db = args.dataset_db.format(args.scene)
        args.dataset_q = args.dataset_q.format(args.scene)
        runner = TopkRetrieval(args)
        dataset_db = eval(args.dataset_db)
        runner.build_database(dataset_db)
        dataset_q = eval(args.dataset_q)
        all_retrieved = runner.retrieve_topk(dataset_db, dataset_q)
        pair_info_path = '{}/{}'.format(args.cache_folder, args.pair_info_file_mask).format(dataset_q.scene, args.db_step, args.topk)
        np.save(pair_info_path, all_retrieved, allow_pickle=True)
        print('Database-query pairs saved to {}.'.format(pair_info_path)) 

    data_loader_test = {'{} pair_id={}'.format(args.dataset_relpose.split('(')[0], pair_id): build_dataset(args.dataset_relpose.format(args.scene, pair_id, args.resolution), args.batch_size, args.num_workers, test=True)
                            for pair_id in range(args.topk)}
    for test_name, testset in data_loader_test.items():
        print('Testing {:s}'.format(test_name))
        pose_folder = '{}/poses_{}_pair-id={}'.format(args.cache_folder, testset.dataset.scene, testset.dataset.pair_id)
        print(pose_folder)

    # dataset1 = SevenScenesRelpose(
    #     scene='{}', 
    #     pair_id={}, 
    #     resolution=(224, 224)
    # )

    print(len(dataset1))

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    run(args)

