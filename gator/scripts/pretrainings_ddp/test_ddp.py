import tyro

from lightning import seed_everything

from gator.utils import misc
from gator.scripts.pretrainings_ddp.pretrain_jigsaw import TrainingArgumentsJigsaw
from gator import logger

def main(args: TrainingArgumentsJigsaw):
    seed_everything(args.seed)

    rank = misc.get_rank()
    world_size = misc.get_world_size()
    logger.info(f"Running on rank {rank}/{world_size}.")

    train_dataloader, eval_dataloader = args.get_dataloaders()
    for i, b in enumerate(train_dataloader):
        print(f"Train batch {i}, rank {rank}")

    for i, b in enumerate(eval_dataloader):
        print(f"Eval batch {i}, rank {rank}")

if __name__ == "__main__":
    args = tyro.cli(TrainingArgumentsJigsaw)
    main(args)
    