from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .dancetrack import build as build_dancetrack
from .mot17 import build as build_mot17
from .mix import build as build_mix
from torch.utils.data import Dataset
from .utils import collate_fn
from utils.utils import is_distributed


def build_dataset(config: dict, split: str) -> Dataset:
    if config["DATASET"] == "DanceTrack":
        return build_dancetrack(config=config, split=split)
    elif config["DATASET"] == "SportsMOT":
        return build_dancetrack(config=config, split=split)
    elif config["DATASET"] == "MOT17":
        return build_mot17(config=config, split=split)
    elif config["DATASET"] == "MOT20":
        return build_mot17(config=config, split=split)
    elif config["DATASET"] == "MIX":
        return build_mix(config=config, split=split)
    else:
        raise ValueError(f"Dataset {config['DATASET']} is not supported!")


def build_sampler(dataset: Dataset, shuffle: bool):
    if is_distributed():
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle is True else SequentialSampler(dataset)
    return sampler


def build_dataloader(dataset: Dataset, sampler, batch_size: int, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        prefetch_factor=1
    )

