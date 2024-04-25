import math
import os
import torch
import random
import numpy as np
from math import floor

from collections import defaultdict
from random import randint
import cv2

from .mot17 import MOT17
from .dancetrack import DanceTrack

import data.transforms as T
from torch.utils.data import Dataset
class MIX(Dataset):
    def __init__(self, config: dict, split: str, transform):
        super(MIX, self).__init__()
        
        self.config = config
        _config = config.copy()
        _config["DATASET"] = "MOT17"
        self.mot17 = MOT17(_config, split, transform)
        _config["DATASET"] = "MOT20"
        self.mot20 = MOT17(_config, split, transform)
        _config["DATASET"] = "DanceTrack"
        self.dancetrack = DanceTrack(_config, split, transform)
        _config["DATASET"] = "SportsMOT"
        self.sportsmot = MOT17(_config, split, transform)
        self.set_epoch(0)
        
        return

    def set_epoch(self, epoch: int):
        self.mot17.set_epoch(epoch)
        self.mot20.set_epoch(epoch)
        self.dancetrack.set_epoch(epoch)
        self.sportsmot.set_epoch(epoch)
        self.lengths = [len(self.mot17), len(self.mot20), len(self.dancetrack), len(self.sportsmot)]
        self.length_sum = sum(self.lengths)
        self.len_cumsum = np.cumsum([0] + self.lengths)
        return

    def __getitem__(self, item):
        if item < self.len_cumsum[1]:
            return self.mot17[item]
        elif item < self.len_cumsum[2]:
            return self.mot20[item - self.len_cumsum[1]]
        elif item < self.len_cumsum[3]:
            return self.dancetrack[item - self.len_cumsum[2]]
        else:
            return self.sportsmot[item - self.len_cumsum[3]]
        return
    
    def __len__(self):
        return self.length_sum


def transforms_for_train(n_grid, coco_size: bool = False, overflow_bbox: bool = False, reverse_clip: bool = False):
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]  # from MOTR
    return T.MultiCompose(
        [
            T.MultiRandomHorizontalFlip(),
            # T.MultiRandomResize(sizes=scales, max_size=1536),
            
            T.MultiRandomSelect(
                T.MultiRandomResize(sizes=scales, max_size=1536),
                T.MultiCompose(
                    [
                        T.MultiRandomResize([400, 500, 600] if coco_size else [800, 1000, 1200]),
                        T.MultiRandomCrop(
                            min_size=384 if coco_size else 800,
                            max_size=600 if coco_size else 1200,
                            overflow_bbox=overflow_bbox
                        ),
                        T.MultiRandomResize(sizes=scales, max_size=1536)
                    ])
            ),
            
            T.MultiHSV(),
        ])



def build(config: dict, split: str):
    if split == "train":
        return MIX(
            config=config,
            split=split,
            transform=transforms_for_train(
                n_grid=config["PATCH_GRID"],
                coco_size=config["COCO_SIZE"],
                overflow_bbox=config["OVERFLOW_BBOX"],
                reverse_clip=config["REVERSE_CLIP"]
            )
        )
    else:
        raise NotImplementedError(f"MOT Dataset 'build' function do not support split {split}.")