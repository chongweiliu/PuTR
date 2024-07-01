import os
import yaml
import torch
import random
import torch.distributed
import torch.backends.cudnn
import numpy as np


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        return 1


def set_seed(seed: int):
    seed = seed + distributed_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # If you don't want to wait until the universe is silent, do not use this below code :)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    return


def yaml_to_dict(path: str):
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


