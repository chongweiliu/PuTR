# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
import torch

from utils.utils import distributed_rank
from .putr import build as build_putr
from .position_embedding import build as build_pe
def build_model(config: dict):
    model = build_putr(config=config)
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model

def build_position_embedding(config: dict):
    return build_pe(config=config)