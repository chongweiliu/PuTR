import torch
import math
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        if self.normalize is True and self.scale is None:
            raise ValueError("Scale should be NOT NONE when normalize is True.")
        if self.scale is not None and self.normalize is False:
            raise ValueError("Normalize should be True when scale is not None.")

    def forward(self, x, y, img_h, img_w) -> torch.Tensor:
        if self.normalize:
            y *= self.scale
            x *= self.scale

        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos_x = x[:, None] / dim_i
        pos_y = y[:, None] / dim_i
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_embed = torch.cat((pos_y, pos_x), dim=1)
        return pos_embed

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_pos_feats, temperature=10000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x):
        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)

        pos_x = x[:, None] / dim_i
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)

        return pos_x

class XYWHPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, exchange_xy=True):
        super(XYWHPositionEmbedding, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi
        self.exchange_xy = exchange_xy

    def forward(self, pos_tensor) -> torch.Tensor:
        dim_i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
        dim_i = self.temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / self.num_pos_feats)
        
        pos_tensor = (pos_tensor * self.scale)[:, :, None] / dim_i[None, None, :]
        pos_tensor = pos_tensor.split([1] * pos_tensor.shape[1], dim=1)
        pos_res = [torch.stack((x[:, 0, 0::2].sin(), x[:, 0, 1::2].cos()), dim=2).flatten(1) for x in pos_tensor]
        
        if self.exchange_xy:
            pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
        pos_embed = torch.cat(pos_res, dim=1)
        return pos_embed
    

def build_xy_pe(config: dict):
    assert config["DIM"] % 2 == 0, f"Hidden dim should be 2x, but get {config['DIM']}."
    num_pos_feats = config["DIM"] / 2
    return PositionEmbeddingSine(num_pos_feats=num_pos_feats, normalize=True, scale=2*math.pi)

def build_xywh_pe(config: dict):
    assert config["DIM"] % 2 == 0, f"Hidden dim should be 2x, but get {config['DIM']}."
    num_pos_feats = config["DIM"] / 2
    return XYWHPositionEmbedding(num_pos_feats=num_pos_feats, exchange_xy=True)

def build_frame_pe(config: dict):
    return SinusoidalPositionalEmbedding(num_pos_feats=config["DIM"], temperature=10000)
