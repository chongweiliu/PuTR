import math
import inspect
from typing import Any, Optional, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as chp

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class Attention(nn.Module):
    def __init__(self, head_dim: int, dim: int, n_heads: int, max_seq_len: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.drop_out = drop_out
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.drop_out)
        self.resid_dropout = nn.Dropout(self.drop_out)

        assert hasattr(torch.nn.functional, 'scaled_dot_product_attention'), "Flash Attention requires PyTorch >= 2.0"

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x + pos), self.wk(x + pos), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=self.drop_out if self.training else 0.0, is_causal=True if mask is None else False)
        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, drop_out: float=0.0):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        return self.drop_out(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
class MLP(nn.Module):
    """The implementation of simple multi-layer perceptron layer
    without dropout and identity connection.

    The feature process order follows `Linear -> ReLU -> Linear -> ReLU -> ...`.

    Args:
        input_dim (int): The input feature dimension.
        hidden_dim (int): The hidden dimension of MLPs.
        output_dim (int): the output feature dimension of MLPs.
        num_layer (int): The number of FC layer used in MLPs.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, bias: bool=False, act: bool=False
    ) -> torch.Tensor:
        super().__init__()
        self.num_layers = num_layers
        self.act = act
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k, bias=bias) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        """Forward function of `MLP`.

        Args:
            x (torch.Tensor): the input tensor used in `MLP` layers.

        Returns:
            torch.Tensor: the forward results of `MLP` layer
        """
        if self.act:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, max_seq_len: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = Attention(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            drop_out=drop_out)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=256,
            drop_out=drop_out
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class PuTR(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, norm_eps: float, patch_grid: int, max_seq_len: int, drop_out: float=0.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.norm_eps = norm_eps
        self.patch_grid = patch_grid
        self.max_seq_len = max_seq_len
        self.drop_out = drop_out
        
        self.input_dim = self.patch_grid * self.patch_grid * 3
        self.dropout = nn.Dropout(self.drop_out)
        self.to_patch_embedding = nn.Linear(self.input_dim, self.dim, bias=False)
        self.id_embedding = nn.Embedding(300, dim)
        
        self.xywh_pos_head = MLP(dim * 2, dim, dim, 2)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.dim, self.n_heads, self.norm_eps, self.max_seq_len, self.drop_out))
        self.norm = nn.ModuleList([RMSNorm(self.dim, eps=self.norm_eps) for _ in range(self.n_layers)])
        self.class_embed = nn.ModuleList([nn.Linear(dim, self.dim) for _ in range(self.n_layers)])
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
        self.save_memory = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, frame_pos: torch.Tensor, xywh_pos: torch.Tensor, mask: Optional[torch.Tensor] = None, is_projected=False) -> torch.Tensor:
        if not is_projected:
            h = self.to_patch_embedding(tokens)
        else:
            h = tokens
        h = h + frame_pos
        h = self.dropout(h)
        
        xywh_pos = self.xywh_pos_head(xywh_pos)
        
        outputs_classes = []
        for layer_idx, layer in enumerate(self.layers):
            if self.save_memory and self.training:
                h = chp.checkpoint(layer, h, xywh_pos, mask)
            else:
                h = layer(h, xywh_pos, mask)
            outputs_classes.append(self.class_embed[layer_idx](self.norm[layer_idx](h)))

        outputs_classes = torch.stack(outputs_classes)

        if self.training:
            return outputs_classes
        return outputs_classes[-1]
    
    def project(self, tokens):
        return self.to_patch_embedding(tokens)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer, ["decay_params", "nodecay_params"]

def build(config: dict):
    return PuTR(
        dim=config["DIM"],
        n_layers=config["N_LAYERS"],
        n_heads=config["N_HEADS"],
        norm_eps=config["NORM_EPS"],
        patch_grid=config["PATCH_GRID"],
        max_seq_len=config["MAX_SEQ_LEN"],
        drop_out=config.get("DROP_OUT", 0.0)
        )