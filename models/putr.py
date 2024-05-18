import math
import inspect
from typing import Any, Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint as chp

class Attention(nn.Module):
    def __init__(self, head_dim: int, dim: int, n_heads: int, drop_out: float=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.dim = dim
        self.drop_out = drop_out
        
        self.wq = nn.Linear(self.dim, self.dim, bias=False)
        self.wk = nn.Linear(self.dim, self.dim, bias=False)
        self.wv = nn.Linear(self.dim, self.dim, bias=False)
        self.wo = nn.Linear(self.dim, self.dim, bias=False)

        self.attn_dropout = nn.Dropout(self.drop_out)
        self.resid_dropout = nn.Dropout(self.drop_out)


        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        assert self.flash, "Flash Attention requires PyTorch >= 2.0"

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

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask, dropout_p=self.drop_out if self.training else 0.0, is_causal=True if mask is None else False)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop_out: float=0.0):
        super().__init__()
        self.linear_1 = nn.Linear(dim, hidden_dim, bias=False)
        self.linear_2 = nn.Linear(hidden_dim, dim, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, norm_eps: float, drop_out: float=0.0):
        super().__init__()
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.attention = Attention(
            head_dim=self.head_dim,
            dim=dim,
            n_heads=n_heads,
            drop_out=drop_out)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            drop_out=drop_out,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, xy_pos: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), xy_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class PuTR(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, norm_eps: float, patch_grid: int, drop_out: float=0.0):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.norm_eps = norm_eps
        self.patch_grid = patch_grid
        self.drop_out = drop_out
        
        self.input_dim = self.patch_grid * self.patch_grid * 3
        self.dropout = nn.Dropout(self.drop_out)
        self.to_patch_embedding = nn.Linear(self.input_dim, self.dim, bias=False)
        self.id_embedding = nn.Embedding(300, dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.dim, self.n_heads, self.norm_eps, self.drop_out))
        self.norm = nn.LayerNorm(self.dim, eps=self.norm_eps)
        self.mlp_head = nn.Linear(dim, self.dim)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))
        self.save_memory = True

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, frame_pos: torch.Tensor, xy_pos: torch.Tensor, mask: Optional[torch.Tensor] = None, is_projected=False) -> torch.Tensor:
        if not is_projected:
            h = self.to_patch_embedding(tokens)
        else:
            h = tokens
            
        h = h + frame_pos
        
        h = self.dropout(h)

        for layer in self.layers:
            if self.save_memory and self.training:
                h = chp.checkpoint(layer, h, xy_pos, mask)
            else:
                h = layer(h, xy_pos, mask)
                
        if self.save_memory and self.training:
            h = chp.checkpoint(self.norm, h)
            return chp.checkpoint(self.mlp_head, h)
        else:
            h = self.norm(h)
            return self.mlp_head(h)
    
    def project(self, tokens):
        return self.to_patch_embedding(tokens)

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all the candidate parameters
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
        drop_out=config.get("DROP_OUT", 0.0)
        )
