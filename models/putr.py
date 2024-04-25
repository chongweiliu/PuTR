# -*- coding: utf-8 -*-
# @Time     : 2024/4/14 18:12
# @Author   : Liu Chongwei
# @FileName : putr
# @Software : PyCharm

# modified from https://github.com/DLLXW/baby-llama2-chinese/blob/main/model.py

import math
import inspect
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint as chp

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


        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
            # mask = torch.triu(mask, diagonal=1)
            # self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x + pos), self.wk(x + pos), self.wv(x)
        # xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
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
        # else:
        #     # manual implementation
        #     scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        #     assert hasattr(self, 'mask')
        #     scores = scores + self.mask[:, :, :seq_len, :seq_len]   # (bs, n_heads, seq_len, cache_len + seq_len)
        #     scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        #     scores = self.attn_dropout(scores)
        #     output = torch.matmul(scores, xv)  # (bs, n_heads, seq_len, head_dim)

        # restore time as batch dimension and concat heads
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
            drop_out=drop_out,
        )
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, xy_pos: torch.Tensor, frame_pos: torch.Tensor, id_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x = x + frame_pos + id_emb
        h = x + self.attention(self.attention_norm(x), xy_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class PuTR(nn.Module):
    last_loss: Optional[torch.Tensor]

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
        self.n_id_embedding = 300
        self.id_embedding = nn.Embedding(self.n_id_embedding, dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.dim, self.n_heads, self.norm_eps, self.max_seq_len, self.drop_out))
        self.norm = nn.LayerNorm(self.dim, eps=self.norm_eps)
        self.mlp_head = nn.Linear(dim, self.dim)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.save_memory = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, frame_pos: torch.Tensor, xy_pos: torch.Tensor, id_emb_idx: torch.Tensor, mask: Optional[torch.Tensor] = None, is_projected=False) -> torch.Tensor:
        if not is_projected:
            h = self.to_patch_embedding(tokens)
        else:
            h = tokens
        id_emb = 1 #self.id_embedding(id_emb_idx)
        # id_emb = id_emb * (id_emb_idx != 0).float().unsqueeze(-1)
        h = h + frame_pos
        
        h = self.dropout(h)

        for layer in self.layers:
            if self.save_memory:
                h = chp.checkpoint(layer, h, xy_pos, frame_pos, id_emb, mask)
            else:
                h = layer(h, xy_pos, frame_pos, id_emb, mask)
        if self.save_memory:
            h = chp.checkpoint(self.norm, h)
            return chp.checkpoint(self.mlp_head, h)
        else:
            h = self.norm(h)
            return self.mlp_head(h)

    
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


    #@torch.inference_mode()
    @torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next==eos:
                break

        return idx

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
