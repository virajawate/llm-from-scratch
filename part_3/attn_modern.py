from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd:int, n_head:int, dropout:float=0.0,
                 rope:bool=True, max_pose:int=4096,
                 sliding_window:int | None = None, attention_sink:int = 0,
                 n_kv_head:int | None = None):
        super().__init__()
        assert n_embd % n_head == 0,  "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        self.wq = nn.Linear(n_embd, self.n_head * self.d_head, bias = False)
        self.wk = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias = False)
        self.wv = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias = False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pose = max_pose
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink
        
    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pose, device = device)
    
    def forward(self, x:torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """
        x : (B, T, C).
        If kv_cache given, we assume generation (T small, often 1).
        """
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device = x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)
            k = apply_rope_single(k, cos, sin)
        
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim = 2)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v
        
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)
        
        # ----- GQA expand : repeat K / V heads to match Q heads before attention ----
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)
        else:
            k_attn, v_attn = k_all, v_all
        
        is_casual = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_casual)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache