"""
1.4 Multi-Head Attention with explicit shape tracing

Dimensions (before masking):
    X: (B, T, model_dim)
    qkv: (B, T, 3*model_dim)
    view: (B, T, 3, n_head, head_dim) where head_dim = model_dim // n_head
    split: Q, K, V each (B, T, n_head, head_dim)
    swap: (B, n_head, T, head_dim)
    scores: (B, n_head, T, T) = q @ k^T / sqrt(head_dim)
    weights:(B, n_head, T, T) = softmax(scores)
    ctx: (B, n_head, T, head_dim) = weights @ V
    merge: (B, T, n_head*head_dim) = (B, T, model_dim)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, n_head: int, dropout: float = 0.0, trace_shapes: bool = True):
        super().__init__()
        assert model_dim % n_head == 0, "Model_dim must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = model_dim // n_head
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias = False)
        self.proj = nn.Linear(model_dim, model_dim, bias = False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
    
    def forward(self, x:torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        if self.trace_shapes:
            print(f"QKV view : {qkv.shape}")
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        w = self.dropout(F.softmax(attn, dim=-1))
        # w = self.dropout(w)
        ctx = torch.matmul(w, v)
        if self.trace_shapes:
            print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")
        out = ctx.transpose(1, 2).contiguous().view(B, T, C) # (B, T, model_dim)
        out = self.proj(out)
        if self.trace_shapes:
            print(f"Out : {out.shape}")
        return out, w