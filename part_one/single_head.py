"""
1.3 Single Headed Attention (explicit shapes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from attn_mask import casual_mask

class SingleHeadedSelfAttention(nn.Module):
    def __init__(self, model_dim:int, k_dim: int, dropout:float=0.0, trace_shapes:bool=False):
        super().__init__()
        self.q = nn.Linear(model_dim, k_dim, bias=False)
        self.k = nn.Linear(model_dim, k_dim, bias=False)
        self.v = nn.Linear(model_dim, k_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes
    
    def forward(self, x:torch.Tensor):
        B, T, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        if self.trace_shapes:
            print(f"q = {q.shape} | k = {k.shape} | v = {v.shape}")
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale 
        mask = casual_mask(T, device=x.device)
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        w = f.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)
        if self.trace_shapes:
            print(f"Weights {w.shape} \nOutput {out.shape}")
        return out, w