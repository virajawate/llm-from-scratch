"""
1.6 Transformer Block :=
LN -> MHA -> residual -> LN -> FFN -> Residual
"""
import torch.nn as nn
from multi_head_attention import MultiHeadSelfAttention
from FFN import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, model_dim:int, n_head:int, dropout:float=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(model_dim)
        self.attn = MultiHeadSelfAttention(model_dim, n_head, dropout)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.ffn  = FeedForward(model_dim, multiplier=4, dropout=dropout)
    
    def forward(self, x):
        x += self.attn(self.ln_1(x))[0]
        x += self.ffn(self.ln_2(x))
        return x