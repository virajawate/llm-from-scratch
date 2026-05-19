"""
Visualization of MHA weights per Head [Grid]
"""
import torch
from multi_head_attention import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid

B, T, model_dim, n_head = 1, 5, 12, 3
x = torch.randn(B, T, model_dim)
attn = MultiHeadSelfAttention(model_dim, n_head, trace_shapes=False)
output, w = attn(x)
save_attention_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")