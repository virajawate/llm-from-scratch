"""
Walkthrough of multi-head attention with explicit matrix math and shapes.
Generates a text log at ./output/mha_shapes.txt
"""

import os
import math
import torch
from multi_head_attention import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), "out", "mha_shapes.txt")

def log(s):
    print(s)
    with open(OUT_TXT, 'a') as f:
        f.write(s+"/n")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, 'w').close()

    B, T, model_dim, n_head = 1, 5, 12, 3
    head_dim = model_dim // n_head
    x = torch.randn(B, T, model_dim)
    attn = MultiHeadSelfAttention(model_dim, n_head, trace_shapes=True)

    log(f"Input X : {tuple(x.shape)} = (B, T, model_dim)")
    qkv = attn.qkv(x)
    log(f"Linear qkv(x) : {tuple(qkv.shape)} = (B, T, 3 * model_dim)")
    qkv = qkv.view(B, T, 3, n_head, head_dim)
    log(f"View to 5D : {tuple(qkv.shape)} = (B, T, 3, heads, head_dim)")
    q, k, v = qkv.unbind(dim=2)
    log(f"Q, K, V split : \nQ = {tuple(q.shape)} \nK = {tuple(k.shape)} \nV = {tuple(v.shape)}")

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    log(f"Transpose heads: \nQ = {tuple(q.shape)} \nK = {tuple(k.shape)} \nV = {tuple(v.shape)} = (B, heads, T, head_dim)")

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    log(f"Scores Q @ K^t : {tuple(scores.shape)} = (B, heads, T, T)")
    
    weights = torch.softmax(scores, dim=-1)
    log(f"Weights(softmax) : {tuple(weights.shape)} = (B, heads, T, T)")
    
    ctx = torch.matmul(weights, v)
    log(f"context @ v : {tuple(ctx.shape)} = (B, heads, T, head_dim)")
    
    output = ctx.transpose(1, 2).contiguous().view(B, T, model_dim)
    log(f"Merge heads : {tuple(output.shape)} = (B, T, model_dim)")
    
    output = attn.proj(output)
    log(f"Final Proj  : {tuple(output.shape)} = (B, T, model_dim)")

    log("---------\nLegend:\n")
    log("B is batch \nT is sequence length \nmodel_dim is Embedding size \nn_head is number of Heads \nhead_dim is model_dim / n_head")
    log("QKV(x) is a single linear producing [Q|K|V]; \nWe reshape then split into Q, K, V")
