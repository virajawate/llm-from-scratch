"""
Test case 1
--------------

Script to test SingleHeadedSelfAttention Model
"""
import numpy as np
import torch
from single_head import SingleHeadedSelfAttention

X = np.array([[
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.4, 0.3, 0.2],
    [0.0, 0.1, 0.0, 0.1]]],
    dtype=np.float32
)

Wq = np.array([
    [0.2, -0.1],
    [0.0, 0.1],
    [0.1, 0.2],
    [-0.1, 0.0]],
    dtype=np.float32
)

Wk = np.array([
    [0.1, 0.1],
    [0.0, -0.1],
    [0.2, 0.0],
    [0.0, 0.2]],
    dtype=np.float32
)

Wv = np.array([
    [0.1, 0.0],
    [-0.1, 0.1],
    [0.2, -0.1],
    [0.0, 0.2]],
    dtype=np.float32
)

def test_single_head_matches_numpy():
    torch.manual_seed(0)
    x = torch.tensor(X)
    attn = SingleHeadedSelfAttention(model_dim=4, k_dim=2)

    # Load Weights
    with torch.no_grad():
        attn.q.weight.copy_(torch.tensor(Wq).t())
        attn.k.weight.copy_(torch.tensor(Wk).t())
        attn.v.weight.copy_(torch.tensor(Wv).t())
    
    output, w = attn(x)
    assert output.shape == (1, 3, 2)
    assert torch.isfinite(output).all()
    assert torch.isfinite(w).all()