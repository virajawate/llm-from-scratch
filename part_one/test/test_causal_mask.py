"""
Test Case:
-----------

how output of the causal mask shape
"""
import torch
from attn_mask import causal_mask

def test_mask_is_upper_triangle():
    m = causal_mask(5)
    assert m.shape == (1, 1, 5, 5)
    assert m[0,0].sum() == torch.triu(torch.ones(5,5), diagonal=1).sum()