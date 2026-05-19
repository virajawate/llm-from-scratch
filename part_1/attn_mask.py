"""
1.3.1 Util Script for causal Mask for Single Head
"""

import torch

def causal_mask(T: int, device = None):
    """
    Returns a bool mask where True means Masked
    Shape: (1, 1, T, T) 
    Suitable for broadcasting with (B, heads, T, T).
    """
    m = torch.triu(torch.ones((T, T),  dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T) 