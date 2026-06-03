from __future__ import annotations
import torch.nn as nn
from moe import MoE

class HybridFFN(nn.Module):
    """
    Blend dense FFN with MoE
    output : y = a * Dense(x) + (1-a) * MoE(x)
    Use a (- [0, 1] to trade between stability (dense) and capacity (MoE).
    """
    def __init__(self):
        pass

    def forward(self):
        pass