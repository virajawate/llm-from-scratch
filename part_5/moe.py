from __future__ import annotations
import torch, torch.nn as nn
from gating import TopKGate
from experts import ExpertMLP

class MoE(nn.Module):
    """
    Mixture-of-Experts Layer (token-wise top-k routing)\
    Implementation is single-GPU friendly (loops over experts for clarity)
    """
    def __init__(self):
        pass

    def forward(self):
        pass
    