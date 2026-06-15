from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """
    Transformer encoder -> pooled representation -> scalar reward.
    BiDirectional encoder is fine for reward modeling (not used for generation).
    """
    def __init__(self):
        pass

    def forward(self):
        pass