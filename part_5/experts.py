from __future__ import annotations
import torch.nn as nn

class ExpertMLP(nn.Module):
    """
    Single Expert MLP (SwiGLU or GLU).
    """
    def __init__(self, dim:int, mult:int = 4, swiglu:bool=True, dropout:float=0.0):
        pass

    def froward(self, x):
        pass