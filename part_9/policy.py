from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _p

sys.path.append(str(_p(__file__).resolve().parents[1]/'part_3'))
try:
    from model_utils.model_modern import GPTModern # User-custom Path
except Exception:
    from model_modern import GPTModern # Fallout

class PolicyWithValue(nn.Module):
    """
    Policy network = SFT LM + tiny value head.
    NOTE: For simplicity we place value head on top of LM logits (vocab-1).
    This avoids depending on hidden-state internals while keeping the tutorial runnable.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self):
        pass

    def generate(self):
        pass
