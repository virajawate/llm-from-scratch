from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))

try:
    from model_utils.model_modern import GPTModern
except Exception:
    from model_modern import GPTModern

class PolicyWithValue(nn.Module):
    """
    Policy Network => SFT LM + tiny value head
    NOTE : For Simplicity we place value head on top of LM logits (vocab+1).
        This avoids depending on hidding-state internals while keeping the tutorial runnable.
    """
    def __init__(self):
        pass

    def forward(self):
        pass

    def generate(self):
        pass