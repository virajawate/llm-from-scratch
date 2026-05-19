from __future__ import annotations
import torch
from dataclasses import dataclass

@property
class KVCache:
    k: torch.Tensor
    v: torch.Tensor

    @property
    def T(self):
        return self.k.size(2)