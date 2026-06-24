from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))

try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))

try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from part_6.formatters import Example, format_example, format_prompt_only

# tokenizer helper

class RLHFTokenizer:
    def __init__(self):
        pass
    
    @property
    def vocab_size(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass
