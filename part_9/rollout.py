from __future__ import annotations
import torch
from typing import List, Tuple

import sys
from pathlib import Path as _p
sys.path.append(str(_p(__file__).resolve().parents[1]/'part_4'))

try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE = True
except Exception:
    _HAS_BPE = False
sys.path.append(str(_p(__file__).resolve().parents[1]/'part_3'))

try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

from part_6.formatters import Example, format_example, format_prompt_only

# ------------- Tokenizer Helpers -----------------
class RLHFTokenizer:
    def __init__(
            self,
            block_size: int, 
            bpe_dir:str | None = None, 
            vocab_size:int = 8000
        ):
        self.block_size = block_size
        self.tok = None
    
    @property
    def vocab_size(self) ->int:
        pass

    def encode(self):
        pass

    def decode(self):
        pass