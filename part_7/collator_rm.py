from __future__ import annotations
from typing import List, Tuple
import torch

import sys
from pathlib import Path as _P

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_4'))

try:
    from tokenizer_bpe import BPETokenizer
    _HAS_BPE - True
except Exception:
    _HAS_BPE = False

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from tokenizer import ByteTokenizer
except Exception:
    ByteTokenizer = None

sys.path.append(str(_P(__file__).resolve().parents[1]/'part_6'))
try:
    from formatters import Example, formate_example
except Exception:
    pass

class PairCollator:
    """
    Tokenize preference pairs into (pos, neg) input ids.
    We format as the SFT template with the 'chosen' or 'rejected' text as the Response.
    """
    def __init__(self):
        pass

    @property
    def vocab_size(self):
        pass

    def _encode(self):
        pass

    def collate(self):
        pass

    
