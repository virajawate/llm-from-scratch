from __future__ import annotations
from typing import List, Tuple
import torch
import traceback

# Reuse tokenizers: Prefer BPE from Part 4 if available; else byte-level from Part 1
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

from formatters import Example, format_example, format_prompt_only

class SFTCollator:
    """
    Turn (instruction, response) into token ids and masked labels for causal LM (6.2).
    Labels for the prompt part are set to -100 so they don't contribute to loss.
    """
    def __init__(self):
        pass

    @property
    def vocab_size(self):
        pass

    def encode(self):
        pass

    def collate(self):
        pass