from __future__ import annotations
from typing import List, Tuple
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
    def __init__(
            self,
            block_size : int,
            bpe_dir : str | None = None,
            vocab_size:int = 8000,
            ):
        self.block_size = block_size
        self.tok = None
        if _HAS_BPE:
            try:
                self.tok = BPETokenizer(vocab_size=vocab_size)
                if bpe_dir:
                    self.tok.load(bpe_dir)
            except Exception:
                self.tok = None
        if self.tok is None and ByteTokenizer is not None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("No Tokenizer available for RLHF")
    
    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)

    def encode(self, text:str) -> List[int]:
        ids = self.tok.encode(text)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ids

    def decode(self, ids : List[int]) -> str:
        if hasattr(self.tok, 'decode'):
            return self.tok.decode(ids)
        return bytes(ids).decode('utf-8', errors='ignore')
