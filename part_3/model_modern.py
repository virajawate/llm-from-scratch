from __future__ import annotations
import torch
import torch.nn as nn
from block_modern import TransformerBlockModern
from tokenizer import ByteTokenizer

import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

class GPTModern(nn.Module):
    def __init__(self, vocab_size:int=256, block_size:int=256,
                 n_layer:int=4, n_head:int=4, n_embd:int=256, dropout:float=0.0,
                 use_rmsnorm:bool = True, use_swiglu:bool=True, rope:bool=True,
                 max_pos:int=4096, sliding_window:int | None = None, attention_sink:int=0,
                 n_kv_head:int | None=None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockModern(n_embd, n_head, dropout, use_rmsnorm, use_swiglu, rope, max_pos, sliding_window, n_kv_head)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)