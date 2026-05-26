from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))
import time
import torch
import shutil
import torch.nn as nn

DEF_NAME = 'model_last.pt'

def _is_tb(logger) -> bool:
    return getattr(logger, "w", None) is not None

def _log_hparams_tb(logger, args, total_steps):
    if not _is_tb(logger): return
    try:
        h = dict(
            vocab_size = args.vocab_size,
            block_size = args.block_size,
            n_layer = args.n_layer,
            n_head = args.n_head,
            n_embd = args.n_embd,
            dropout = args.dropout,
            lr = args.lr,
            warmup_steps = args.warmup_steps,
            batch_size = args.batch_size,
            grad_accum = args.grad_accum_steps,
            mixed_precision = args.mixed_precision,
            steps = args.steps,
            epochs = args.epochs
        )
        logger.hparams(h, {"meta/total_steps" : float(total_steps)})
    except Exception:
        pass