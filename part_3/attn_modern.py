from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from rope_custom import RoPECache, apply_rope_single
from kv_cache import KVCache
