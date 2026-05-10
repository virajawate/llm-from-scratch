"""
1.3 Single Headed Attention (explicit shapes).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from attn_mask import casual_mask