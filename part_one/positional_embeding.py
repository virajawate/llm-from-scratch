"""
1.1 Positional Encoding (Absolute Learning + Sinusoidal)
"""
import math
import torch
import torch.nn as nn

class LearnedPositionEncoding(nn.Module):
    """
    Description : Nueral Network
    """

    def __init__(self, max_len: int, model_dim:int):
        super().__init__()
        self.emb = nn.Embedding(max_len, model_dim)
    
    def forward(self, x:torch.Tensor):
        """
        We need to divide B, T and Model_dim 
        Transfer T to Device
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        pose_embd = self.emb(pos)
        return x + pose_embd.unsqueeze(0)