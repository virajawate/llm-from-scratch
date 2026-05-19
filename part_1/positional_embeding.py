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

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, model_dim: int):
        super().__init__()
        pose_embd = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(1e4) / model_dim))
        pose_embd[:, 0::2] = torch.sin(position * div_term)
        pose_embd[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pose_embd)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)