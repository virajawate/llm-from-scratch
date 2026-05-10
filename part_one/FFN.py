"""
1.5 FFN (Fully Connected Feed Forward Network)
Dimensions:
input - (B, T, model_dim)
inner - (B, T, multi * model_dim)
output- (B, T, model_dim)
"""
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, model_dim:int, multiplier:int=4, dropout:float=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, multiplier * model_dim),
            nn.GELU(),
            nn.Linear(multiplier * model_dim, model_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)