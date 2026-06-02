from __future__ import annotations
import torch, torch.nn as nn

class TopKGate(nn.Module):
    """
    Top-k Softmax gating with switch-style loading-balancing aux loss.
    Args:
        dim : input hidden size
        n_expert : number of experts
        k : number of experts to route per token (1/2 typically)
    Returns:
        (indicies, weights, aux_loss) where
        indicies : (S, K) long, expert ids for each tokens
        weights : (S, K) float, gate weights (sum =< 1 per token)
        auc_loss : scalar load-balancing penalty
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1):
        pass

    def forward(self, x: torch.Tensor):
        pass