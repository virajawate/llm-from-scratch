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
        super().__init__()
        assert k >= 1 and k <= n_expert
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        logits = self.w_g(x)
        probs = torch.softmax(logits, dim = -1)
        topk_vals, topk_idx = torch.topk(probs, k = self.k, dim = -1)
        S, E = probs.size(0). probs.size(1)
        importance = probs.mean(dim=0)
        hard1 = topk_idx[:, 0]
        load = torch.zeros(E, device = x.device)
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        load = load / max(5,1)
        aux_loss = (E * (importance * load).sum())
        return topk_idx, topk_vals, aux_loss