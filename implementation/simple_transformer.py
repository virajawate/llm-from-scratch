from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn

"""
Blocks of Transformer:
----------

1. Scaled - Dot Production Attention
2. Attention Block
3. Feed-Forward Block
4. Encoder Block
5. Decoder Block
"""

class Attention(nn.Module):
    def __init__(self, n_embd:int, n_head:int, dropout:float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head 
        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # scale = 1.0 / math.sqrt(self.d_head) # in below fn
        y = fn.scaled_dot_product_attention(q, k, v,attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd:int, mult:int = 4, dropout:float = 0.0):
        super().__init__()
        self.ff_seq = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff_seq(x)

class EncoderBlock(nn.Module):
    def __init__(self, n_head:int, n_embd:int, dropout:float):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_embd)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.attn = Attention(n_embd, n_head, dropout)
        self.ff_n = FeedForward(n_embd, mult=4, dropout=dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.ff_n(self.norm_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, block_size:int, vocab_size:int, n_embd:int, dropout:int, n_head:int, n_layer:int):
        super().__init__()
        self.block_size = block_size
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.postn_embd = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.EncoderBlock = nn.ModuleList([EncoderBlock(n_head, n_embd, dropout) for _ in range(n_layer)])
        self.norm_layer = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias = False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, mean=0.0, std=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx:torch.Tensor, targets:torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        pose = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embd(idx) + self.postn_embd(pose)
        x = self.drop(x)
        for encoder in self.EncoderBlock:
            x = encoder(x)
        x = self.norm_layer(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = fn.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:torch.Tensor,
        max_new_tokens:int=200,
        temperature: float = 1.0,
        top_k: int | None = 50, 
        top_p: float | None = None
    ):
        self.eval()
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = self.top_k_top_p_filtering(logits, top_k = top_k, top_p = top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def top_k_top_p_filtering(self, logits: torch.Tensor, top_k:int | None = None, top_p:int | None=None):
        """
        Filter a distribution of logits using top_k and / or nucleus (top-p) filtering.
        - logits: (B, vocab)
        Returns filtered logits with -inf for masked entries.
        """
        B, V = logits.shape
        filtered = logits.clone()

        if top_k is not None and top_k < V:
            topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
            kth = topk_vals[:, -1].unsqueeze(-1)
            filtered[filtered < kth] = float('-inf')

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum > top_p

            mask[..., 0] = False
            sorted_logits[mask] = float('-inf')
            filtered = torch.full_like(filtered, float('-inf'))
            filtered.scatter_(1, sorted_idx, sorted_logits)
        
        return filtered