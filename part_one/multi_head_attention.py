"""
1.4 Multi-Head Attention with explicit shape tracing

Dimensions (before masking):
    X: (B, T, model_dim)
    qkv: (B, T, 3*model_dim)
    view: (B, T, 3, n_head, head_dim) where head_dim = model_dim // n_head
    split: Q, K, V each (B, T, n_head, head_dim)
    swap: (B, n_head, T, head_dim)
    scores: (B, n_head, T, T) = q @ k^T / sqrt(head_dim)
    weights:(B, n_head, T, T) = softmax(scores)
    ctx: (B, n_head, T, head_dim) = weights @ V
    merge: (B, T, n_head*head_dim) = (B, T, model_dim)

"""
