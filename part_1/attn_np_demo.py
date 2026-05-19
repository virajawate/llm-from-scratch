"""
1.2 Self Attention Numerical Demo from First Principle

We use T=3 tokens, model_dim=4, K_dim=V_dim=2, single-head.
This script prints intermediate tensors so you can trace the math.

Dimensions Summary (Single Head)
--------------------------------
X:          (B=1, T=3, model_dim=4)
Wq/Wk/Wv:   (model_dim=4, K_dim=2)
Q, K, V:    (1, 3, 2)
Scores:     (1, 3, 3) = Q @ K^T
Weights:    (1, 3, 3) = softmax over last dim
Output:     (1, 3, 2) = Weights @ V
"""

import numpy as np
np.set_printoptions(precision=4, suppress=True)

X = np.array([[
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.4, 0.3, 0.2],
    [0.0, 0.1, 0.0, 0.1],
]], dtype=np.float32)

Wq = np.array([
    [0.2, -0.1],
    [0.0, 0.1],
    [0.1, 0.2],
    [-0.1, 0.0]
], dtype=np.float32)

Wk = np.array([
    [0.1, 0.1],
    [0.0, -0.1],
    [0.2, 0.0],
    [0.0, 0.2]
], dtype=np.float32)

Wv = np.array([
    [0.1, 0.0],
    [-0.1, 0.1],
    [0.2, -0.1],
    [0.0, 0.2]
], dtype=np.float32)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

print(f"Q shape : {Q.shape} \n----\n Q = {Q[0]} \n=====\n")
print(f"K shape : {K.shape} \n----\n K = {K[0]} \n=====\n")
print(f"V shape : {V.shape} \n----\n V = {V[0]} \n=====\n")

scale = 1.0 / np.sqrt(Q.shape[-1])
attn_scores = (Q @ K.transpose(0, 2, 1)) * scale # (1, 3, 3)

mask = np.triu(np.ones((1, 3, 3), dtype=bool), k=1)
attn_scores = np.where(mask, -1e9, attn_scores)

weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print(f"Weights shape : {weights.shape}, \nAttention Weights (causal) = \n{weights[0]}")

output = weights @ V
print(f"Output shape : {output.shape}, \nOutput = \n{output[0]}")