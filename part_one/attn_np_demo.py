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
