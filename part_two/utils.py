"""
Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
-logits: (B, vocab)
Returns filtered logits with -inf for masked entries.
"""