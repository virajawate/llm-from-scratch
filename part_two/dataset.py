"""
Holds RAW bytes of a text file and yields (x, y) blocks for LM.
- block_size    : Sequence Length (context window)
- split         : Fraction for training (rest is val)
"""