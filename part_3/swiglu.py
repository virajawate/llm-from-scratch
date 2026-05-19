import torch.nn as nn

class SwiGLU(nn.Module):
    """
    SwiGLU FFN : (xW1) (x) swish(xW2) = W3 with expansion factor 'mult'.
    """
