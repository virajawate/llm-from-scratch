import torch
from tokenizer import ByteTokenizer

def test_roundtrip():
    tokn = ByteTokenizer()
    s = "Hello, ByteTokener!"
    ids = tokn.encode(s)
    assert ids.dtype == torch.long
    s2 = tokn.decode(ids)
    assert len(s2) > 0
    print(s2)