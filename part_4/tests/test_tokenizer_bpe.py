import os, tempfile
from tokenizer_bpe import BPETokenizer

def test_bpe_train_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, 'tiny.txt')
        with open(txt, 'w') as f:
            f.write("Hello Hello World")
        tok = BPETokenizer(vocab_size=100)
        tok.train(txt)
        output = os.path.join(d, 'tok')
        tok.save(output)
        tok2 = BPETokenizer()
        tok2.load(output)
        ids = tok2.encode('Hello World')
        assert isinstance(ids, list) and len(ids) > 0