from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except:
    ByteLevelBPETokenizer = None

class BPETokenizer:
    """
    Minimal BPE wrapper (Huggingface Tokenizer).
    Trains on a text file or a folder of .txt files.
    Saves merges/vocab to output_dir.
    """
    def __init__(self, vocab_size:int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("Please 'pip install tokenizers' for BPETokenizer.")
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        files: List[str] = []
        p = Path(data_path)
        if p.is_dir():
            files = [str(fp) for fp in p.glob("**/*.txt")]
        else:
            files = [str(p)]
        tok = ByteLevelBPETokenizer()
        tok.train(files=files, vocab_size=self.vocab_size, min_frequency=2, special_tokens=self.special_tokens)
        self._tok = tok

    def save(self, output_dir:Union[str,Path]):
        output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "Train or Load before save()."
        self._tok.save_model(str(output))
        self._tok.save(str(output / "tokenizer.json"))
        meta = {"vocab_size" : self.vocab_size, "special_tokens":self.special_tokens}
        (output/"bpe_meta.json").write_text(json.dumps(meta))

    def load(self, dir_path:Union[str, Path]):
        dir_path = Path(dir_path)
        vocab = dir_path / "vocab.json"
        merges = dir_path / "merges.txt"
        tokenizer = dir_path / "tokenizer.json"
        if not vocab.exists() or not merges.exists():
            vs = list(dir_path.glob("*.json"))
            ms = list(dir_path.glob("*.txt"))
            if not vs or not ms:
                raise FileNotFoundError(f"Could not find vocab / merges in {dir_path}")
            vocab = vs[0]
            merges = ms[0]
        tok = Tokenizer.from_file(str(tokenizer))
        self._tok = tok
        meta_file = dir_path / "bpe_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            self.vocab_size = meta.get("vocab_size", self.vocab_size)
            self.special_tokens = meta.get("special_tokens", self.special_tokens)

    def encode(self, text: str):
        ids = self._tok.encode(text).ids
        return ids

    def decode(self, ids):
        return self._tok.decode(ids)