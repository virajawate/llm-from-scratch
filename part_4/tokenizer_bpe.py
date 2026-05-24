from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Union

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
except:
    ByteLevelBPETokenizer = None

class BPETTokenizer:
    """
    Minimal BPE wrapper (Huggingface Tokenizer).
    Trains on a text file or a folder of .txt files.
    Saves merges/vocab to output_dir.
    """