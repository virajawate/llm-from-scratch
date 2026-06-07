from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os
import traceback

try:
    from datasets import load_dataset
except Exception:
    print("Couldn't import 'datasets', will use fallback data only.\n")
    load_dataset = None
    
from formatters import Example

@dataclass
class SFTItem:
    prompt: str
    response: str

def load_tiny_hf(split: str = "train[:200]", sample_dataset: bool = False) -> List(SFTItem):
    """
    Try to load a tiny instruction dataset from HF
    Fallback to a baked-in list.
    We use 'tatsu-lab/alpaca' as a familiar schema (instruction, input, output) and keep only a slice.
    """
    items : List[SFTItem] = []
    if load_dataset is not None and not sample_dataset:
        try:
            ds = load_dataset("tatsu-lab/alpaca", split=split)
            for row in ds:
                instr = row.get("instruction", "").strip()
                inp = row.get("input", "").strip()
                output = row.get("output", "").strip()
                if inp:
                    instr = instr + "\n" + inp
                if instr and output:
                    items.append(SFTItem(prompt=instr, response=output))
        except Exception:
            pass
    if not items:
        # Fallback tiny list
        seeds = [
            ("First prime number", "2"),
            ("What are the three primary colors?", "Red"),
            ("Device name which points to directions", "compass")
        ]
        items = [SFTItem(prompt = p , response = r) for p, r in seeds]
    return items