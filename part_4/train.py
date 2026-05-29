from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT
