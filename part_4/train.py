from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT

def estimate_loss(model: GPT, ds: ByteDataset, args)->dict:
    model.eval()
    output = {}
    with torch.no_grad():
        for split in ["trian", "val"]:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = ds.get_batch(split, args.batch_size, args.device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            output[split] = sum(losses) / len[losses]
    model.train()
    return output