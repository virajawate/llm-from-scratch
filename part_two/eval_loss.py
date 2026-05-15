from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    d_set = ByteDataset(args.data, block_size=args.block_size)
    ckpnts = torch.load(args.ckpt, map_location=device)