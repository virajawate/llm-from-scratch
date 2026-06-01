from __future__ import annotations
import argparse, torch
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern
from tokenizer_bpe import BPETokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, default='')
    p.add_argument('--tokens', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['model']
    cfg = ckpt.get('config') or {}

    # Tokenizer (if Present)
    tok = None
    tok_dir_file = Path(args.ckpt).with_name('tokenizer_dir.txt')
    if tok_dir_file.exists():
        tok_dir = tok_dir_file.read_text().strip()
        tok = BPETokenizer()
        tok.load(tok_dir)
        vocab_from_tok = tok.vocab_size
    else:
        vocab_from_tok = None

    # ----- Build Config (perfer saved config : otherwise infer) -----
    if not cfg:
        V, C = sd['tok_emb.weight'].shape
        block_size = sd['pos_emb.weight'].shape[0] if 'pos_emb.weight' in sd else 256

        import re
        layer_ids = {int(m.group(1)) for k in sd.keys() if (m := re.match(r"blocks\.(\d+)\.", k))}
        n_layer = max(layer_ids) + 1 if layer_ids else 1
        n_head = 8 if C % 8 == 0 else 4 if C % 4 == 0 else 2 if C % 2 == 0 else 1
        cfg = dict(
            vocab_size=vocab_from_tok or V,
            block_size = block_size,
            n_layer = n_layer,
            n_head = n_head,
            n_embd = C, 
            dropout = 0.0,
            use_rmsnorm = True,
            rope = True, 
            max_pose = 4096,
            sliding_window=None, 
            attention_sink=0,
        )
    
    # ----- Build & Load Model -----
    model = GPTModern(**cfg).to(device).eval()
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    if tok:
        ids = tok.encode(args.prompt)
        if len(ids) == 0: ids = [10]
    else:
        ids = [10] if args.prompt == '' else list(args.prompt.encode('utf-8'))
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=args.tokens)
    output_ids = output[0].tolist()
    if tok:
        print(tok.decode(output_ids))
    else:
        print(bytes(output_ids).decode('utf-8', error='ignore'))

if __name__ == '__main__':
    main()