from __future__ import annotations
import argparse, torch

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels
from rollout import model_logprobs

import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel
from ppo_loss import ppo_losses

def compute_reward(reward_model : RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = reward_model(x)
    return float(r[0].item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Tokenizer
    tok = RLHFTokenizer(
        block_size=args.block_size,
        bpe_dir=args.bpe_dir
    )

    # Load SFT policy as initial policy AND reference
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head = cfg.get('n_head', 2)
    n_embd = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model']) # initialize LM weight from SFT

    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(
        vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size),
        block_size=rckpt['config'].get('block_size', tok.block_size),
        n_layer=rckpt['config'].get('n_layer', 4),
        n_head=rckpt['config'].get('n_head', 4),
        n_embd=rckpt['config'].get('n_embd', 256),
    ).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Small Prompt Pool
    prompts = sample_prompts(16)

    step = 0
    while step < args.step:
        # -------- COLLECT ROLLOUT BATCH --------
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts)]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device = device)
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)
                out_ids.append(out[0].tolist())
        
        # Split Prompt / Response per Sample
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            # find boundary : index where prompt ends in the tokenized form
            # Use original prompt tokenization length (clipped by block_size)
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]
            # Compute rewards via RM on formatted prompt + response text
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))
        
        # Pad to same length