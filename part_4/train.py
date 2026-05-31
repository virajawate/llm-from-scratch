from __future__ import annotations
import argparse, time, signal
from pathlib import Path
import sys

import torch
import torch.nn as nn

from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1] / 'part_3'))
from model_modern import GPTModern

from tokenizer_bpe import BPETokenizer
from dataset_bpe import make_loader
from lr_scheduler import WarmupCosineLR
from amp_accum import AmpGrad
from checkpointing import (
    load_checkpoint,
    _log_hparams_tb,
    _maybe_log_graph_tb,
    _maybe_log_attention,
    _is_tb,
    _log_model_stats,
    _log_samples_tb,
    _log_runtime,
    atomic_save_all
)
from logger import init_logger

def run_cfg_from_args(args, vocab_size:int)->dict:
    return dict(
        vocab_size = vocab_size,
        block_size = args.block_size,
        n_layer = args.n_layer,
        n_head = args.n_head,
        n_embd = args.n_embd,
        dropout = args.dropout,
        use_rmsnorm = True,
        use_swiglu = True,
        rope = True,
        max_pose = 4096,
        sliding_window = None,
        attention_sink=0,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--output', type=str, default = 'runs/part4')

    # Tokenizer / Model dims
    p.add_argument('--bpe', action='store_true', help='train and use a BPE tokenizer.')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)

    # Train
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=300, help="Max Optimizer steps for this run")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--mixed_precision", action='store_true')
    p.add_argument("--grad_accum_steps", type=int, default=4)
    
    # Misc
    p.add_argument('--log', choices=['wandb', 'tensorboard', 'none'], default='tensorboard')
    p.add_argument('--save_every', type=int, default=50, help='Save Checkpoint every N optimizer steps.')
    p.add_argument('--keep_last_k', type=int, default=2, help='Keep last K step checkpoint (plus model_last.pt)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output Dir and checkpoints
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir/"model_last.pt"
    have_ckpt = ckpt_path.exists()

    # --- Load Checkpoint Meta if Present
    ckpt = None
    saved_tok_dir = None
    if have_ckpt:
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "config" not in ckpt:
            raise RuntimeError(
                "Checkpoint is missing 'config',"
                "Please resave a checkpoint that includes the model config."
            )
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # ---- Tokenizer ----
    tok = None
    tok_dir = None
    if have_ckpt:
        if not saved_tok_dir:
            raise RuntimeError(
                "Checkpoint was found but tokenizer_dir.txt is missing."
                "Resume requires the original tokenizer."
            )
        tok = BPETokenizer(); tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[Resume] Loaded Tokenizer from {tok_dir} (vocab = {vocab_size})")
    else:
        if args.bpe:
            tok = BPETokenizer(vocab_size=args.vocab_size)
            tok.train(args.data)
            tok_dir = str(output_dir / 'tokenizer'); Path(tok_dir).mkdir(parents=True, exist_ok=True)
            tok.save(tok_dir)
            vocab_size = tok.vocab_size
            print(f"[init] Trained tokenizer to {tok_dir} (vocab = {vocab_size})")
        else:
            tok = None
            vocab_size = 256

    # ----- Dataset -----
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size, shuffle=True)
    
    # ----- Build Model Config -----
    if have_ckpt:
        cfg_build = ckpt["config"]
        if cfg_build.get("vocab_size") != vocab_size:
            raise RuntimeError(
                f"Tokenizer Vocab ({vocab_size}) != checkpoint config vocab ({cfg_build.get("vocab_size")})."
                "This deterministic script forbids vocab changes on resume."
            )
        else:
            cfg_build = run_cfg_from_args(args, vocab_size)
        
        # ----- init model/opt/sched/amp -----
        model = GPTModern(**cfg_build).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
        total_steps = min(args.steps, args.epochs * len(train_loader))
        warmup = min(args.warmup_steps, max(total_steps // 10, 1))
        sched = WarmupCosineLR(optim, warmup_steps=warmup, total_steps=total_steps, base_lr=args.lr)
        amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

        # ----- Strict Resume -----
        step = 0
        if have_ckpt:
            step = load_checkpoint(model, str(ckpt_path), optimizer=optim, scheduler=sched, amp=amp, strict=True)
            print(f"[Resume] Loaded Checkpoint at step {step}")

        # ----- Logging -----
        logger = init_logger(args.log, output_dir=str(output_dir))
        _log_hparams_tb(logger, args, total_steps)
        if _is_tb(logger):
            try:
                ex_x, ex_y = next(iter(train_loader))
                _maybe_log_graph_tb(logger, model, ex_x.to(device), ex_y.to(device))
            except Exception:
                pass
        
        # ----- Graceful Save on SIGINT / SIGTERM -----
        save_requested = {"flag": False}
        def _on_term(sig, frame): save_requested["flag"] = True
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

        # ----- Train Loop -----
        model.train()
        while step < args.step:
            for xb, yb in train_loader:
                if step >= args.steps: break
                if save_requested["flag"]:
                    atomic_save_all(model, optim, sched, amp, step, output_dir, tok_dir, args.keep_last_k, cfg_build)
                    print(f"[Signal] Saved Checkpoint at step {step} to {output_dir}. Exiting.")
                    return
                
                it_t0 = time.time()
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=amp.amp):
                    logits, loss, _ = model(xb, yb)
                amp.backward(loss)

                if amp.should_step():
                    amp.step(); amp.zero_grad()
                    lr = sched.step()
                    step += 1

                    # Periodic Checkpoints 
                    if step % args.save_every == 0:
                        atomic_save_all(model, optim, sched, amp, step, output_dir, tok_dir, args.keep_last_k, cfg_build)
                        if _is_tb(logger):
                            logger.text("meta/checkpoint", f"Saved at step {step}", step)
                    
                    # Logging
                    if step % 50 == 0:
                        logger.log(step=step, loss=float(loss.item()), lr=float(lr))
                        _log_runtime(logger, step, it_t0, xb, device)
                        _log_model_stats(logger, model, step, do_hist=False)
                        _maybe_log_attention(logger, model, xb, step, every=100)
                        _log_samples_tb(logger, model, tok, xb, device, step, max_new_tokens=64)
        
        # ----- Final Save -----
        atomic_save_all(model, optim, sched, amp, step, output_dir, tok_dir, args.keep_last_k, cfg_build)
        print(f"Saved Checkpoint to {output_dir}/model_last.pt")

if __name__ == "__main__":
    main()