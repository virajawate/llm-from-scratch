from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))
import time
import torch
import shutil
import torch.nn as nn

DEF_NAME = 'model_last.pt'

def _is_tb(logger) -> bool:
    return getattr(logger, "w", None) is not None

def _log_hparams_tb(logger, args, total_steps):
    if not _is_tb(logger): return
    try:
        h = dict(
            vocab_size = args.vocab_size,
            block_size = args.block_size,
            n_layer = args.n_layer,
            n_head = args.n_head,
            n_embd = args.n_embd,
            dropout = args.dropout,
            lr = args.lr,
            warmup_steps = args.warmup_steps,
            batch_size = args.batch_size,
            grad_accum = args.grad_accum_steps,
            mixed_precision = args.mixed_precision,
            steps = args.steps,
            epochs = args.epochs
        )
        logger.hparams(h, {"meta/total_steps" : float(total_steps)})
    except Exception:
        pass

def _maybe_log_graph_tb(logger, model, xb, yb):
    if not hasattr(logger, "graph"):
        return
    try:
        class _TensorOnly(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m.eval()
            def forward(self, x, y = None):
                output = self.m(x, y) if y is not None else self.m(x)
                if isinstance(output, (list, tuple)):
                    for o in output:
                        if torch.is_tensor(o):
                            return o
                    return output[0]
                return output
        
        wrapped = _TensorOnly(model).to(xb.device)
        logger.graph(wrapped, (xb, yb))
    except Exception:
        pass

def _log_model_stats(logger, model, step: int, do_hist: bool = False):
    if not _is_tb(logger):return
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        total_param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]), 2).item()
        grads = [p.grad for p in params if p.grad is not None]
        total_grad_norm = float('nan')
        if grads:
            total_grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
        logger.log(step = step, **{
            "train/param_global_12": total_param_norm,
            "train/grad_global_12": total_grad_norm,
        })
        if do_hist:
            for name, p in model.named_parameters():
                logger.hist(f"params/{name}", p, step)
                if p.grad is not None:
                    logger.hist(f"grads/{name}", p.grad, step)
    except Exception:
        pass

def _maybe_log_attention(logger, model, xb, step:int, every:int=100):
    """
    Logs Q/K/V Histograms for each Transformer block using the current minibatch xb.
    No Model edits, No Hooks, Runs a light no-grad recomputation of the pre-attn path.
    - Takes first batch and first head only to keep logs tiny
    - Uses pre-RoPE values (simpler & stable for histogram).
    """
    if not _is_tb(logger) or step == 0 or (step % every):
        return
    try:
        import torch
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            x = model.tok_emb(xb)
            x = model.drop(x)
            B, T, _ = x.shape

            for li, blk in enumerate(getattr(model, "blocks", [])):
                h = blk.ln1(x)
                attn = blk.attn

                q = attn.wq(h).view(B, T, attn.n_head, attn.d_head).transpose(1, 2)
                k = attn.wk(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)
                v = attn.wv(h).view(B, T, attn.n_kv_head, attn.d_head).transpose(1, 2)

                q1 = q[:1, :1].contiguous().view(-1).float().cpu()
                k1 = k[:1, :1].contiguous().view(-1).float().cpu()
                v1 = v[:1, :1].contiguous().view(-1).float().cpu()

                q1 = q1[torch.isfinite(q1)]
                k1 = k1[torch.isfinite(k1)]
                v1 = v1[torch.isfinite(v1)]

                if q1.numel() > 0: logger.hist(f"qkv/block{li}/q_hist", q1, step)
                if k1.numel() > 0: logger.hist(f"qkv/block{li}/k_hist", k1, step)
                if v1.numel() > 0: logger.hist(f"qkv/block{li}/v_hist", v1, step)

                if q1.numel(): logger.log(step=step, **{f"qkv/block{li}/q_l2_mean": float(q1.square().mean().sqrt())})
                if k1.numel(): logger.log(step=step, **{f"qkv/block{li}/k_l2_mean": float(k1.square().mean().sqrt())})
                if v1.numel(): logger.log(step=step, **{f"qkv/block{li}/v_l2_mean": float(v1.square().mean().sqrt())})

                x = x + blk.ffn(blk.ln2(x))
    except Exception as e:
        print(f"[qkv] logging failed: {e}")

def _log_runtime(logger, step:int, it_t0:float, xb, device):
    try:
        dt = time.time() - it_t0
        toks = int(xb.numel())
        toks_per_s = toks / max(dt, 1e-6)
        mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        logger.log(step = step, **{
            "sys/throughput_tokens_per_s": toks_per_s,
            "sys/step_time_s": dt,
            "sys/gpu_mem_alloc_mb": mem
        })
    except Exception:
        pass

def _log_samples_tb(logger, model, tok, xb, device, step:int, max_new_tokens:int = 64):
    if not _is_tb(logger): return
    if tok is None: return
    try:
        model.eval()
        with torch.no_grad():
            output = model.generate(xb[:1].to(device), max_new_tokens = max_new_tokens, temperature=1.0, top_k=50)
        model.train()
        text = tok.device(output[0].tolist())
        logger.text("sample/generation", text, step)
    except Exception:
        pass



# ---------------- Checkpoint Save Utils -------------
def checkpoint_paths(output_dir:Path, step:int):
    return output_dir / f"model_step{step:07d}.pt", output_dir / "model_last.pt"

def atomic_save_all(model, optim, sched, amp, step:int, output_dir:Path,
                    tok_dir:str | None, keep_last_k:int, config:dict):
    """
    Write model_last.pt (with config) + a rolling pre-step copy.
    """
    save_checkpoint(model, optim, sched, amp, step, str(output_dir), tok_dir, config=config)
    pre_step, last = checkpoint_paths(output_dir=output_dir, step)
    try:
        shutil.copy2(last, pre_step)
    except Exception:
        pass

    try:
        ckpts = sorted(output_dir.glob("model_step*.pt"))
        for old in ckpts[:-keep_last_k]:
            old.unlink(missing_ok=True)
    except Exception:
        pass