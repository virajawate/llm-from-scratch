"""
Part 2/
    orchestrator.py
    tokenizer.py
    dataset.py
    utils.py
    model_gpt.py
    train.py
    sample.py
    eval_loss.py
    tests/
        test_tokenizer.py
        test_dataset_shift.py
    runs/

NOTE ON IMPORTS
---------------
All imports are LOCAL. Run from inside "part_two/".

Example quickstart (CPU ok):
    cd part_tow
    python train.py --data tiny.txt --steps 300 --sample_every 100
    python sample.py --ckpnt runs/min-gpt/model_best.py --token 200 --prompt "Where are you going"

"""

import subprocess, sys, pathlib, shlex

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / 'runs' / 'min-gpt'

def run(cmd: str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    # Quick smoke training on a tiny file path tiny_hi.txt;
    run("python train.py --data tiny_hi.txt --steps 400 --sample_every 100 --eval_interval 100 --batch_size 128 --n_layer 2 --n_head 2 --n_embd 128")

    # Sample from the best checkpoint
    run(f"python sample.py --ckpt {RUNS}/model_best.pt --tokens 200 --prompt 'Hi I am here'")

    # Evaluate Final Value Loss
    run(f"python eval_loss.py --data tiny_hi.txt --chpt {RUNS}/model_best.pt --iters 50 --block_size 128")