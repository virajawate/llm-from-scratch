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

import os, subprocess, sys, shlex

ROOT = os.getcwd()
RUNS = os.path.join(ROOT,'runs','min-gpt')
CKPNT = os.path.join(ROOT,'runs','min-gpt', 'model_final.pt')
# DATA = os.path.join(ROOT, 'dataset', 'poems_eng.txt')
DATA = os.path.join(ROOT, 'dataset', 'AliceAdventure.txt')
def run(cmd: str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    # Quick smoke training on a tiny file path tiny_hi.txt;
    run(f"python train.py --data '{DATA}' --steps 10000 --sample_every 100 --eval_interval 100 --amp --dropout 0.2")
    # Sample Run
    run(f"python sample.py --ckpt '{CKPNT}' --tokens 200 --prompt 'Rain on a dark day'")
    # Evaluate Final Value Loss
    run(f"python eval_loss.py --data '{DATA}' --ckpt '{CKPNT}' --iters 50 --block_size 256")