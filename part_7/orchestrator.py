"""
Part_7/
    - [x] orchestrator.py
    - [x] data_prefs.py
    - [x] collator_rm.py
    - [x] model_reward.py
    - [x] loss_reward.py
    - [x] train_rm.py
    - [x] eval_rm.py
    - tests/
        - [x] test_bt_loss.py
        - [x] test_reward_forward.py

Run the orchestrator
cd part_7
python orchestrator.py --demo
pytest -q
"""
import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action="store_true", help="tiny reward-model demo")
    args = p.parse_args()

    run("python -m pytest -q tests/test_bt_loss.py")
    run("python -m pytest -q tests/test_reward_forward.py")

    if args.demo:
        run("python train_rm.py --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --loss bt --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split train[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")
        run("python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer")
    
    print("\nPart 7 checks complete.")