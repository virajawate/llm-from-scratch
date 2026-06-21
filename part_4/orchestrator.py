"""
Repository Layout

Part 4/
    []orchestrator.py
    [x]tokenizer_bpe.py
    [x]dataset_bpe.py
    [x]lr_scheduler.py
    [x]amp_accum.py
    [x]checkpointing.py
    [x]logger.py
    [x]train.py
    [x]sample.py
    tests/
        [X]test_tokenizer.py
        []test_scheduler.py
        [X]test_resume_shapes.py
----
Run inside part_4:
    cd part_4
    python orchestrator.py --demo
    pytest -q
    tensorboard --logdir=runs/part4-demo
"""

import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd:str):
    print(f"\n>>{cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode !=0:
        sys.exit(res.returncode)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true', help='run a tiny smoke train + sample')
    args = p.parse_args()

    run ("python -m pytest -q tests/test_tokenizer_bpe.py")
    run ("python -m pytest -q tests/test_scheduler.py")
    run ("python -m pytest -q tests/test_resume_shapes.py")

    if args.demo:
        run("python train.py --data ../part_2/dataset/poems_eng.txt --output ./runs/part4_demo --bpe --vocab_size 8000 --epochs 1 --steps 3000 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard")
        run("python sample.py --ckpt runs/part4_demo/model_last.pt --tokens 100 --prompt 'Who is Viraj'")
    
    print("\n-----Part 4 Check Complete-----\n")