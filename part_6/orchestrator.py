"""
part_6/
    - []orchestrator.py
    - [X]formatters.py
    - [X]dataset_sft.py
    - [x]collator_sft.py
    - [x]curriculum.py
    - [x]evaluate.py
    - [x]train_sft.py
    - [x]sample_sft.py
    - tests/
        - []test_formatter.py
        - []test_making.py

Run from inside part_6:
    - cd part_6
    - python orchestrator.py --demo
    - pytest -q

"""
import argparse, pathlib, subprocess, sys, shlex
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd:str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="tiny SFT demo on a few samples")
    args = p.parse_args()

    run("python -m pytest -q tests/test_formatter.py")
    run("python -m pytest -q tests/test_masking.py")

    if args.demo:
        run("python train_sft.py --data huggingface --ckpt ../part_4/runs/part4_demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128")
        run("python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What are the three primary colors?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What does DNA stand for?' --tokens 30 --temperature 0.2")
        run("python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'Reverse engineer this code to create a new version\ndef factorialize(num):\n factorial = 1\n for i in range(1, num):\n factorial *= i\n \n return factorial' --tokens 64 --temperature 0.2")

    print("\nPart 6 checks complete.")