"""
    Structure
    part_8/
    - [] orchestrator.py
    - [x] policy.py
    - [x] rollout.py
    - [x] ppo_loss.py
    - [x] train_ppo.py
    - [x] eval_ppo.py
    - tests/
        - [x] test_ppo_loss.py
        - [x] test_policy_forward.py

---- Run
> cd part_8
> python orchestrator.py --demo
> pytest -q
"""
import argparse, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd: str):
    print(f"\n>> {cmd}")
    result = subprocess.run(cmd.split(), cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--demo", action='store_true', help='tiny PPO demo')
    args = p.parse_args()

    # 1) Unit Test
    run("python -m pytest -q tests/test_ppo_loss.py")
    run("python -m pytest -q tests/test_policy_forward.py")

    # 2) Optional Demo (requires SFT+RM checkpoints from Part 6 & 7)
    if args.demo:
        run(f"python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps {args.steps} --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4_demo/tokenizer")
        run("python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4_demo/tokenizer")
    
    print("Part 8 completed.")