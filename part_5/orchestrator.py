"""
Part_5/
    [] orchestrator.py
    [] README_PART5_OUTPUT.md
    [x] gating.py
    [x] expert.py
    [x] moe.py
    [x] block_hybrid.py
    [x] demo_moe.py
    tests/
        [x]test_gate_shapes.py
        [x]test_moe_forward.py
        [x]test_hybrid_block.py

Run from part_5/
---
cd part_5
python orchestrator.py --demo
pytest -q

"""
import argparse, pathlib, subprocess, sys, shlex

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd:str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny MoE demo.")
    args = p.parse_args()

    run("python -m pytest -q tests/test_gate_shape.py")
    run("python -m pytest -q tests/test_moe_forward.py")
    run("python -m pytest -q tests/test_hybrid_block.py")

    if args.demo:
        run("python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1")
    
    print("\nPart_5 checks complete.")