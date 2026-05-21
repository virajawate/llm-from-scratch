"""
Part 3/
    [] orchestrator.py
    [x]tokenizer.py
    [x]rmsnorm.py
    [x]rope.py
    [x]swiglu.py
    [x]kv_cache.py
    [x]attn_modern.py
    [x]block_modern.py
    [x]model_modern.py
        [x]utils.py
    [x]demo_generate.py
    tests/
        [x]test_rmsnorm.py
        [x]test_rope_apply.py
        [x]test_kvcache_shapes.py
--------------------------------
Run from inside 'part_3'
cd part_3
python orchestrator.py --demo
pytest -q    
"""

import os, argparse, pathlib, subprocess, sys, shlex

ROOT = os.path.join(os.getcwd())

def run(cmd:str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)

    if res.returncode != 0:
        sys.exit(res.returncode)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run a tiny generation demo")
    args = p.parse_args()

    run("python -m pytest -q tests/test_rmsnorm.py") 
    run("python -m pytest -q tests/test_rope_apply.py") 
    run("python -m pytest -q tests/test_kvcache_shapes.py")

    if args.demo:
        run("python demo_generate.py --rmsnorm --rope --swiglu --sliding_window 64 --sink 4 --tokens 200")
    
    print("\nPart 3 checks complete.")