"""
Main Script

orchestrator.py         = Run Demos/Tests/Visualization
positional_embeding.py  = 1.1 Position Encodings
attn_np_demo.py         = 1.2 Self Attention Math with Tiny Numbers
single_head.py          = 1.3 Single-Head Attention (Pytorch)
multi_head.py           = 1.4 Multi-Head Attention (with shape tracing)
FFN.py                  = 1.5 Feed-Forward Network (GELU, width = multiple * model_dim)
block.py                = 1.6 Transformer Block (residuals + LayerNorm)
attn_mask.py            = Causal Mask Helpers
vis_utils.py            = Plotting Helpers (matrices and attention maps)
demo_mha_shapes.py      = prints explicit matrix multiplication and shapes step by step
demo_vis_multi_head.py  = saves attention heatmaps per head (Grid)
output/                 = (created at runtime) images and logs live here
tests/ 
    test_attn_math.py   = correctnes : tiny example v/s pytorch single-hand
    test_causal_mask.py = verifies masking behavior

"""

import subprocess, sys, pathlib, argparse, shlex

ROOT    = pathlib.Path(__file__).resolve().parent
OUTPUT  = f"{ROOT}/output"

def run(cmd:str):
    print(f"\n>> {cmd}")
    res = subprocess.run(shlex.split(cmd), cwd=ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visualize", action="store_true", help="Run Visualization scripts and save PNGs to ./out")
    args = p.parse_args()

    OUTPUT.mkdir(exist_ok=True)

    run("python attn_np_demo.py")
    run("python -m pytest -q tests/test_attn_math.py")
    run("python -m pytest -q tests/test_causal_mask.py")

    run("python demo_mha_shapes.py")

    if args.visualize:
        run("python demo_vis_multi_head.py")
        print(f"\nVisualization Images saved to : {OUTPUT}")
    
    print("\nAll Part 1 Demo/Tests Completed")

if __name__ == "__main__":
    main()