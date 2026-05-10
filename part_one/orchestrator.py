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
