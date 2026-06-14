# Output Part_5

```sh
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_5>python orchestrator.py

>> python -m pytest -q tests/test_gate_shapes.py
.                                                                                                                 [100%]
1 passed in 1.29s

>> python -m pytest -q tests/test_moe_forward.py
.                                                                                                                 [100%]
1 passed in 1.37s

>> python -m pytest -q tests/test_hybrid_block.py
.                                                                                                                 [100%]
1 passed in 1.41s

Part_5 checks complete
```

```sh
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_5>python orchestrator.py --demo

>> python -m pytest -q tests/test_gate_shapes.py
.                                                                                                                 [100%]
1 passed in 1.28s

>> python -m pytest -q tests/test_moe_forward.py
.                                                                                                                 [100%]
1 passed in 1.25s

>> python -m pytest -q tests/test_hybrid_block.py
.                                                                                                                 [100%]
1 passed in 1.23s

>> python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1
Output shape : (2, 3, 128) | aux=1.2776
Primary expert load (counts) : [3, 2, 1, 0]

Part_5 checks complete.

```

After Debugging the code
```cmd
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_5>python orchestrator.py --demo

>> python -m pytest -q tests/test_gate_shapes.py
.                                                                                                                                                                                      [100%]
1 passed in 3.37s

>> python -m pytest -q tests/test_moe_forward.py
.                                                                                                                                                                                      [100%]
1 passed in 2.03s

>> python -m pytest -q tests/test_hybrid_block.py
.                                                                                                                                                                                      [100%]
1 passed in 1.91s

>> python demo_moe.py --tokens 6 --hidden 128 --experts 4 --top_k 1
Output shape : (2, 3, 128) | aux=1.3607
Primary expert load (counts) : [3, 3, 0, 0]

Part_5 checks complete.
```