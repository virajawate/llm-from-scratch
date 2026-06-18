## Output

```cmd
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_7>python orchestrator.py

>> python -m pytest -q tests/test_bt_loss.py
.                                                                                                [100%]
1 passed in 1.25s

>> python -m pytest -q tests/test_reward_forward.py
.                                                                                                [100%]
1 passed in 1.76s

Part 7 checks complete.
```

```cmd
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_7>python orchestrator.py --demo

>> python -m pytest -q tests/test_bt_loss.py
.                                                                                                [100%]
1 passed in 1.27s

>> python -m pytest -q tests/test_reward_forward.py
.                                                                                                [100%]
1 passed in 1.55s

>> python train_rm.py --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --loss bt --bpe_dir ../part_4/runs/part4-demo/tokenizer
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Step 25 | Loss 0.6852 | Acc 0.75
Step 50 | Loss 0.6819 | Acc 0.62
Step 75 | Loss 0.6717 | Acc 0.50
Step 100 | Loss 0.6389 | Acc 0.50
Step 125 | Loss 0.6208 | Acc 0.62
Step 150 | Loss 0.4472 | Acc 0.62
Step 175 | Loss 0.6314 | Acc 0.38
Step 200 | Loss 0.3025 | Acc 0.75
Step 225 | Loss 0.6601 | Acc 0.50
Step 250 | Loss 0.2592 | Acc 0.88
Step 275 | Loss 0.6373 | Acc 0.38
Step 300 | Loss 0.2811 | Acc 0.62
Saved reward model to runs/rm-demo/model_last.pt

>> python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split train[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
C:\Users\viraj\miniconda3\envs\llm_env\Lib\site-packages\torch\nn\modules\transformer.py:531: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\NestedTensorImpl.cpp:182.)
  output = torch._nested_tensor_from_mask(
Pairs = 8 accuracy (r_pos > r_neg) = 0.500

>> python eval_rm.py --ckpt runs/rm-demo/model_last.pt --split test[:8] --bpe_dir ../part_4/runs/part4-demo/tokenizer
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
C:\Users\viraj\miniconda3\envs\llm_env\Lib\site-packages\torch\nn\modules\transformer.py:531: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\NestedTensorImpl.cpp:182.)
  output = torch._nested_tensor_from_mask(
Pairs = 8 accuracy (r_pos > r_neg) = 0.000

Part 7 checks complete.
```