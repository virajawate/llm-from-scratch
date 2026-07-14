# Part 8 output
## With Training
```sh
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_8>python orchestrator.py --demo

>> python -m pytest -q tests/test_ppo_loss.py
.                                                                                                                                                                            [100%]
1 passed in 1.41s

>> python -m pytest -q tests/test_policy_forward.py
.                                                                                                                                                                            [100%]
1 passed in 1.25s

>> python train_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --steps 100 --batch_size 4 --resp_len 128 --bpe_dir ../part_4/runs/part4_demo/tokenizer
step 10 | loss 4.8582| value loss 9.7165 | KL_move 0.000007 | KL_ref -0.000043
step 20 | loss 2.5248| value loss 5.0496 | KL_move 0.000017 | KL_ref -0.000175
step 30 | loss 1.5263| value loss 3.0525 | KL_move 0.000017 | KL_ref -0.000348
step 40 | loss 1.0383| value loss 2.0765 | KL_move 0.000075 | KL_ref -0.000875
step 50 | loss 1.0004| value loss 2.0007 | KL_move 0.000042 | KL_ref -0.000858
step 60 | loss 0.6908| value loss 1.3816 | KL_move 0.000270 | KL_ref -0.003322
step 70 | loss 0.7456| value loss 1.4911 | KL_move 0.000097 | KL_ref -0.002181
step 80 | loss 0.5634| value loss 1.1267 | KL_move -0.000145 | KL_ref 0.139694
step 90 | loss 0.6613| value loss 1.3226 | KL_move 0.001719 | KL_ref -0.013720
step 100 | loss 0.6085| value loss 1.2170 | KL_move 0.002142 | KL_ref 0.459372
Saved PPO policy to runs/ppo-demo/model_last.pt

>> python eval_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4_demo/tokenizer
Avg RM reward 1.5130
Part 8 completed.

```
## Without Training
```sh
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_8>python orchestrator.py --demo

>> python -m pytest -q tests/test_ppo_loss.py
.                                                                                                                                                                            [100%]
1 passed in 2.48s

>> python -m pytest -q tests/test_policy_forward.py
.                                                                                                                                                                            [100%]
1 passed in 1.36s

>> python eval_ppo.py --policy_ckpt ../part_6/runs/sft-demo/model_last.pt --reward_ckpt ../part_7/runs/rm-demo/model_last.pt --split train[:24] --bpe_dir ../part_4/runs/part4_demo/tokenizer
Avg RM reward 1.5130
Part 8 completed.
```