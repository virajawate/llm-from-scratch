# Output part_4 
```cmd
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_4>python orchestrator.py --demo

>>python -m pytest -q tests/test_tokenizer_bpe.py
.                                                                                            [100%]
1 passed in 0.02s

>>python -m pytest -q tests/test_scheduler.py
.                                                                                            [100%]
1 passed in 0.01s

>>python -m pytest -q tests/test_resume_shapes.py
.                                                                                            [100%]
1 passed in 2.28s

```
```cmd
>>python train.py --data ../part_2/dataset/VirajAwate.txt --output ./runs/part4_demo --bpe --vocab_size 8000 --epochs 1 --steps 3000 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard
[00:00:00] Pre-processing files (0 Mo)    ██████████████████████████████████████                100%
[00:00:00] Tokenize words                 ██████████████████████████████████████ 40       /       40
[00:00:00] Count pairs                    ██████████████████████████████████████ 40       /       40
[00:00:00] Compute merges                 ██████████████████████████████████████ 139      /      139
[init] Trained tokenizer to runs\part4_demo\tokenizer (vocab = 8000)
C:\Coding\LLMfromScratch\llm-from-scratch\part_3\model_modern.py:30: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert T <= self.block_size
C:\Coding\LLMfromScratch\llm-from-scratch\part_3\rope_custom.py:22: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
C:\Coding\LLMfromScratch\llm-from-scratch\part_3\rope_custom.py:22: TracerWarning: Converting a tensor to a Python number might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
C:\Coding\LLMfromScratch\llm-from-scratch\part_3\rope_custom.py:46: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  assert x.size(-1) % 2 == 0
Saved Checkpoint to runs\part4_demo/model_last.pt
```
```cmd
>>python sample.py --ckpt runs/part4_demo/model_last.pt --tokens 100 --prompt 'Who is Viraj'
Who is Viraj Awate is in private company
Viraj Awate is in a good company
Viraj Awate is friends with Rajath Vineet Rohit
Viraj Awate is in love with philosophy
Viraj Awate is in loves to hear music that is out of the world
Viraj Awate is in the world of people with prejedice
Viraj Awate is in India
Viraj Awate is my name
Viraj Awate is Robotics Engineer
Viraj Awate is living in pune
Viraj Awate is nativly from Akkol
Viraj Awate is 28 years old
Viraj Awate is in
```