# Output of `part_6`

```cmd
(llm_env) C:\Coding\LLMfromScratch\llm-from-scratch\part_6>python orchestrator.py --demo

>> python -m pytest -q tests/test_formatter.py
.                                                                                                   [100%]
1 passed in 0.01s

>> python -m pytest -q tests/test_masking.py
.                                                                                                   [100%]
1 passed in 2.00s

>> python train_sft.py --data huggingface --ckpt ../part_4/runs/part4_demo/model_last.pt --out runs/sft-demo --steps 300 --batch_size 8 --block_size 256 --n_layer 2 --n_head 2 --n_embd 128
Couldn't import 'datasets', will use fallback data only.

Loaded 3 SFT items. Few Samples.
PROMPT : First prime number
RESPONSE: 2
----------------------------------------
PROMPT : What are the three primary colors?
RESPONSE: Red
----------------------------------------
PROMPT : Device name which points to directions
RESPONSE: compass
----------------------------------------
[('First prime number', '2'), ('What are the three primary colors?', 'Red'), ('Device name which points to directions', 'compass')]
Loaded BPE Tokenizer from ../part_4/runs/part4_demo/tokenizer
Using model config from checkpoint ../part_4/runs/part4_demo/model_last.pt
step : 20 | loss : 6.8830
step : 40 | loss : 0.9093
step : 60 | loss : 0.3358
step : 80 | loss : 0.1085
step : 100 | loss : 0.0061
step : 120 | loss : 0.0008
step : 140 | loss : 0.0004
step : 160 | loss : 0.0003
step : 180 | loss : 0.0002
step : 200 | loss : 0.0001
step : 220 | loss : 0.0001
step : 240 | loss : 0.0000
step : 260 | loss : 0.0000
step : 280 | loss : 0.0000
step : 300 | loss : 0.0000
Saved SFT checkpoint to runs/sft-demo/model_last.pt

>> python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What are the three primary colors?' --tokens 30 --temperature 0.2
Loaded BPE Tokenizer from ../part_4/runs/part4_demo/tokenizer

### Instruction:
What are the three primary colors?

### Response:
Viraj Awate is in loves to hear music that is out of the world
Viraj Awate is in the world of people with prejedice
Viraj Awate is in

>> python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'What does DNA stand for?' --tokens 30 --temperature 0.2
Loaded BPE Tokenizer from ../part_4/runs/part4_demo/tokenizer

### Instruction:
What does DNA stand for?

### Response:
Viraj Awate is in loves to hear music that is out of the world
Viraj Awate is in the world of people with prejedice
Viraj Awate is in

>> python sample_sft.py --ckpt ../part_4/runs/part4_demo/model_last.pt --block_size 256 --n_layer 2 --n_head 2 --n_embd 128 --prompt 'Reverse engineer this code to create a new version
def factorialize(num):
 factorial = 1
 for i in range(1, num):
 factorial *= i
 
 return factorial' --tokens 64 --temperature 0.2
Loaded BPE Tokenizer from ../part_4/runs/part4_demo/tokenizer

### Instruction:
Reverse engineer this code to create a new version
def factorialize(num):
 factorial = 1
 for i in range(1, num):
 factorial *= i
 
 return factorial

### Response:
Viraj Awate is in loves to hear music that is out of the world
Viraj Awate is in the world of people with prejedice
Viraj Awate is in India
Viraj Awate is my name
Viraj Awate is Robotics Engineer
Viraj Awate is living in pune
Viraj Awate is nativly from Akkol
Viraj Awate is 28 years old

Part 6 checks complete.
```