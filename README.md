# llm-from-scratch
Following LLM from scratch tutorial

# Run `part_1` locally
```sh
cd part_one
python orchestrator.py --visualize
```

## Output of the above cmd
[Part1_output](part_1/PARTONE_README.md)

# Run `part_2` locally
```sh
cd part_2
python train.py --data tiny.txt --steps 300 --sample_every 100
python sample.py --ckpnt runs/min-gpt/model_final.py --token 200 --prompt "Where are you going"
```
[Part2_output](part_2/PARTTWO_README.md)

# Run `part_3` locally
```sh
cd part_3
python orchestrator.py --demo
```
[Part3_output](part_3/PARTTHREE_README.md)

# Run `part_4` locally
```sh
cd part_4

python orchestrator.py --demo

python train.py --data ../part_2/dataset/VirajAwate.txt --output ./runs/part4_demo --bpe --vocab_size 8000 --epochs 1 --steps 3000 --batch_size 16 --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --mixed_precision --grad_accum_steps 2 --log tensorboard

python sample.py --ckpt runs/part4_demo/model_last.pt --tokens 100 --prompt 'Who is Viraj'
```
[Part4_output](part_4/PARTFOUR_README.md)
