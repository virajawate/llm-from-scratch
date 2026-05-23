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