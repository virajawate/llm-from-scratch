"""
Part 2/
    orchestrator.py
    tokenizer.py
    dataset.py
    utils.py
    model_gpt.py
    train.py
    sample.py
    eval_loss.py
    tests/
        test_tokenizer.py
        test_dataset_shift.py
    runs/

NOTE ON IMPORTS
---------------
All imports are LOCAL. Run from inside "part_two/".

Example quickstart (CPU ok):
    cd part_tow
    python train.py --data tiny.txt --steps 300 --sample_every 100
    python sample.py --ckpnt runs/min-gpt/model_best.py --token 200 --prompt "Where are you going"

"""