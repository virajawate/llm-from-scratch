"""
Repository Layout

Part 4/
    []orchestrator.py
    [x]tokenizer_bpe.py
    [x]dataset_bpe.py
    [x]lr_scheduler.py
    [x]amp_accum.py
    [x]checkpointing.py
    [x]logger.py
    [x]train.py
    [x]sample.py
    tests/
        [X]test_tokenizer.py
        []test_scheduler.py
        [X]test_resume_shapes.py
----
Run inside part_4:
    cd part_4
    python orchestrator.py --demo
    pytest -q
    tensorboard --logdir=runs/part4-demo
"""