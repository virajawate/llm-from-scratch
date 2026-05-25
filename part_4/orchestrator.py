"""
Repository Layout

Part 4/
    []orchestrator.py
    [x]tokenizer_bpe.py
    [x]dataset_bpe.py
    []le_scheduler.py
    []amp_accum.py
    []checkpointing.py
    []logger.py
    []train.py
    []sample.py
    tests/
        []test_tokenizer.py
        []test_scheduler.py
        []test_resume_shapes.py
----
Run inside part_4:
    cd part_4
    python orchestrator.py --demo
    pytest -q
    tensorboard --logdir=runs/part4-demo
"""