I scanned the repository outside `part_9`. It builds from attention basics through a small GPT, modern transformer components, training improvements, MoE, supervised fine-tuning (SFT), reward modeling, and PPO.

## A good first project: study how context and data affect generation

Start with `part_2`, then use `part_4` to compare a few controlled training runs.

1. Train the small GPT on one of the included texts, such as `part_2/dataset/AliceAdventure.txt`.
2. Save a fixed set of prompts and generation settings.
3. Change **one factor at a time**: training steps, context length, or tokenizer.
4. Compare the generated text and validation loss. Keep a short log of each run and what you expected to happen.

This gives you a concrete way to connect tokenization, next-token prediction, atteVntion, and training to behavior you can observe. `part_2` uses byte-level data; `part_4` adds BPE tokenization and more training controls.

## Then build up in stages

- **Inspect attention:** In `part_1`, trace how the causal mask prevents a token from seeing future tokens. Try a few inputs and compare the attention shapes and outputs.
- **Compare transformer components:** In `part_3`, study RoPE, RMSNorm, SwiGLU, and KV caching. Change one component at a time and observe its effect on generation or inference.
- **Explore MoE:** In `part_5`, vary the number of experts and `top_k`. Track which experts receive tokens and how the auxiliary loss relates to load balance.
- **Make a tiny instruction model:** In `part_6`, create a small prompt/response dataset and fine-tune the base model. Keep training and evaluation examples separate so you can tell memorization from generalization.
- **Study preference training last:** Parts `7` and `8` introduce reward modeling and PPO. Treat them as experiments in how feedback changes a policy, and evaluate on held-out examples.

## A few things to check before trusting the later demos

- The Part 6 output shows generated samples that look like the base model’s personal-data training text. The logged sample commands point at the Part 4 checkpoint, rather than the SFT checkpoint under `part_6/runs/sft-demo`. So those outputs don’t demonstrate whether SFT worked.
- The Part 6 demo reports only three fallback examples because the `datasets` package could not be imported. With so few examples, a near-zero training loss is evidence of fitting those examples, not broad instruction-following ability.
- Part 7’s displayed test accuracy is `0.000` on only eight pairs, and Part 8’s reported reward is not enough by itself to show PPO improved responses. Use held-out prompts and inspect the generated answers as well as the scores.
- The top-level README has a likely typo: it says `cd part_one`, while the folder is `part_1`.

A useful habit throughout: keep a small experiment notebook with the change, expected result, actual result, and a few generated samples. That turns the tutorial code into a sequence of testable explanations rather than just a set of demos.