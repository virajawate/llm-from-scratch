# LLM and Autoencoder Learning Checklist

Work through the exercises in order. For each experiment, record the hypothesis, the one change made, the result, and what you learned. Use fixed seeds and held-out data when comparing models.

## LLM foundations and implementation

- [ ] **Trace causal self-attention** (`part_1`): write down the shapes of Q, K, V, attention scores, and outputs for a small batch; explain why scores are scaled by the head dimension.
- [ ] **Test causal masking** (`part_1`): add a test showing that changing a future token cannot affect the outputs for preceding positions.
- [ ] **Implement attention by hand** (`part_1`): compare a small NumPy or PyTorch implementation with the built-in scaled dot-product attention function.
- [ ] **Explain a transformer block** (`part_1`, `part_2`): trace residual connections, normalization, attention, and the feed-forward network; test expected tensor shapes.
- [ ] **Train a tiny GPT** (`part_2`): train on an included text, plot train and validation loss, and inspect samples from fixed prompts.
- [ ] **Run controlled language-model experiments** (`part_2`): compare context length, model size, training steps, or dropout one at a time; relate changes in validation loss to generated text.
- [ ] **Compare tokenizers** (`part_2`, `part_4`): measure token counts and validation loss with byte-level and BPE tokenization on the same text and model budget.
- [ ] **Audit checkpoint selection** (`part_2/train.py`): fix the best-validation-checkpoint comparison, initialize the best loss correctly, and add a test proving the best checkpoint is saved.
- [ ] **Explore training stability** (`part_4`): compare learning-rate schedules, gradient accumulation, and mixed precision; log validation loss and training time.

## Modern transformer features

- [ ] **Compare positional encodings** (`part_2`, `part_3`): explain learned position embeddings versus RoPE and test behavior when sequence length changes.
- [ ] **Compare normalization and MLP variants** (`part_3`): evaluate LayerNorm vs RMSNorm and GELU vs SwiGLU while keeping other settings fixed.
- [ ] **Verify KV-cache correctness** (`part_3`): compare cached and uncached next-token logits for the same prompt within a small numerical tolerance.
- [ ] **Benchmark KV caching** (`part_3`): measure generation speed and memory at several prompt lengths; explain the observed trade-offs.
- [ ] **Understand sliding-window attention** (`part_3`): test which positions remain visible and how cache length changes during generation.
- [ ] **Inspect sampling controls** (`part_2`, `part_3`): compare greedy, temperature, top-k, and top-p decoding using identical prompts and seeds.
- [ ] **Study Mixture-of-Experts routing** (`part_5`): record tokens assigned per expert for different expert counts and top-k values; explain routing and auxiliary load-balancing loss.

## Fine-tuning and preference optimization

- [ ] **Build a small SFT dataset** (`part_6`): create prompt/response examples, inspect tokenization and labels, and verify loss is applied to response tokens as intended.
- [ ] **Add held-out SFT evaluation** (`part_6`): split training and validation examples; compare base and fine-tuned models on unseen prompts and inspect responses, not just loss.
- [ ] **Check the SFT checkpoint path** (`part_6`): load the fine-tuned checkpoint for generation and confirm that evaluation actually uses it.
- [ ] **Train a reward model** (`part_7`): create preference pairs, explain the Bradley-Terry loss, and evaluate pairwise accuracy on held-out preferences.
- [ ] **Inspect PPO mechanics** (`part_8`): plot policy loss, value loss, approximate KL, and reward; explain what clipping is intended to constrain.
- [ ] **Compare PPO and GRPO** (`part_8`, `part_9`): document their different advantage/reward calculations and compare on the same small task and evaluation set.
- [ ] **Evaluate for regressions** (`part_7`-`part_9`): retain fixed held-out prompts and compare response quality before and after preference optimization; do not treat reward alone as proof of improvement.

## Autoencoders: add a new `autoencoders/` module

- [ ] **Build a dense autoencoder**: train on MNIST, explain encoder, bottleneck, and decoder dimensions, and report held-out reconstruction loss.
- [ ] **Visualize reconstructions**: show original/reconstructed image pairs and identify which details the model loses.
- [ ] **Vary the latent dimension**: compare reconstruction quality and latent size; plot the trade-off.
- [ ] **Build a convolutional autoencoder**: compare it with the dense model and explain why spatial structure matters for images.
- [ ] **Build a denoising autoencoder**: corrupt inputs with noise but train against clean targets; compare reconstructions across corruption levels.
- [ ] **Implement a variational autoencoder (VAE)**: predict latent mean and log-variance, use the reparameterization trick, and explain the reconstruction and KL-divergence terms.
- [ ] **Experiment with the VAE KL weight (beta)**: compare reconstruction sharpness and latent regularization; plot both loss terms separately.
- [ ] **Explore the latent space**: interpolate between encoded examples and decode intermediate points; describe whether transitions are smooth and meaningful.
- [ ] **Try reconstruction-based anomaly detection**: train on normal examples, choose a threshold using validation data, and report precision/recall on held-out anomalies.

## Reproducibility and project quality

- [ ] **Make experiments reproducible**: set and record random seeds, model configuration, dataset split, and dependency versions.
- [ ] **Add focused tests**: test data/target shifting, output shapes, loss finiteness, checkpoint save/load, and any corrected behavior.
- [ ] **Log useful metrics**: save train/validation curves, generation examples, and runtime so experiments can be compared later.
- [ ] **Document each project**: add a short README with setup/run commands, expected outputs, design explanations, and known limitations.
