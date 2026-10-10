# Single-H100 MoE training proposal

Review target: one full H100 with at least 80 GB VRAM. NVIDIA lists 80 GB for
[H100 SXM and native BF16 support](https://www.nvidia.com/en-us/data-center/h100/).
These are starting settings, not measured H100 performance or a guaranteed
memory fit. The repository's custom flash kernel does not use FlashAttention-3
or Tensor Core attention; routing still synchronizes with the CPU per layer.

| Setting | Yumon Code | Yumon Pet |
|---|---:|---:|
| Architecture | MoE | MoE |
| Experts / selected per token | 4 / 1 | 4 / 1 |
| Width / layers | 1024 / 24 | 1024 / 24 |
| Attention heads / head width | 32 / 32 | 32 / 32 |
| SwiGLU hidden width per expert | 4096 | 4096 |
| Context | 512 | 256 |
| Starting batch | 8 | 16 |
| Maximum input tokens per step | 4096 | 4096 |
| Learning rate, linear decay | 1e-4 to 2e-5 | 1e-4 to 1e-5 |
| Epoch budget | 15 | 15 (existing early stop applies) |
| AdamW weight decay / epsilon | 0.01 / 1e-7 | 0.01 / 1e-7 |
| Gradient norm clip | 1.0 | 1.0 |
| Dropout | 0.05 | 0.05 |
| Balance / z-loss coefficient | 0.01 / 0.001 | 0.01 / 0.001 |

Both use CUDA BF16 and BalancedCheckpointing automatically in headless Linux
builds. Right-padded batches use causal Flash Attention. Router probabilities,
balance and z-loss reductions are FP32; expert activations, weights, gradients
and optimizer moments remain BF16. There are no FP32 master weights, warmup,
gradient accumulation, or distributed training in this proposal.

At a 16,384-token vocabulary, the model has approximately **1.34B total / 436M
active parameters per token**, counting embeddings and the vocabulary head.
BF16 weights, gradients and two AdamW moments alone are roughly 10 GiB; peak
memory also includes activations, logits, temporary buffers, and allocator
overhead. Start with the batches above and measure on the pod before increasing
them. The vocabulary projection and matmul outputs remain stored under balanced
checkpointing. A 2048-wide, 24-layer, four-expert variant would be about 5.30B
total parameters, so the previous dense config should not simply acquire four
experts without revisiting memory and data requirements.

## Code configuration and launch

[`yumon-code.json`](yumon-code.json) is the executable configuration used by
Docker and `train_code`. It now selects MoE and uses a separate checkpoint root.
Architecture and expert settings do not affect the sample cache, so the current
512-token cache and tokenizer can be reused if their identity checks pass.
Increasing context requires rebuilding the cache; the whole-item chunker skips
functions/structs that exceed the context budget.

```sh
# CPU-only cache/config validation (also performed by the Docker build).
cargo run --release --no-default-features --bin train_code -- --config configs/yumon-code.json --check

# On RunPod, from the existing image's /app directory after rebuilding it:
./train_code --config configs/yumon-code.json
```

Old JSON configurations without an `architecture` field still select the dense
decoder. Explicit `"architecture": "decoder-only"` remains supported. MoE run
names include expert count and top-k to avoid loading dense checkpoints.

## Pet launch

Pet's grid in `src/brain/train.rs` supplies the dimensions above. The CLI now
defaults to MoE and honors both expert flags. Build the Pet binary/cache using
the commented Pet instructions in Dockerfile, then launch:

```sh
YUMON_SAMPLE_CACHE=/app/training-cache/samples.bin ./yumon-pet train-brain \
  --architecture moe --moe-experts 4 --moe-top-k 1 \
  --batch-size 16 --epochs 15 --out-dir /workspace/checkpoints/yumon-pet-moe-h100
```

The active Docker image still builds Code only. Pet needs its own 256-token
cache and tokenizer. Old MoE records retain their RoPE field for loading
compatibility; changing width, heads, layers, or expert settings requires a new
run. Dense model weights cannot be resumed as MoE weights.

## First pod validation

Attach persistent storage at `/workspace`; a RunPod network volume keeps
[checkpoint data beyond the pod lifecycle](https://docs.runpod.io/pods/storage/types).
Allow space for multiple full-precision checkpoint copies (about 5 GiB per
model at this vocabulary), tokenizer files, and logs. The current trainer resumes
model weights, not AdamW state or the exact within-epoch position.

Before a full run, execute the small CUDA correctness tests from a source checkout:

```sh
cargo test --release --no-default-features --features cuda-training --lib cuda_bf16_ -- --ignored --nocapture --test-threads=1
```

For a one-epoch Code pilot, copy the JSON and set `epochs` to 1 and `out_dir` to
a separate pilot path. For Pet use `--epochs 1` and a separate output directory.
Check finite loss, held-out loss/top-3 accuracy, step time, and peak GPU memory
with `nvidia-smi`. If memory permits, try doubling batch size in a separate pilot
and compare tokens/second. Small BF16 updates can round away at low learning
rates; monitor validation progress before extending the epoch budget.

No pod has been rented, training launched, or H100 benchmark performed for this
proposal. Corpus size and token counts are not sufficient here to claim this
capacity or 15 epochs are optimal; validation should determine the useful budget.
