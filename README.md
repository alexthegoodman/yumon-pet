# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

## Hyperparameters

### Medium MoE

```
{
  "num_experts": 4,
  "top_k": 1,
  "aux_loss_weight": 0.01,
  "z_loss_weight": 0.001,
  "dropout_rate": 0.05,
  "vocab_size": 4096,
  "epochs_trained": 12,
  "final_loss": 0.06480747,
  "batch_size": 8,
  "training_stage": "Language",
  "embed_dim": 256,
  "hidden_units": 256,
  "n_layers": 16,
  "attn_heads": 4,
  "ff_dim": 1024,
  "max_seq_len": 32
}
```

## Get Started

You may need clang for chat_web.

- `cargo run --release --bin yumon-pet -- train-brain` to start training on the provided (or your own) dataset
- `cargo run --release --bin chat_ui` to get started chatting
- `cargo run --release --bin yumon_world` to start a Yumon World simulation
- `cargo run --release --bin endless_data` TUI to answer endless questions in order to generate some data
- `cargo run --release --bin train_bpe` train your tokenizer on your data

- `trunk serve --release` for chat web (or `trunk build --release` for deployment)

### Training on RunPod (Docker)

The training path (`brain::train::run`, driven by `train-brain`) builds headless with
`--no-default-features` - this skips the desktop/GUI feature (native window, webview,
3D engine, gamepad) that the other bins use, so the Docker image needs no GTK/WebKit/
udev packages. It still uses burn's `Wgpu` backend (Vulkan), so all it needs at
runtime is `libvulkan1` + the GPU's driver.

1. **Build and push the image** (from a machine with Docker and the full repo checked
   out, since the image bakes in `yumon_bpe/` and the text training data):

   ```
   docker build -t alexthegoodman/yumon-brain:latest .
   docker push alexthegoodman/yumon-brain:latest
   ```

2. **Create a RunPod Network Volume** (Storage → Network Volumes) sized for your
   checkpoints, in the same region as the pod you'll launch.

3. **Launch a GPU Pod** from your custom image (`<registry>/yumon-brain:latest`),
   attaching the Network Volume at `/workspace`. The container's default command runs
   `train-brain` and writes checkpoints to `/workspace/checkpoints/brain/<run-name>/`
   (`model.bin`, `metadata.json`, tokenizer copy, and a loss-chart PNG per stage) -
   same run configs (model sizes/epochs/stages) as `src/brain/train.rs` uses locally,
   since the CLI's `--epochs`/`--batch-size`/`--max-articles` flags are legacy no-ops
   for this path; edit the `runs` vec in that file to change them. Training resumes
   automatically from whatever's already in a run's checkpoint directory.

4. **Pull checkpoints down** once you're happy with a run (or periodically - it saves
   every epoch and every 500 batches): easiest is the RunPod web File Manager on the
   pod/volume, or `runpodctl send`/`scp` if you've enabled SSH on the pod, to copy
   `/workspace/checkpoints/brain/` to your machine.

If wgpu can't find a GPU at startup, exec into the pod and run `vulkaninfo --summary`
to confirm the NVIDIA Vulkan ICD is visible (RunPod's nvidia-container-toolkit should
mount it automatically given `NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility`,
already set in the image).

## Sparse MoE training

```sh
cargo run --release --no-default-features --bin yumon-pet -- train-brain --architecture moe --moe-experts 4 --moe-top-k 1 --batch-size 16 --epochs 15 --out-dir checkpoints/moe
```

This selects `Architecture::Moe` in the existing training grid. It uses the same
stage data, AdamW, charts, periodic text generation, and checkpoint workflow.
The grid dimensions and data sources remain in `src/brain/train.rs`. The default
architecture remains xLSTM; `--architecture encoder-decoder` selects the dense
encoder/decoder. The separate `train_ui` and `chat_ui` binaries still use their
existing hardcoded models; MoE training is selected through `train-brain`.

The MoE decoder uses causal scaled attention with RoPE and sparse SwiGLU FFNs.
Every non-padding token selects `--moe-top-k` of `--moe-experts` experts. Tokens
are gathered into compact expert batches, unused experts are skipped, and the
weighted results are scattered back. No expert capacity limit or token dropping
is used. Only the integer dispatch indices are read back to the CPU; expert
weights, activations, matrix multiplies, and gradients remain on the GPU.
Top-1 is the cheapest setting. Top-k must be between 1 and the expert count.

With 4 experts and top-1, expert matrix multiplies process one quarter of the
rows required by evaluating all 4 experts for every token. This gives more
parameter capacity at roughly one dense FFN's active expert arithmetic; it does
**not** promise a 4x speedup over a single dense FFN. All experts still occupy
memory, and attention, the vocabulary projection, optimizer state, routing,
and dispatch also cost time. Variable-sized dispatch currently synchronizes
with the host once per layer, so small workloads can be slower. This is sparse
execution, not a fused grouped-GEMM kernel or multi-GPU expert parallelism.
The checked-in training entry point uses WGPU; the generic model also supports
Burn CUDA, but CUDA runtime performance must be measured on a CUDA device.

The router uses full-softmax probabilities for selected experts (without
renormalizing top-1 to 1), preserving gradients from the language loss. Training
adds [Switch-style load balancing](https://www.jmlr.org/papers/v23/21-0998.html)
(weight 0.01) and router z-loss (0.001), averaged across layers; padding is
excluded. These weights are configurable in `YumonMoeBrainConfig`. Loss charts
show language cross-entropy, while auxiliary loss and per-expert dispatched row
counts are logged every 100 batches.

Checkpoint directories include expert count and top-k. `model.bin`,
`metadata.json`, and the checkpoint's own `tokenizer.json` are saved together.
`YumonMoeBrain::load` restores the model and tokenizer for inference. Resume
rejects incompatible configurations/tokenizers and failed checkpoint loads
instead of silently starting over. As with the existing trainer, optimizer
state is reinitialized on resume; this is a weights resume, not an exact restart.

Run correctness checks and the GPU timing probe with:

```sh
cargo test --no-default-features --lib moe_ -- --test-threads=1
cargo test --release --no-default-features --lib moe_timing_probe -- --ignored --nocapture --test-threads=1
```

The timing probe includes forward/backward and host dispatch overhead, comparing
sparse execution to a dense masked reference with the same router and experts.
It excludes attention, the vocabulary head, and optimizer updates, so it measures
the FFN benefit rather than claiming an end-to-end training speedup.


### Datasets

- Custom / Bespoke
- https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations
- https://www.kaggle.com/datasets/thedevastator/distillchat-v1-mixture-of-conversations-dataset

## Evaluation

### Yumon characteristics

Primary:
- Teachable (do this, go there, get that + reward signals and lesson cache)
- Conversational (what do you think about... + memory strength)
- Smart (model parameters + data)

Secondary:
- Loyal
- Connective
- Entertaining
- Organizational
- Affordable

## TODO

- Training UI (train tokenizer, organize data, run structured and unstructured training sessions, etc)