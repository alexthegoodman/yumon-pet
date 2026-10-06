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
- `cargo run --release --bin yumon_universe -- --checkpoint <language-checkpoint> --seed 42 --theme suburban|urban` to explore a procedural neighborhood or downtown (Windows, desktop feature)
- `cargo run --release --bin yumon_rss -- --checkpoint <language-checkpoint>` for a desktop Product Hunt companion (desktop feature)
- `cargo run --release --bin endless_data` TUI to answer endless questions in order to generate some data
- `cargo run --release --bin train_bpe` train your tokenizer on your data

- `trunk serve --release` for chat web (or `trunk build --release` for deployment)

### Yumon RSS

Yumon RSS sits in the bottom-left desktop corner in a transparent, always-on-top
window. It loads Product Hunt's public daily Atom feed and caches the product
list on disk. No API token is needed. Yumon reacts to one product description
on launch, then another every five minutes after each comment completes. Each
card shows the product, its feed description (usually a short tagline), and
Yumon's comment. This covers products available in the public feed, rather than
every submission or full product-page description.

The snapshot is reused for the rest of the local calendar day, including after
restarting. At the first turn after the date changes, a new snapshot is fetched.
Submission dates are not used to filter products: a product may be featured later
than its submission. Refresh reloads the feed to pick up new arrivals while
preserving the comment queue and five-minute timer. Once the queue is exhausted,
Yumon waits for a refresh or the next day. Scroll through the products and
comments, click a product to open it in your browser, or drag the header to move
the window.

The **Chat** tab lets you send messages to the same local Yumon model. Press Enter
to send or Shift+Enter for a new line. The product feed continues on its timer
while you chat; sending waits briefly if Yumon is already reading a product.
Chat stays available across tab switches and keeps the latest 200 messages for
the current session. It is not saved when the app closes. Yumon answers the
current message directly; earlier chat messages are displayed but are not added
to the model prompt. Long messages are shortened to fit the checkpoint's context.

```sh
cargo run --release --bin yumon_rss -- --checkpoint <language-checkpoint> --architecture moe
cargo run --release --bin yumon_rss -- --check-feeds
```

The default checkpoint matches `chat_ui` and `yumon_universe`. Use
`--architecture moe|xlstm|encoder-decoder` to match your **Language-stage**
checkpoint. Long descriptions are shortened for the model's context window while
the complete feed description remains visible. Comments come from the local model and may
be imperfect.

The latest 200 entries are saved in the platform's local application data folder
(`%LOCALAPPDATA%\Yumon\YumonRSS\data\history.json` on Windows). Use
`--history <path>` to choose a different JSON file. The daily snapshot and completed
product IDs are saved in `producthunt-cache.json` beside history; use `--cache <path>`
to override it. Restarting resumes the queue without repeating completed products
from that snapshot. `--check-feeds` prints cached products and descriptions without
loading a model or window (and fetches them if no current cache exists).
Unreadable history or cache files produce an error rather than being overwritten.
Feed failures appear in the window and retry at the next turn; an existing cache
for today can still be used if a manual refresh fails.

```sh
cargo test --bin yumon_rss
```


### Training on RunPod (Docker)

The next Language run uses a **256-token** context. Chat inputs include earlier
turns from the same conversation as plain dialogue, for example:

```text
Human: My favorite color is blue.
Yumon: I like blue too.
Human: What color did I choose?
```

The target remains just `You chose blue.` (plus the existing EOS token). With no
history, the input remains the original message. The latest complete turns are
kept in chronological order; oldest turns are dropped to leave space for BOS,
the existing Language separator, the current message, and the full reply with
EOS. Samples whose current message and reply cannot fit are skipped. Short
replies stay eligible at 256 tokens. Independent sentence/pair sources have no
invented conversation history.

Verify the formatting, token budgets, unchanged targets, conversation isolation,
and a sample of the real chat corpus without a GPU training run:

```sh
cargo test --lib --no-default-features language_ -- --nocapture
```

Rebuild the Docker image to include these changes. Run directories now contain
`256len`, so the trainer starts a new context-size run rather than resuming a
32-token checkpoint. The existing MoE expert/top-k sweep is retained. A longer
context increases GPU memory use; choose `--batch-size` for the pod's capacity
when using MoE.

The training path (`brain::train::run`, driven by `train-brain`) builds headless with
`--no-default-features` - this skips the desktop/GUI feature (native window, webview,
3D engine, gamepad) that the other bins use, so the Docker image needs no GTK/WebKit/
udev packages. Headless Linux training uses Burn's CUDA backend; desktop/local
training retains WGPU. The Docker runtime includes CUDA's runtime compiler and
uses the pod's NVIDIA driver.

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
   configured in the training grid. MoE honors `--epochs` and `--batch-size`;
   other architectures use the grid's duration and batch size. `--max-articles`
   is unused for this path. Training resumes
   automatically from whatever's already in a run's checkpoint directory.

4. **Pull checkpoints down** once you're happy with a run (or periodically - it saves
   every epoch and every 500 batches): easiest is the RunPod web File Manager on the
   pod/volume, or `runpodctl send`/`scp` if you've enabled SSH on the pod, to copy
   `/workspace/checkpoints/brain/` to your machine.

For CUDA startup failures, check `nvidia-smi` in the pod and verify that the
host driver supports the CUDA version in the Docker runtime image.

## Sparse MoE training

```sh
cargo run --release --no-default-features --bin yumon-pet -- train-brain --architecture moe --batch-size 8 --epochs 15 --out-dir checkpoints/moe --moe-experts 4 --moe-top-k 1
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
