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
- `cargo run --release --no-default-features --bin train_bpe -- [out_dir]` train the tokenizer (default `yumon_bpe/`) on the Language-stage sources, deduped. Lowercased byte-level BPE, 4096 vocab, no prefix space (so text after `,` `"` `:` decodes without extra spaces). A new tokenizer needs a fresh model run.

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
the current session. It is not saved when the app closes. Successful earlier
exchanges are included as natural-language memories using the same `Human:` /
`Yumon:` format as Language training. The latest complete turns are kept in
chronological order, dropping oldest turns to fit the checkpoint's context and
reserve reply space. With no fitting memories, the prompt is just the current
message. Long messages are shortened to fit; errors are excluded from memories.

```sh
cargo run --release --bin yumon_rss -- --checkpoint <language-checkpoint> --architecture moe
cargo run --release --bin yumon_rss -- --check-feeds
```

The default checkpoint is the 256-token MoE Language run at
`D:/models/runpod/256h_32l_4a_256len_b32_Moe_e4_k1_Language_800k`. Use
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


### Prepare samples once, then train on RunPod (Docker)

Build the prepared sample cache locally using the same sources, deduplication,
limits and shuffle as training. The current training grid uses the Language
stage with a **256-token** context. The cache contains the final token IDs,
labels, actions, world contexts, replies and conversation histories in one
compressed binary file; RunPod only decompresses and deserializes it.

```sh
cargo run --release --no-default-features --bin cache_samples -- --max-seq-len 256
cargo run --release --no-default-features --bin cache_samples -- --inspect
```

The output is `training-cache/samples.bin` (ignored by Git, included in Docker).
The default tokenizer is `yumon_bpe/`; use `--tokenizer <directory>` to select
another one and copy that same tokenizer into `yumon_bpe/` before deploying.
The builder refuses to overwrite an existing snapshot. For a rebuild, remove
the old cache explicitly or use `--output <new-file>` and replace the deployed
cache after verifying it. `--stage structured`, `--seed <n>` and `--limit <n>`
are available. Each file stores one stage and context length, matching the
current single-stage training grid. A different stage or context needs its own
snapshot and a matching `YUMON_SAMPLE_CACHE` path.

Chat inputs include earlier turns from the same block as chronological
`Human:` / `Yumon:` dialogue. Oldest complete turns are dropped to fit the
context and preserve the full reply. The cache saves the resulting samples
exactly; it does not rebuild conversation history at training time.

To use a cache for local training:

```powershell
$env:YUMON_SAMPLE_CACHE = "training-cache/samples.bin"
cargo run --release --bin yumon-pet -- train-brain
```

```sh
YUMON_SAMPLE_CACHE=training-cache/samples.bin cargo run --release --no-default-features --bin yumon-pet -- train-brain
```

Without `YUMON_SAMPLE_CACHE`, local training still prepares the raw sources.
With it set, a missing, damaged, or incompatible cache fails explicitly. The
header checks the format version, tokenizer fingerprint, training stage and
context length. Rebuild the cache after changing the tokenizer, source data,
source limits, seed, sample preparation, or context length. Cached generated
worlds/actions stay fixed across training launches; epoch shuffling still runs.

AM-DeepSeek data is optional at **local cache creation** time. If
`data/am_deepseek/am_0.9M.jsonl.zst` exists, it is included with the usual usable-pair
cap (default 300000). Set `YUMON_AM_PATH` and `YUMON_AM_LIMIT` before running
`cache_samples` to choose another local file or cap. Truncated downloads remain
supported. The container no longer downloads this corpus; whatever was included
locally is already in the snapshot.

1. Build and push after the cache has been generated:

   ```sh
   docker build -t alexthegoodman/yumon-brain:latest .
   docker push alexthegoodman/yumon-brain:latest
   ```

   Docker copies `yumon_bpe/` and `training-cache/samples.bin`, and excludes raw
   corpora from the build context. The build fails if the finished cache is absent.
   Compilation uses `--no-default-features`; the runtime uses Burn's CUDA backend.

2. Create a RunPod Network Volume for checkpoints and attach it at `/workspace`.

3. Launch a GPU pod from the image. The default command starts `train-brain`
   with `YUMON_SAMPLE_CACHE=/app/training-cache/samples.bin` and saves checkpoints
   under `/workspace/checkpoints/brain/<run-name>/`. It resumes existing checkpoints.
   It trains the dense decoder-only model by default (`--architecture decoder-only`).
   Decoder-only and MoE honor `--epochs` and `--batch-size`; model dimensions and
   context come from the grid in `src/brain/train.rs`.

4. Download checkpoints from the volume with the RunPod File Manager, `runpodctl`
   or SCP. Verify that the host driver supports the Docker image's CUDA version
   if CUDA startup fails.

Loader/cache checks without a GPU or the repository tokenizer:

```sh
cargo test --lib --no-default-features sample_cache
cargo test --lib --no-default-features loading_tests
```

## Decoder-only training (default)

```sh
cargo run --release --no-default-features --bin yumon-pet -- train-brain --batch-size 32 --epochs 15
```

`YumonDecBrain` (`src/brain/decoder_model.rs`): dense pre-norm decoder, RoPE,
SwiGLU MLP, same data, batches, loss, validation and checkpoint layout as MoE
(`metadata.json` is `DecMetadata`). Attention is the causal FlashAttention op in
`src/brain/flash_attn/` (CubeCL kernels, Burn autodiff backward), so no
[seq, seq] scores are stored. Both training backends are
`Autodiff<CubeBackend<_>, BalancedCheckpointing>` without burn-fusion.
Batches must be right-padded (the attention op is causal only, no key mask).

Measured on a UHD 770 (wgpu), seq 256, batch 32, 8 heads:

| | stored after forward | step |
|---|---|---|
| attention only, naive vs flash, head dim 32 | 225 vs 32 MiB | 159 vs 64 ms |
| attention only, naive vs flash, head dim 64 | 257 vs 64 MiB | 164 vs 287 ms |
| 256w 16L model, no checkpointing vs balanced | 4614 vs 3701 MiB | 37.2 vs 39.6 s |

Balanced checkpointing only recomputes memory-bound ops; matmul outputs and
the [tokens, vocab] logits are still stored. Probe (one variant per process):

```sh
YUMON_PROBE=dec-balanced PROBE_WIDTH=256 PROBE_LAYERS=16 PROBE_BATCH=32 cargo test --release --lib training_memory_probe -- --ignored --nocapture
cargo test --lib decoder_model -- --test-threads=1
```

If a run panics with `matmul_naive ... Cube count too big`, delete
`target/autotune` (the matmul autotune key ignores batch size; Burn's
`RotaryEncoding`, still used by MoE, hits it at batch*heads*seq >= 65536).

## Sparse MoE training

```sh
cargo run --release --no-default-features --bin yumon-pet -- train-brain --architecture moe --batch-size 8 --epochs 15 --out-dir checkpoints/moe --moe-experts 4 --moe-top-k 1
```

This selects `Architecture::Moe` in the existing training grid. It uses the same
stage data, AdamW, charts, periodic text generation, and checkpoint workflow.
The grid dimensions and data sources remain in `src/brain/train.rs`. The default
architecture is decoder-only; `--architecture encoder-decoder` selects the dense
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

The MoE grid is 1024 wide, 24 layers, 8 heads, ff 4096, 512-token context,
4 experts top-1, and default batch size 16 (overridable with `--batch-size`).
With a 4096-token vocabulary this is about 1.32B total and 411M active parameters
per token. FP32 weights, gradients, and Adam states alone need about 21GB;
activations and temporary buffers add to that. Peak VRAM has not been measured
for this configuration.

### Synthetic puzzles, commands, and memory

`gen_synthetic_data` asks Llama through the existing Ollama `/api/chat` endpoint
to generate actual puzzles and answers, command-following examples, and related
multi-turn memory conversations. Puzzles cover location tracking, ordering,
counting, spatial relations, constraints, and invented rules. Commands cover
exact output formats, sorting, conditionals, action plans, clarification, and
blocked actions. Llama is instructed to check answers privately; local validation
checks structure and text, not semantic correctness. Review answers before use.

Preview the prompts without contacting the endpoint or writing files:

```sh
cargo run --no-default-features --bin gen_synthetic_data -- --preview
```

Generate a small review batch (12 pairs per topic; memory keeps complete
conversations and may exceed the target by up to four pairs):

```sh
cargo run --release --no-default-features --bin gen_synthetic_data -- --topics puzzles,commands,memory --target 12 --seed 42
```

Use `--endpoint` or `OLLAMA_ENDPOINT` to override the default endpoint, and
`--model` to select the served Llama model (default `llama3`). Output stays in
`data/synthetic/{puzzles,commands,memory}.txt`; generation never moves it to
`archive`. Independent pairs have separate blank-line-delimited chat blocks;
memory conversations retain their related turns in one block. Progress remains
under `data/synthetic/.state`. `--pairs-per-call` counts conversations for memory
and independent pairs for the other topics.

After review, move approved files to `archive/synthetic/`. Training and tokenizer
training automatically include those three archived files when present, and
never ingest their staged copies under `data/synthetic`. Rebuild the Docker image
after approval to include them. Shape changes require a fresh model run; retain
the current tokenizer unless you intend to retrain it and start fresh.

MoE language loss is the mean over supervised (non-padding) tokens. Burn 0.20's
`CrossEntropyLoss` masks padding but divides by every position, so runs before
2026-10-06 logged a loss scaled down by the supervised fraction. Those numbers
are not comparable with newer runs. 1% of samples (at most 2048) are held out.
Validation loss is logged with each 500-batch inference snapshot and at epoch
end, saved as `val_loss` in `metadata.json`, and used for the early stop.

The data loader removes exact duplicate samples (same prompt and target
tokens) across all sources. Every sample has a prompt. `wiki_extract.txt` (one
paragraph per line, cleaned from the simplewiki XML) is split into sentences
that become alternating Human/Yumon turns, one conversation per paragraph, with
earlier turns as memories when they fit. Article openings (`'April' (Apr.) is
...`) start with a templated question about the title. `quotes.csv` becomes a
request built from the quote's first category tag, answered with the quote.
distillchat is loaded in full. Per-source counts and decoded wiki/quote samples,
without loading everything at once:

```sh
cargo test --release --no-default-features --lib language_data_ -- --ignored --nocapture --test-threads=1
```

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
