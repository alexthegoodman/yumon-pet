# 🐾 Yumon ePet

A tabletop AI companion that observes the world through a camera, detects what it sees and how the user feels, then responds with Wikipedia-style factual sentences and a matching emotional expression.

---

## Architecture

```
                    ┌─────────────────────────────────────┐
  [Image Input]     │         Vision CNN                  │
  [3×32×32]    ──►  │  Conv×3 → BN → ReLU → MaxPool      │
                    │  → Dense(512) → Dense(256)          │
                    │  ├─ classification_head → [100]     │  class_probs
                    │  └─ emote_head          → [7]       │  user_emote_probs
                    └────────────────┬────────────────────┘
                                     │ 107-dim context vector
                                     ▼
                    ┌─────────────────────────────────────┐
  [Text Seed]  ──►  │         LSTM Brain                  │
                    │  Embed(char) ++ Context             │
                    │  → LSTM(512) → Dropout → Dense(256) │
                    │  ├─ token_head  → [vocab_size]      │  reply text
                    │  └─ emote_head → [7]               │  Yumon's emote
                    └─────────────────────────────────────┘
```

### Vision CNN
- **Input**: 32×32 RGB image
- **Backbone**: 3× Conv2D blocks with BatchNorm, ReLU, MaxPool
- **classification_head**: Dense(100) — CIFAR-100 fine-grained categories
- **emote_head**: Dense(7) — FER2013 emotion detection (angry/disgust/fear/happy/neutral/sad/surprise)
- **Training**: Joint dual-head loss, CIFAR-100 + FER2013 simultaneously

### LSTM Brain
- **Input per timestep**: character embedding (64-dim) concatenated with vision context (114-dim)
- **Context vector**: `[class_probs: 100] ++ [user_emote_probs: 7] ++ [user_emote_onehot: 7]`
- **Backbone**: LSTM(512) → Dropout(0.3) → Dense(256, ReLU)
- **token_head**: Dense(vocab_size) — autoregressive character-level language model
- **yumon_emote_head**: Dense(7) — Yumon's own emotional response (mirrors user emotion)
- **Training**: Next-character prediction on SimpleWiki, random vision context injection

### Emote mirroring
Yumon mirrors the user's detected emotion. The emote head is trained with a cross-entropy loss against the user's FER2013 class label, so at inference Yumon outputs the same emotional state it perceives.

---

## Datasets

| Dataset | Purpose | Path |
|---------|---------|------|
| CIFAR-100 (binary) | Classification pre-training | `data/cifar-100-binary/` |
| FER2013 | Emote detection | `data/fer2013-archive/` |
| SimpleWiki | Language model pre-training | `data/simplewiki-latest-pages-articles.xml` |

---

## Setup

### Prerequisites
- Rust (stable) — install via [rustup.rs](https://rustup.rs)

### Build
```bash
cargo build --release
```

---

## Usage

### 1. Train Vision CNN
```bash
cargo run --release -- train-vision \
  --cifar-dir data/cifar-100-binary \
  --fer-dir   data/fer2013-archive \
  --out-dir   checkpoints/vision \
  --epochs    30 \
  --batch-size 64
```

Expected training time: ~2–4 hours on CPU, ~20 minutes with GPU backend.  
Watch for **emote val acc** — aim for >50% before proceeding (FER2013 is challenging; human performance ~65%).

### 2. Train LSTM Brain
```bash
cargo run --release -- train-brain \
  --wiki-xml  data/simplewiki-latest-pages-articles.xml \
  --vision-checkpoint checkpoints/vision \
  --out-dir   checkpoints/brain \
  --epochs    10 \
  --batch-size 32 \
  --max-articles 50000
```

Expected training time: ~1–3 hours on CPU.  
Watch for **loss** descending below ~1.5 for coherent character-level generation.

### 3. Chat (Inference)
```bash
# With an image
cargo run --release -- chat \
  --vision-checkpoint checkpoints/vision \
  --brain-checkpoint  checkpoints/brain \
  --image path/to/photo.jpg

# Text-only (neutral vision context)
cargo run --release -- chat \
  --vision-checkpoint checkpoints/vision \
  --brain-checkpoint  checkpoints/brain \
  --user-emote happy
```

### 4. Check Status
```bash
cargo run --release -- status
```

---

## Training Curriculum

**Stage 1 — Vision (parallel dual-head)**
- CIFAR-100 loss trains the backbone to recognise objects
- FER2013 loss trains the emote head to detect facial emotion
- Both losses flow through the shared backbone simultaneously

**Stage 2 — Language (conditioned pre-training)**
- LSTM learns character-level Wikipedia-style language
- Random Dirichlet-sampled vision context is injected at every timestep
- Emote head learns from heuristic keyword labels (angry/sad/happy etc.)
- This establishes the conditioning interface before real image–text pairs exist

**Stage 3 — Fine-tuning (future)**
- Collect (image, desired_reply, emote) triplets
- Run with `TrainingMode::Reinforce` to refine with reward signals

---

## File Structure

```
yumon/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs                  # CLI entry point
│   ├── vision/
│   │   ├── mod.rs               # Constants & module exports
│   │   ├── model.rs             # Dual-head CNN definition
│   │   ├── cifar.rs             # CIFAR-100 binary loader
│   │   ├── fer.rs               # FER2013 JPEG loader
│   │   ├── loader.rs            # Single-image inference loader
│   │   └── train.rs             # Joint dual-head training loop
│   └── brain/
│       ├── mod.rs               # Constants & module exports
│       ├── model.rs             # LSTM brain + generation
│       ├── tokenizer.rs         # Character-level tokenizer
│       ├── wiki.rs              # SimpleWiki XML parser
│       └── train.rs             # Language model training loop
├── checkpoints/
│   ├── vision/                  # Saved vision weights + metadata
│   └── brain/                   # Saved brain weights + tokenizer
└── data/
    ├── cifar-100-binary/
    ├── fer2013-archive/
    └── simplewiki-latest-pages-articles.xml
```

---

## Switching to GPU

To use the Burn WGPU backend (Metal/Vulkan/DX12), change `Cargo.toml`:

```toml
burn = { version = "0.16", features = ["wgpu", "autodiff", "train"] }
```

Then update `src/vision/train.rs` and `src/brain/train.rs`:
```rust
pub type TrainBackend = burn::backend::Autodiff<burn::backend::Wgpu>;
```

---

## Design Notes

### Why character-level tokenization?
- Vocabulary is tiny (~200 chars) — fast softmax, small embedding table
- No OOV issues; any text can be encoded
- Works well at tweet-length outputs (≤140 tokens)
- Downside: slower generation per token vs. word/BPE; fine for tabletop pace

### Why random context during pre-training?
- We don't have paired (image → wiki sentence) data
- Random context teaches the LSTM that the context *can* vary, preventing it from ignoring it
- The conditioning interface is ready for fine-tuning with real image–text pairs

### Emote mirroring
The `yumon_emote_head` is trained to output the *same* emote class as the user's detected emotion, making Yumon empathetic by design. This can be inverted post-training by negating the head weights for a contrarian personality variant.