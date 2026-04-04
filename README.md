# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

## Hyperparameters

### Medium

```
pub const EMBED_DIM:    usize = 256;
pub const HIDDEN_UNITS: usize = 256;
pub const ATTN_HEADS:   usize = 2;
pub const N_LAYERS:     usize = 2;
pub const FF_DIM:       usize = 1024;
pub const TEMPERATURE:  f32   = 0.9;
pub const TOP_K:        usize = 10;
pub const MAX_SEQ_LEN:  usize = 280;
```

### Small

```
pub const EMBED_DIM:    usize = 64;
pub const HIDDEN_UNITS: usize = 64;
pub const ATTN_HEADS:   usize = 2;
pub const N_LAYERS:     usize = 2;
pub const FF_DIM:       usize = 256;
pub const TEMPERATURE:  f32   = 0.9;
pub const TOP_K:        usize = 10;
pub const MAX_SEQ_LEN:  usize = 320;
```

## Get Started

- `cargo run --release --bin yumon-pet -- train-brain` to start training on the provided (or your own) dataset
- `cargo run --release --bin chat_ui` to get started chatting
- `cargo run --release --bin yumon_world` to start a Yumon World simulation
- `cargo run --release --bin endless_data` TUI to answer endless questions in order to generate some data
- `cargo run --release --bin train_bpe` train your tokenizer on your data

### Datasets

- Custom / Bespoke
- https://www.kaggle.com/datasets/lmsysorg/chatbot-arena-conversations