# Yumon Pet

Yumon is a tabletop ePet that responds to various inputs with short text replies and emotes.

## Datasets

| Dataset | Purpose | Path |
|---------|---------|------|
| CIFAR-100 (binary) | Classification pre-training | `data/cifar-100-binary/` |
| FER2013 | Emote detection | `data/fer2013-archive/` |
| SimpleWiki | Language model pre-training | `data/simplewiki-latest-pages-articles.xml` |
### Build
```bash
cargo build --release
```

### 1. Train Vision CNN
```bash
cargo run --bin yumon-pet --release -- train-vision \
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
cargo run --bin yumon-pet --release -- train-brain \
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
cargo run --bin yumon-pet --release -- chat \
  --vision-checkpoint checkpoints/vision \
  --brain-checkpoint  checkpoints/brain \
  --image path/to/photo.jpg

# Text-only (neutral vision context)
cargo run --bin yumon-pet --release -- chat \
  --vision-checkpoint checkpoints/vision \
  --brain-checkpoint  checkpoints/brain \
  --user-emote happy
```

### 4. Check Status
```bash
cargo run --bin yumon-pet --release -- status
```

