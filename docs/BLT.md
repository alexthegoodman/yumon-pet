# Byte Latent Transformer

Config section `blt` for `train_code` (`src/brain/blt.rs`, after Pagnoni et al. 2024 and
facebookresearch/blt). It trains on the same BPE cache as the token decoder: each batch row is
decoded to bytes (`TokenBytes`, exact for byte-level BPE), and a byte is supervised when its
token was. Configs are the N-Grammer baseline (`yumon-dec-base-pretrain-jl-ffn64-ngram.json`)
minus `jl` / `ngram` / `token_embed_dim`, so `nats_per_token` compares directly with its val_loss
(3.6635 nats/token, `checkpoints/yumon-dec-ngram-white-dwarf`).

```sh
cargo run --release --bin train_code -- --config configs/yumon-blt-space.json [--check]
```

| config | what |
|---|---|
| `yumon-blt-space.json` | BLT, a patch after each word/space run (cap 16) |
| `yumon-blt-static4.json` | BLT, 4-byte patches |
| `yumon-bytelm-entropy.json` | plain byte decoder (64 wide, 2 layers): the entropy model, also an unpatched baseline |
| `yumon-blt-entropy.json` | BLT, a patch where the byte LM's entropy > 2.0 nats (cap 16); train the byte LM first |

| field | default | meaning |
|---|---|---|
| `model` | `blt` | `blt` or `byte-lm` |
| `max_bytes` | 1024 | byte context; rows are cut at a token boundary (1536 drops no tokens on massive-2) |
| `patching` | space, 16 | `{"mode": "static", "size"}`, `{"mode": "space", "max_len"}`, `{"mode": "entropy", "model", "threshold", "max_len"}` |
| `local_dim` / `local_heads` / `local_ff` | 128 / 4 / 512 | byte encoder and decoder width |
| `encoder_layers` / `decoder_layers` | 1 / 2 | causal byte blocks |
| `hash_ngrams` / `hash_buckets` | 3..8 / 4096 | hashed n-gram embedding tables (FNV-1a) added to the byte embedding |
| `cross_k` | 2 | vectors per patch the decoder cross-attention reads |
| `eval_bytes` | 160 | bytes generated per eval prompt |

The global transformer takes `embed_dim`, `n_layers`, `attn_heads` and `ff_dim` from the config.

## Model

1. Byte embedding plus the hashed n-gram embeddings, scaled by 1/sqrt(tables).
2. Encoder: causal byte blocks, then each patch (query = mean of its bytes) cross-attends to its own bytes.
3. Global: causal blocks over patches. Patch 0 is BOS alone.
4. Decoder: per layer, each byte attends over `cross_k` vectors from the global output of the
   last patch closed at or before it (`id(p + 1) - 1`), then a causal byte block; 260-way head.

All patching modes are causal: whether byte p starts a patch depends only on bytes before p, so
nothing leaks (tests check logits against later bytes).

## Training

- Metrics: bpb, nats per byte, nats per BPE token, top-1 over bytes, bytes per patch.
- Every step: the decoder trainer's dashboard on a terminal (losses in nats per byte), else a
  JSON line every 50 steps to stdout and `blt.jsonl`.
- Every 500 steps: `model.bin`, `optimizer.bin`, `progress.json` and `blt_chart.png`. A rerun
  resumes mid-epoch at the saved batch with the AdamW state.
- Every 2,000 steps and at each epoch end: validation and the eval prompts (`inference_log.txt`).
- Run names: `blt_space16_g128h_6l_4a_512ff_l128x512e1d2_hash345678x4096_k2_1536b_b4`, `bytelm_..._tied`.

massive-2 at the space config: 3.77 bytes per token, about 3.2 bytes per patch (static4: 3.96),
5.85M parameters (3.15M in hash tables), ~400 ms/step at batch 4 on the UHD 770 (~43 h/epoch);
bpb fell 5.09 -> 2.76 over the first 700 steps. Full runs are not done.

## Generation

`ByteModel::decoder` gives a KV-cached decoder (`BltDecoder`, `ByteLmDecoder`): feed the prompt,
then one byte per step. Positions only read closed patches and patch starts are causal, so each
byte runs the encoder and decoder blocks once against their byte caches, and each patch runs the
encoder cross-attention and global blocks once, when its last byte arrives. Entropy patching steps
the byte LM's own `KvCache`. A test checks cached logits against the full forward for every mode.

Untrained space-config model, greedy, UHD 770 (`kv_cache_speed`, ignored test), two runs:

| prompt + new bytes | full rerun ms/byte | cached ms/byte |
|---|---|---|
| 40 + 160 | 61-123 | 11-15 |
| 1200 + 160 | 164-267 | 12-14 |

Worst logit difference along those rows: 4.8e-7.

## Known issue

Burn 0.20 / CubeCL 0.9 on WGPU: the matmul input gradient dX = dY W^T is wrong (error ~3) for
some small row counts (e.g. 8 x 128 x 260, 40 x 32 x 260); 400+ rows were exact, forward and dW
too. Real batches are well above that. Repro: `wgpu_matmul_input_gradient` (ignored test).
