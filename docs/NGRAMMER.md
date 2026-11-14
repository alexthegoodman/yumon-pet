# N-Grammer layers

Config section `ngram` for the dense decoder (`architecture: "decoder-only"`), implemented in
`src/brain/ngrammer.rs` after [lucidrains/n-grammer-pytorch](https://github.com/lucidrains/n-grammer-pytorch)
and Roy et al. 2022. It is separate from `jl` (JL maps shrink widths; this adds information).

```json
"ngram": { "layers": [0, 3], "num_clusters": 64, "heads": 4, "loss_weight": 0.25, "seed": 1 }
```

| field | default | meaning |
|---|---|---|
| `layers` | `[0]` | blocks each layer sits before (0 = on the token embeddings, after any residual JL map) |
| `num_clusters` | 64 | centroids per head, 2..=256 |
| `heads` | 4 | product-quantization heads; width at the layer must divide by it |
| `loss_weight` | 0.25 | weight of the centroid loss |
| `seed` | 1 | initial centroids; layer i uses seed + i |

## Layer

1. x [b, s, w] is split into `heads` sub-vectors, rms-normalised (detached) and snapped to the
   nearest centroid, giving ids [b, s, heads].
2. Bigram id = (previous id, id), previous = a BOS state at position 0. Row `(h*(K+1) + prev)*K + id`
   of a per-head table of width w / heads.
3. The embeddings are rms-normed, projected w -> w and added to x. The projection starts at zero:
   a fresh layer is a no-op, so arms start from the same function as the baseline.

Differences from upstream: no EMA k-means. Centroids are parameters trained by
`loss_weight * mean |centroid - h|^2` (h detached), returned as the auxiliary loss
(`YumonDecBrain::forward_aux`, picked up by `CausalLmTrain::forward_train`). The bigram table is
exact (K * (K+1) rows per head) rather than hashed, and the output is added instead of concatenated,
so widths do not change. Dead centroids are not re-seeded.

## Use

- `train_code --ngram 64x4@0` / `--ngram none` overrides the config's arm (keeps its loss_weight, seed).
- Run names gain `_ngram64x4_at0-3`; resuming checks the config matches.
- Cached generation keeps each layer's last ids in the `KvCache`; tests check cached logits equal the full forward.
- Checkpoints: `ngram` is the last record field and a metadata key, so older checkpoints load (layouts `ProjDecBrain` / `JlDecBrain` / `LegacyDecBrain`).
- Weights: `YumonDecBrainConfig::ngram_param_count()`, included in `param_count()`.
- Configs: `configs/yumon-dec-base-pretrain-jl-ffn64-ngram{,-multi,-k128}.json` (from the jl-ffn64 config) and `yumon-dec-base-pretrain-ngram.json` (no JL).

Not run: any real training; only unit tests and `--check`.
