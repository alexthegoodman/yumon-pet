# Squared-ReLU MLP

Config key `mlp_kind` for the dense decoder (`architecture: "decoder-only"`). `relu2` replaces each
block's SwiGLU with the squared ReLU from So et al. 2021, "Primer: Searching for Efficient
Transformers for Language Modeling". Implemented as `Relu2Mlp` in `src/brain/model.rs` and
`MlpKind` in `src/brain/decoder_model.rs`.

```json
"mlp_kind": "relu2"
```

| value | MLP | weights per block |
|---|---|---|
| `swiglu` (default) | w2(silu(w1 x) * w3 x) | 2 x in x ff + ff x width |
| `relu2` | w2(relu(w1 x)^2) | in x ff + ff x width |

`in` is the MLP input width: the model width, or the `ffn` JL map's K. Both projections are bias-free
with N(0, 0.02) init, like SwiGLU's. At the same `ff_dim`, relu2 saves `in x ff` per block. To match
SwiGLU's weights instead, use `ff_dim` = ff x (2 in + width) / (in + width).

## White-dwarf shape

128 wide, 6 layers, ff 512, `ffn` JL at K = 32:

- SwiGLU MLP: 98,304 weights per block. relu2: 81,920 (-16,384). Whole model 5.89M -> 5.79M.
- The JL map already shrinks w1/w3 to 32 x 512, so w2 (512 x 128) is most of the MLP. The saving
  is small here; it grows without the `ffn` map (w1 and w3 are then 128 x 512 each).
- Same weights as SwiGLU: `ff_dim` 614.

## Use

- `train_code --mlp swiglu|relu2` overrides the config's arm; it combines with `--gate`.
- Run names gain `_relu2`; resuming checks the arm matches. A relu2 arm seeds the backend like the
  `jl` / `ngram` / gate arms.
- Checkpoints: `relu2_mlps` is the last record field and `mlp_kind` a metadata key (absent =
  swiglu). Each block's own `mlp` is then an unused width-1 SwiGLU placeholder (like the tied
  1 x 1 `token_head`), so the block record layout and every older checkpoint stay readable. Older
  records load through `GatedDecBrain` and the earlier layouts.
- `wir_decoder` / `mlp_weights` read SwiGLU weights only and refuse relu2 models.
- Config `configs/yumon-dec-base-pretrain-jl-ffn64-ngram-relu2.json` is the white-dwarf config
  with `relu2` and out_dir `checkpoints/yumon-dec-ngram-relu2`.

## Status

Tests in decoder_model.rs: relu2 forward = w2(relu(w1 x)^2), param counts match the module, cached
logits = full forward (also with residual + ffn JL, gates, N-Grammer and a factorized tied table),
roundtrips, BalancedCheckpointing gradients = plain, and pre-relu2 gated records load. Smoke
training (16 samples, resume) passed. Real A/B runs are not done.
