# Yumon

Yumon Pet is a conversational desktop companion. Yumon Code trains on Rust code
with and without doc comments.

## Train and deploy Yumon Code

Run these from the repository root with the original buckets, tokenizer and
saved comment journals already prepared.

Check `docs/commands.md` for more details.

1. Train the tokenizer and prepare matching training caches:

   ```sh
   cargo run --release --no-default-features --bin train_bpe
   ```

2. Build the Rust books pretraining stage with the new tokenizer:

   ```sh
   cargo run --release --no-default-features --bin cache_samples -- --code-config training-cache/code_tokenizer_retrained/config.json --rust-docs
   ```

3. Build the Docker image:

   ```sh
   docker build -t alexthegoodman/yumon-code:latest .
   ```

4. Push the image:

   ```sh
   docker push alexthegoodman/yumon-code:latest
   ```

Docker automatically copies the new bundle from
`training-cache/code_tokenizer_retrained` and validates it during the build.
Launch the image on RunPod with persistent storage at `/workspace`; training
starts automatically.

To train locally with the same prepared bundle and regimen as RunPod, run this
from the repository root after step 1:

```sh
cargo run --release --no-default-features --bin train_code -- --config training-cache/code_tokenizer_retrained/config.json --require-curriculum
```

The bundle's `config.json` uses local tokenizer/cache paths and the same model,
training settings and balanced documented/undocumented curriculum as Docker's
`docker-config.json`. Set `out_dir` in the bundle's `config.json` to your local
checkpoint directory before launching; it inherits the preparation config's
output root. 

Optional `cuda-training` selects RunPod's CUDA/FP32 backend on Windows too;
omit it to use WGPU on Windows. Append `--check` to validate the bundle without
starting training.

Configure tokenizer and model settings in
[configs/yumon-code.json](configs/yumon-code.json). With `docs` set, epoch 1
trains only on the official Rust books (see
[Rust books stage](docs/commands.md#code-rust-books-pretraining-stage); rebuild
it after step 1). The code curriculum then uses low, adds mild, and adds
moderate from its third epoch onward. Each tier
has equal documented and distinct undocumented samples. Tokenizer fitting
excludes validation files. Existing artifacts are preserved; use a new output
directory for subsequent tokenizer runs.

See the [command reference](docs/commands.md) for first-time data preparation,
validation, alternate configs, Pet apps and smoke tests. Details:
[comment generation](docs/synthetic-code-comments.md),
[complexity scoring](docs/code-complexity.md),
[RunPod sizing](configs/runpod-h100.md).
