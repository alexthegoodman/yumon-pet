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

2. Build the Docker image:

   ```sh
   docker build -t alexthegoodman/yumon-code:latest .
   ```

3. Push the image:

   ```sh
   docker push alexthegoodman/yumon-code:latest
   ```

Docker automatically copies the new bundle from
`training-cache/code_tokenizer_retrained` and validates it during the build.
Launch the image on RunPod with persistent storage at `/workspace`; training
starts automatically.

Configure tokenizer and model settings in
[configs/yumon-code.json](configs/yumon-code.json). The curriculum uses low in
epoch 1, adds mild in epoch 2, and adds moderate from epoch 3 onward. Each tier
has equal documented and distinct undocumented samples. Tokenizer fitting
excludes validation files. Existing artifacts are preserved; use a new output
directory for subsequent tokenizer runs.

See the [command reference](docs/commands.md) for first-time data preparation,
validation, alternate configs, Pet apps and smoke tests. Details:
[comment generation](docs/synthetic-code-comments.md),
[complexity scoring](docs/code-complexity.md),
[RunPod sizing](configs/runpod-h100.md).
