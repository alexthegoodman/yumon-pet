# Generate doc comments for Yumon Code

`gen_synthetic_captions` reads `training-cache/code.bin`, assigns each chunk a
tier with the frozen `training-cache/code-reference-v1.json`, and cycles through
the low, mild and moderate tiers (the least advanced tier next). It makes one
request per sample to a dedicated Hugging Face Inference Endpoint using the
[chat completions API](https://huggingface.co/docs/hugs/guides/inference). The
endpoint must already serve your chosen Qwen model; pass its exact served model ID.

```powershell
cargo run --release --no-default-features --bin gen_synthetic_captions -- --endpoint "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" --model "YOUR-SERVED-MODEL-ID" --token "YOUR-HF-TOKEN"
```

`--endpoint`, `--model` and `--token` are required arguments (no environment
variables; the token is never written to disk). The endpoint must use HTTPS. A
base URL, a URL ending in `/v1`, or the full `/v1/chat/completions` URL works.

## Output

Each finished sample appends and syncs one record to each of two files in the data
directory (`curriculum` in `--config`, default `configs/yumon-code.json` ->
`training-cache`):

- `documented/<tier>.bin`: code taken from the front of the tier in `code.bin`, with
  the generated `///` comment inserted before the item's attributes (source bytes
  otherwise unchanged).
- `undocumented/<tier>.bin`: a different sample taken from the back of the tier,
  without doc comments, so the two counts always match. Captioning never reaches the
  back half unless you caption more than half of a tier; it stops where they meet.

Records hold the source file, location and text; there are no hashes, indices or
manifests. Stop and rerun any time: code already stored in either folder is
skipped, an interrupted trailing record is dropped, and a stop between the two
appends is repaired on start (a plain side that is short is topped up from the back). Run only one generator at a time. Running it again
after adding code to `rust-code` (and rebuilding `code.bin`) captions just the new
code. Samples the model could not document are skipped for that run and retried on
the next.

## Prompt and limits

Comments are one to three short sentences, at most 80 words and six lines. Oversized
comments are rewritten shorter within five attempts. The request disables thinking
([Qwen setting](https://huggingface.co/Qwen/Qwen3.8-27B#instruct-or-non-thinking-mode));
a leading `<think>...</think>` block is removed if the server still includes one.
Responses containing code, fences, unfinished reasoning, invalid comments or
truncated completions are rejected; `finish_reason=length` retries with twice the
token budget up to `--retry-max-tokens`. Other retryable failures back off;
authentication and configuration errors stop immediately. Defaults: temperature
0.2, `--max-tokens` 512, request timeout 300 s. Documented samples needing more than
`--max-seq-len` (default 1024) tokens are not stored. Generated comments still need
human review for accuracy.

## Training

Set `curriculum` to `training-cache` and `max_seq_len` to the context you train with;
samples that do not fit are skipped at load. Check before training:

```powershell
cargo run --no-default-features --bin train_code -- --config configs/yumon-code.json --check
```
