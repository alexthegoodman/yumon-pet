# Generate doc comments for Yumon Code

`gen_synthetic_captions` cycles through the four prepared Rust buckets:
one low sample, one mild sample, one moderate sample, then one high sample,
repeating until every sample has doc comments. Finished buckets are skipped.
It makes one request per sample to a dedicated
Hugging Face Inference Endpoint using the
[chat completions API](https://huggingface.co/docs/hugs/guides/inference).
The endpoint must already serve your chosen Qwen model. Pass its exact served
model ID; the tool does not deploy or change the endpoint's model.

Supply `--endpoint`, `--model`, and `--token` as command-line arguments;
environment variables are not required or read. `--api-key` is an alias for
`--token`. The token is never written to generated output files.
Keep credentials out of tracked files. The endpoint must use
HTTPS. A base URL, a URL ending in `/v1`, or the complete
`/v1/chat/completions` URL is accepted.

Run from the repository root, replacing these placeholders:

```powershell
cargo run --release --no-default-features --bin gen_synthetic_captions -- --endpoint "https://YOUR-ENDPOINT.endpoints.huggingface.cloud" --model "YOUR-SERVED-MODEL-ID" --token "YOUR-HF-TOKEN"
```

Defaults:

- Input: `training-cache/code_buckets`, as produced by `sort_code_buckets`.
- Output: `training-cache/code_buckets_documented`.
- Tokenizer: `yumon_code_bpe` (must match the input caches).
- Output context: 1024 tokens; initial model response limit: 512 tokens,
  doubling on truncation up to 8192 tokens.
- Temperature: 0.2; request timeout: 300 seconds; no delay between samples.

Use `--input`, `--output`, `--tokenizer`, `--max-seq-len`, `--max-tokens`,
`--retry-max-tokens`, `--temperature`, `--timeout`, or `--delay` to override
these values. `--retry-max-tokens` must be at least `--max-tokens`.
`--help` lists all options. Each sample logs its bucket and index before the
request, then confirms the synced result with total progress, percentage, and
elapsed seconds. The complete documented sample is still printed every 50
samples, along with bucket completion messages.

The model returns only `///` doc comments. The tool inserts them immediately
before the function, struct, or wrapped method's attributes, preserving all
original source bytes, existing comments, source paths, locations, and order.
New comments are requested as one to three short sentences, capped at 80 words
and six lines. Oversized comments are not saved: the model is immediately asked
to rewrite them more concisely, within the same five-attempt limit. The response
token budget is headroom for completing a request, not a desired comment length.
Existing saved comments are reused without applying the new length limits.
The request explicitly disables thinking using Qwen's
[documented API setting](https://huggingface.co/Qwen/Qwen3.8-27B#instruct-or-non-thinking-mode).
A completed leading `<think>...</think>` block is removed if the server still
includes one; only the final doc comments are saved. Responses containing code,
fences, unfinished reasoning, invalid comments, or truncated completions are
rejected. A `finish_reason=length` response retries immediately with twice the
token budget, bounded by `--retry-max-tokens` and five total attempts. Other
retryable HTTP failures and invalid responses use backoff. Retry logs include
the attempt number, token budget, and finish reason when available.
Authentication and request configuration
errors stop immediately. Generated documentation still needs human review
for factual accuracy.

Each bucket has an append-only `.jsonl` comment journal. Every successful
generation is synced before processing the next sample. Rerun the same command
after interruption to reuse saved comments and skip completed buckets. The
next request goes to the least advanced unfinished bucket, with tier order
breaking ties, so an interrupted round continues at the appropriate bucket.
Older sequential runs are supported: lagging buckets catch up first.
The manifest checks input hashes, tokenizer, model, prompt, and generation settings
before resuming. Change those settings by choosing a new output directory.
Run only one generator per output directory at a time.

Existing journals from before the truncation fix resume with the same command.
Keep the original `--max-tokens` value (512 by default) to match the manifest;
the new retry behavior and `--retry-max-tokens` do not invalidate saved comments.

Code and comments are never truncated or silently dropped. If a documented
sample exceeds the output context, its comment stays in the journal. Increase
`--max-seq-len` and rerun. If earlier buckets are already complete, their context
must also match; use a new output directory and copy `generation.json` plus
the four `.jsonl` files there to reuse comments with a larger context.

Completed buckets are published as raw-code `.bin` caches with BOS/code inputs
and code/EOS labels. Empty input tiers are skipped only when the input sorter's
report confirms a zero count. Excluded, unscored, and failed buckets are not
processed. After all four tiers finish, `reference.json` and a new `report.json`
make the result usable by the existing curriculum loader. Tier assignments
remain those of the original code; the documented samples are not rescored.

To train on these buckets, set `curriculum` in your code configuration to
`training-cache/code_buckets_documented` and set `max_seq_len` to the generator's
output context. Check the configuration before training:

```powershell
cargo run --no-default-features --bin train_code -- --config configs/yumon-code.json --check
```

The existing curriculum still trains on low/mild/moderate only. The documented
high bucket is available for inspection but remains excluded from that schedule.
