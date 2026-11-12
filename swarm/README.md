# swarm

Yumon Swarm: tens or hundreds of tiny specialist decoders (planners and workers)
writing Rust through skeletons with typed holes, starting from the
`yumon-code-dec-sugar-maniac` checkpoint. The plan, phases, results and log are in
[docs/SWARM.md](../docs/SWARM.md).

Status: phase 0 done (extraction, role configs, coverage). Phase 1 (oracle swarm) is next.

```
cargo test --release                 # extractor tests (byte-exact rebuild, skeleton shape)
cargo run --release --bin make_tokenizer   # tokenizer.json: code BPE + 25 marker tokens
cargo run --release --bin phase0     # stats, coverage, marker costs -> out/phase0/
cargo run --release --bin show_failures -- --kind pat   # fills that fail to parse
```

Own `[workspace]`, path dependency on `yumon-pet` (`default-features = false`), target dir
`../target`. Roles live in `roles/*.json`.
