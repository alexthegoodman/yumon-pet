# swarm

Yumon Swarm: tens or hundreds of tiny specialist decoders (planners and workers)
writing Rust through skeletons with typed holes, starting from the
`yumon-code-dec-sugar-maniac` checkpoint. The plan, phases, results and log are in
[docs/SWARM.md](../docs/SWARM.md).

Status: phases 0 and 1 done (extraction, role configs, coverage, oracle swarm). Phase 2 (workers) is ready to train: nothing has been trained yet.

```
cargo test --release                 # extractor tests (byte-exact rebuild, skeleton shape)
cargo run --release --bin make_tokenizer   # tokenizer.json: code BPE + 25 marker tokens
cargo run --release --bin phase0     # stats, coverage, marker costs -> out/phase0/
cargo run --release --bin show_failures -- --kind pat   # fills that fail to parse
```

Own `[workspace]`, path dependency on `yumon-pet` (`default-features = false`), target dir
`../target`. Roles live in `roles/*.json`.
cargo run --release --bin oracle -- --check 1500   # phase 1: oracle swarm, cargo check, per-kind swaps (about 1 min)

Phase 2, one separate small decoder per role (see docs/SWARM.md "Phase 2 setup"):

```
cargo run --release --bin train_role -- --role block --check          # data counts + model size, no training
cargo run --release --bin train_role -- --role block                   # train -> ../checkpoints/swarm/block
cargo run --release --bin eval_role  -- --role block                   # greedy exact/parse + swap into the oracle swarm
.	rain_roles.ps1 -Roles block,arms -Eval                              # several roles, one after another
```

`roles/kinds.json` holds each role's `out_vocab` (output rows), optional `model` (own shape, or
another `checkpoint`) and `train` overrides (epochs, batch, first_lr, last_lr, max_samples).
Progress prints to stdout and appends to `train.csv` / `val.csv` in the role's directory.

