#!/usr/bin/env python3
"""Rebuild the Code caches/buckets after changing ../../rust-code, keeping old comments.

  python scripts/refresh_code_data.py prepare    # cache -> sort -> verify old samples survive -> new generation.json
  python scripts/refresh_code_data.py finalize   # train_bpe -> train_code --check -> rust-docs

Run from the repository root. Nothing existing is modified or overwritten; each
stage skips work whose output already exists, so it is safe to re-run. Commands
run one at a time. The cache is shuffled, so sample order is not preserved; old comments
are carried over by hash via `gen_synthetic_captions --reuse`.
"""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TC = ROOT / "training-cache"
LOCAL_V2 = ROOT / "configs/yumon-code-local-v2.json"
PROD_V2 = ROOT / "configs/yumon-code-v2.json"
REFERENCE = TC / "code-reference-v1.json"
OLD_BUCKETS, NEW_BUCKETS = TC / "code_buckets", TC / "code_buckets_v2"
OLD_DOC, NEW_DOC = TC / "code_buckets_documented", TC / "code_buckets_documented_v2"
NEW_CACHE = TC / "code_v2.bin"
# Directory names (matched anywhere in the tree) holding intentionally unparseable Rust.
EXTRA_EXCLUDES = ["highlighting-tests"]
TIERS = {1: "tier_1_low", 2: "tier_2_mild", 3: "tier_3_moderate", 4: "tier_4_high"}


def die(msg):
    print(f"\nSTOP: {msg}", file=sys.stderr)
    sys.exit(1)


def cargo(*args):
    cmd = ["cargo", "run", "--release", "--no-default-features", "--bin", *args]
    print("\n>>", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        die(f"command failed: {' '.join(cmd)}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def tier_hashes(records_path):
    """Per-tier ordered (source_sha256, source_path) for successfully scored samples."""
    out = {t: [] for t in TIERS}
    with open(records_path, encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["analysis_status"] == "success":
                out[r["complexity_tier"]].append((r["source_sha256"], r["source_path"]))
    return out


def prepare():
    if not LOCAL_V2.exists():
        cfg = json.loads((ROOT / "configs/yumon-code-local.json").read_text(encoding="utf8"))
        cfg["cache"] = "training-cache/code_v2.bin"
        cfg["curriculum"] = "training-cache/code_buckets_v2"
        cfg["exclude_dirs"] += [d for d in EXTRA_EXCLUDES if d not in cfg["exclude_dirs"]]
        write_json(LOCAL_V2, cfg)
        print(f"wrote {LOCAL_V2.relative_to(ROOT)}")

    if NEW_CACHE.exists():
        print(f"skip cache_samples: {NEW_CACHE.name} exists")
    else:
        cargo("cache_samples", "--", "--code-config", str(LOCAL_V2))

    if NEW_BUCKETS.exists():
        print(f"skip sort_code_buckets: {NEW_BUCKETS.name} exists")
    else:
        cargo("sort_code_buckets", "--", "--code-config", str(LOCAL_V2),
              "--reference", str(REFERENCE), "--output", str(NEW_BUCKETS))

    # The cache is a seeded shuffle, so old samples are NOT a prefix of the new tiers.
    # Journals are positional, so they cannot be copied; instead every old sample must
    # still exist (same hash, same tier) so gen_synthetic_captions --reuse can match by hash.
    old, new = tier_hashes(OLD_BUCKETS / "records.jsonl"), tier_hashes(NEW_BUCKETS / "records.jsonl")
    for t, name in TIERS.items():
        o, n = {h for h, _ in old[t]}, {h for h, _ in new[t]}
        if o - n:
            die(f"{name}: {len(o - n)} old samples are missing from the new bucket (tier or source changed)")
        print(f"{name}: all {len(o)} old samples present; {len(n) - len(o)} new samples")

    if NEW_DOC.exists():
        print(f"skip manifest: {NEW_DOC.name} exists")
    else:
        report = json.loads((NEW_BUCKETS / "report.json").read_text(encoding="utf8"))
        manifest = json.loads((OLD_DOC / "generation.json").read_text(encoding="utf8"))
        inputs = {}
        for name in TIERS.values():
            sha = hashlib.sha256((NEW_BUCKETS / f"{name}.bin").read_bytes()).hexdigest()
            if report["bucket_caches"][name]["sha256"] != sha:
                die(f"{name}.bin digest differs from new report.json")
            inputs[name] = sha
        manifest["inputs"] = inputs
        manifest["reference_sha256"] = report["reference_sha256"]
        tok = hashlib.sha256((ROOT / "yumon_code_bpe/tokenizer.json").read_bytes()).hexdigest()
        if tok != manifest["tokenizer_sha256"]:
            die("yumon_code_bpe tokenizer.json differs from the one the old comments used")
        NEW_DOC.mkdir()
        write_json(NEW_DOC / "generation.json", manifest)
        print(f"created {NEW_DOC.name}/generation.json for the new buckets (journals start empty)")

    print("\nprepare OK. Captioning command (old comments are reused by hash, no endpoint calls for them):")
    print("  cargo run --release --no-default-features --bin gen_synthetic_captions -- "
          "--input training-cache/code_buckets_v2 --output training-cache/code_buckets_documented_v2 "
          "--reuse training-cache/code_buckets_documented "
          '--endpoint "..." --model "unsloth/Qwen3.8-27B-GGUF" --token "..."')


def finalize():
    if not NEW_DOC.exists():
        die("run `prepare` first")
    cfg = json.loads((ROOT / "configs/yumon-code.json").read_text(encoding="utf8"))
    cfg["exclude_dirs"] += [d for d in EXTRA_EXCLUDES if d not in cfg["exclude_dirs"]]
    cfg["curriculum"] = "training-cache/code_buckets_v2"
    cfg["documented"] = "training-cache/code_buckets_documented_v2"
    write_json(PROD_V2, cfg)

    name, k = "code_tokenizer_v2", 1
    while (TC / name).exists():
        k += 1
        name = f"code_tokenizer_v2_{k}"
    cargo("train_bpe", "--", "--code-config", str(PROD_V2), str(TC / name))
    bundle_cfg = TC / name / "config.json"
    cargo("train_code", "--", "--config", str(bundle_cfg), "--check")
    if "docs" in json.loads(bundle_cfg.read_text(encoding="utf8")):
        cargo("cache_samples", "--", "--code-config", str(bundle_cfg), "--rust-docs", "--preview-samples", "0")
    else:
        print("bundle config has no `docs` key; skipping Rust books stage")
    print(f"\nfinalize OK: training-cache/{name}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    {"prepare": prepare, "finalize": finalize}.get(stage, lambda: die("usage: refresh_code_data.py prepare|finalize"))()
