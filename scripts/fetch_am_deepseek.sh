#!/bin/sh
# Pull the AM-DeepSeek-R1-Distilled dataset onto the RunPod volume at container start,
# instead of baking ~3 GB into the image. Same data as the Kaggle upload, but Hugging Face
# serves it with no login, and it supports HTTP range requests.
#
# Env:
#   YUMON_AM_PATH    destination file (default /workspace/data/am_deepseek/am_0.9M.jsonl.zst).
#                    Its basename picks which dataset file to fetch (am_0.5M / am_0.9M / ...).
#   AM_PREFIX_MB     fetch only the first N MiB of the .zst instead of the whole file. The Rust
#                    loader streams the zst and treats the cut as end of data. The head of the
#                    file is skewed toward a few sources, so prefer the full file if you can.
#   AM_DISABLE=1     skip entirely.
#
# Never fails the container: training runs without the dataset (train.rs skips it if absent).

DEST="${YUMON_AM_PATH:-/workspace/data/am_deepseek/am_0.9M.jsonl.zst}"
URL="https://huggingface.co/datasets/a-m-team/AM-DeepSeek-R1-Distilled-1.4M/resolve/main/$(basename "$DEST")"

[ "$AM_DISABLE" = "1" ] && { echo "[fetch_am] AM_DISABLE=1, skipping"; exit 0; }
mkdir -p "$(dirname "$DEST")" || exit 0

if [ -n "$AM_PREFIX_MB" ]; then
    BYTES=$((AM_PREFIX_MB * 1024 * 1024))
    if [ -f "$DEST" ] && [ "$(wc -c < "$DEST")" -ge "$BYTES" ]; then
        echo "[fetch_am] $DEST already has >= ${AM_PREFIX_MB} MiB, skipping"; exit 0
    fi
    echo "[fetch_am] fetching first ${AM_PREFIX_MB} MiB of $URL"
    curl -fsSL --retry 5 --retry-delay 5 -r "0-$((BYTES - 1))" -o "$DEST.part" "$URL" \
        && mv "$DEST.part" "$DEST" \
        || echo "[fetch_am] WARNING: prefix download failed, continuing without it"
    exit 0
fi

if [ -f "$DEST" ]; then
    echo "[fetch_am] $DEST exists, skipping"; exit 0
fi
echo "[fetch_am] downloading $URL -> $DEST (resumable)"
# -C - resumes $DEST.part if the pod restarted mid-download.
curl -fsSL --retry 10 --retry-delay 5 -C - -o "$DEST.part" "$URL" \
    && mv "$DEST.part" "$DEST" \
    || echo "[fetch_am] WARNING: download failed, continuing without it"
exit 0
