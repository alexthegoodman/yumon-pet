# High-Yield Decompression Strategies

## Decompressor Model

Depending on your target compute budget and deployment constraints, here are three ways to build the decompressor, ordered from lightest to most expressive:1. Contextual N-Gram / Statistical Disambiguation (Zero Neural Overhead)If your mapping is mostly 1-to-many synonyms (e.g., compressed token BIG can decompress to large, huge, massive, significant, vast):How it works: Build a simple $n$-gram language model (like a KenLM tri-gram or 4-gram) on the raw, rich English corpus.Decoding: When your tiny model outputs BIG, you look up its allowable word set $\{w_1, w_2, \dots, w_k\}$ and pick:$$\arg\max_{w_i} P(w_i \mid w_{t-2}, w_{t-1})$$Why it shines: Takes virtually zero RAM, runs in microseconds on CPU, and ensures grammatical collocations sound natural (e.g., selecting "significant impact" over "massive impact" based strictly on local bi-gram frequency).

## Selector Model

Selects from multiple options produced by the decompressor model. Is trained on selecting real sentences out from fake ones.