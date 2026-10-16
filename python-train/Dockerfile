# syntax=docker/dockerfile:1

################################################################################
# Yumon Pet Sparse MoE Language Model Training for RunPod (CUDA 13)
#
# Base image: RunPod PyTorch CUDA 13 template (Ubuntu 24.04, PyTorch, CUDA 13)
# Can be overridden via: --build-arg BASE_IMAGE=...
################################################################################
ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    YUMON_SAMPLE_CACHE=/app/training-cache/samples.bin \
    TOKENIZER_PATH=/app/tokenizer.json \
    CHECKPOINT_DIR=/workspace/checkpoints

WORKDIR /app

# Ensure python3 and pip are present if an alternative minimal base image is used
RUN if ! command -v pip3 >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates curl && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dataset & tokenizer
# Layered before code so code edits do not invalidate data cache layers
ARG SAMPLES_SRC=training-cache/samples.bin
COPY ${SAMPLES_SRC} ./training-cache/samples.bin

COPY tokenizer.json .

# Copy training and model scripts
COPY *.py ./

# Preflight verification during docker build (CPU-safe) to catch any missing data early
RUN python3 -c "\
from dataloader import load_samples_cache, YumonPetDataset; \
meta, samples = load_samples_cache('training-cache/samples.bin', max_samples=4); \
assert len(samples) >= 1, 'Failed to load samples'; \
from model import YumonPetMoeBrain, ModelConfig; \
cfg = ModelConfig(vocab_size=meta.get('vocab_size', 16384), embed_dim=64, n_layers=2, attn_heads=4, ff_dim=128, max_seq_len=64, num_experts=4, top_k=1); \
m = YumonPetMoeBrain(cfg); \
ds = YumonPetDataset(samples, max_seq_len=64); \
t, l = ds[0]; \
logits, aux = m(t.unsqueeze(0)); \
assert logits.shape == (1, 64, cfg.vocab_size); \
print(f'Build preflight passed: loaded {len(samples)} test samples and executed forward pass successfully.')"

# Checkpoints land on /workspace/checkpoints when a RunPod network volume is mounted.
# Default command starts training immediately upon pod launch.
CMD ["python3", "train.py"]
