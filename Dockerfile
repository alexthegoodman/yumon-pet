# syntax=docker/dockerfile:1

########################################
# Builder: compile the headless training binary.
#
# Built with `--no-default-features` so the "desktop" feature (tao/wry/gilrs/
# three-d - native window + webview + 3D engine + gamepad, used by the
# desktop/world/chat UI bins) is skipped entirely. Training only needs the
# Code CLI (src/bin/train_code.rs), which never touches any of that, so we avoid having
# to install GTK/WebKit/udev dev packages just to compile them.
########################################
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# cudarc (cubecl-cuda's CUDA binding) picks which driver-API symbols to load
# based on a target CUDA version. This builder has no CUDA toolkit installed,
# so it can't run `nvcc --version` to auto-detect one, and cudarc's fallback
# is to assume the *newest* CUDA version it knows about - which then eagerly
# dlsym's symbols (e.g. cuCtxGetDevice_v2) that don't exist in an older
# driver, panicking at container start with "undefined symbol".
#
# (cubecl-cuda 0.9.0's actual compile-time floor is CUDA 12.0 - anything
# below fails to build, verified locally.) cudarc 0.18.2 supports up to CUDA
# 13.1, so target 13010 with the CUDA 13.2 runtime below. 13020 is not a
# supported value in this dependency version. Keep the major version aligned
# so dynamic loading searches for CUDA 13 toolkit libraries such as NVRTC.
# Override at build time with `--build-arg CUDARC_CUDA_VERSION=...` if needed.
ARG CUDARC_CUDA_VERSION=13010
ENV CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION}

# Pet training (disabled):
# RUN cargo build --release --no-default-features --bin yumon-pet
RUN cargo build --release --no-default-features --bin train_code

########################################
# Runtime: training uses burn's CUDA backend, which talks to the CUDA driver
# directly (via cudarc's dynamic loading of libcuda/libnvrtc at process
# start). No Vulkan/GL loader needed here: RunPod's driver stack doesn't
# expose a usable Vulkan/GL adapter, which is why this used to run on
# wgpu/Vulkan and crash there with "No possible adapter available for
# backend".
#
# libcuda.so itself is host-mounted at container start by RunPod's
# nvidia-container-toolkit, but libnvrtc.so (the CUDA runtime compiler
# cubecl-cuda JIT-compiles kernels with) is a CUDA *toolkit* library, not a
# driver library - the toolkit never mounts it, it has to be baked into the
# image. That's why a plain debian:bookworm-slim base failed with a "cuda
# not found"-style error even after the CUDARC_CUDA_VERSION fix: nothing in
# this image provided libnvrtc.so.
#
# Using RunPod's own official base image (github.com/runpod/containers,
# official-templates/base) rather than a bare nvidia/cuda image, per
# runpod's own guidance for images meant to run on their fleet.
#
# Note: this tag's version also sets the *minimum required driver* -
# nvidia-container-toolkit refuses to even start the container if the host
# driver is older than what the image declares. A 12.6.3 nvidia/cuda image
# was rejected here with "unsatisfied condition: cuda>=12.6" against the
# actual RunPod host driver, so if 13.2.0 hits the same rejection, that's
# the same real constraint (older driver on that specific pod), not
# something this image swap alone fixes - the fix in that case is a pod
# with a newer driver (check `nvidia-smi` -> "CUDA Version: X.Y" on it
# before deploying), or dropping this tag and CUDARC_CUDA_VERSION back down
# to something like 12.0 (nvidia/cuda:12.0.0-runtime-ubuntu22.04).
########################################
FROM runpod/base:1.4.0-cuda1320-ubuntu2404 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Precompute this file locally with the cache_samples binary before building.
# Runtime training uses only the snapshot; no corpus download or tokenization.
# Pet cache (disabled; Code reads its cache path from the JSON config):
# ENV YUMON_SAMPLE_CACHE=/app/training-cache/samples.bin

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV RUST_BACKTRACE=1

WORKDIR /app

# The default image trains Yumon Code. No Pet binary or cache is included.
# Config paths must match the destinations below (or point at mounted volume files).
COPY --from=builder /build/target/release/train_code ./train_code
ARG CODE_TOKENIZER=yumon_code_bpe
ARG CODE_CACHE=training-cache/code.bin
ARG CODE_CONFIG=configs/yumon-code.json
COPY ${CODE_TOKENIZER}/ ./yumon_code_bpe/
COPY ${CODE_CACHE} ./training-cache/code.bin
COPY ${CODE_CONFIG} ./configs/yumon-code.json
RUN ./train_code --config configs/yumon-code.json --check
CMD ["./train_code", "--config", "configs/yumon-code.json"]

# Pet training (disabled; restore together with the Pet build and ENV above):
# COPY --from=builder /build/target/release/yumon-pet ./yumon-pet
# COPY yumon_bpe ./yumon_bpe

# COPY training-cache/samples.bin ./training-cache/samples.bin
# Catch a missing local preparation step at image build time.
# RUN test -s "$YUMON_SAMPLE_CACHE"

# Checkpoints must land on a mounted RunPod Network Volume (not this image's
# writable layer) so they survive the pod being stopped/terminated - mount
# your volume at /workspace. See README.md for the full RunPod walkthrough.
# CMD ["./yumon-pet", "train-brain", "--out-dir", "/workspace/checkpoints/brain", "--batch-size", "16"]
