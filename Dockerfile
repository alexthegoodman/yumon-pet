# syntax=docker/dockerfile:1

########################################
# Builder: compile the headless training binary.
#
# Built with `--no-default-features` so the "desktop" feature (tao/wry/gilrs/
# three-d - native window + webview + 3D engine + gamepad, used by the
# desktop/world/chat UI bins) is skipped entirely. Training only needs the
# CLI bin (src/main.rs), which never touches any of that, so we avoid having
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
# below fails to build, verified locally.) Set to 12080 to match the
# runpod/base:*-cuda1281-* runtime image below (12.8.1 rounds down to
# cudarc's nearest known value, 12.8.0 - fine, older ABI subset). Keep this
# in sync with that image's version; override at build time with
# `--build-arg CUDARC_CUDA_VERSION=12060` if you switch images again.
ARG CUDARC_CUDA_VERSION=12080
ENV CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION}

RUN cargo build --release --no-default-features --bin yumon-pet

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
# actual RunPod host driver, so if 12.8.1 hits the same rejection, that's
# the same real constraint (older driver on that specific pod), not
# something this image swap alone fixes - the fix in that case is a pod
# with a newer driver (check `nvidia-smi` -> "CUDA Version: X.Y" on it
# before deploying), or dropping this tag and CUDARC_CUDA_VERSION back down
# to something like 12.0 (nvidia/cuda:12.0.0-runtime-ubuntu22.04).
########################################
FROM runpod/base:1.0.2-cuda1281-ubuntu2204 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
COPY --from=builder /build/target/release/yumon-pet ./yumon-pet
COPY yumon_bpe ./yumon_bpe

# Only the text data actually used by brain::train::run's data loader -
# CIFAR-100/FER2013/ebooks/images are vision-only and unused here.
COPY data/ideas.txt data/wiki_extract.txt data/bible_bbe.csv data/bible_asv.csv \
     data/creative_stories.txt data/The-Office-Lines-V4.csv \
     data/friends_all_episodes_clean.csv ./data/
COPY archive/arena_extract.txt archive/ov_chats.txt archive/you_chats.txt \
     archive/clean_chats.txt ./archive/

# Checkpoints must land on a mounted RunPod Network Volume (not this image's
# writable layer) so they survive the pod being stopped/terminated - mount
# your volume at /workspace. See README.md for the full RunPod walkthrough.
CMD ["./yumon-pet", "train-brain", "--out-dir", "/workspace/checkpoints/brain"]
