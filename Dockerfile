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

RUN cargo build --release --no-default-features --bin yumon-pet

########################################
# Runtime: minimal image - training uses burn's CUDA backend, which talks to
# the CUDA driver directly (via cudarc's dynamic loading of libcuda/libnvrtc
# at process start). No Vulkan/GL loader needed here: RunPod's driver stack
# doesn't expose a usable Vulkan/GL adapter, which is why this used to run on
# wgpu/Vulkan and crash there with "No possible adapter available for
# backend". The actual CUDA driver is provided at container start by RunPod's
# nvidia-container-toolkit - nothing GPU-specific to install here.
########################################
FROM debian:bookworm-slim AS runtime

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
