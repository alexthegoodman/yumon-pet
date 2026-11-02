#!/usr/bin/env python3
"""Export pretrained VGG-19 convolutional backbone weights for Burn.

Extracts all 16 convolutional layers (up to relu5_4, as used in SRGAN-VGG54)
and serializes them into a fast, portable Little-Endian binary format.
"""

import struct
from pathlib import Path
import torch
import torchvision.models as models

MAGIC = b"YUMONVGG"
VERSION = 1

def main():
    out_path = Path("training-cache/vgg19_features.bin")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained torchvision VGG-19...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()

    conv_layers = [layer for layer in vgg if isinstance(layer, torch.nn.Conv2d)]
    print(f"Extracted {len(conv_layers)} Conv2d layers.")

    with open(out_path, "wb") as f:
        # Write header
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(conv_layers)))

        total_weights = 0
        total_biases = 0

        for idx, conv in enumerate(conv_layers):
            w = conv.weight.detach().cpu().to(torch.float32).numpy()
            b = conv.bias.detach().cpu().to(torch.float32).numpy()

            out_c, in_c, kh, kw = w.shape
            assert len(b) == out_c

            # Write layer metadata
            f.write(struct.pack("<IIII", out_c, in_c, kh, kw))

            # Write weight floats
            w_bytes = w.tobytes()
            f.write(w_bytes)

            # Write bias metadata and floats
            f.write(struct.pack("<I", out_c))
            b_bytes = b.tobytes()
            f.write(b_bytes)

            total_weights += w.size
            total_biases += b.size
            print(f"  Layer {idx:02d}: Conv2d({in_c} -> {out_c}, [{kh}, {kw}]) - {w.size} weights, {b.size} biases")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully exported VGG-19 backbone to {out_path} ({file_size_mb:.2f} MB)")
    print(f"Total params: {total_weights + total_biases:,} ({total_weights:,} weights, {total_biases:,} biases)")

if __name__ == "__main__":
    main()
