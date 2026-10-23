#!/usr/bin/env python3
"""
Preprocess animals_images dataset for Yumon Paint / Multimodal Art MoE:
1. Resizes all 1024x1024 images in data/animals_images/images/ to 64x64 into data/animals_images/images-64/.
2. Formats animal prompts from prompts.txt (substituting {mammal} with each animal class).
3. Emits DiffusionDb-compatible part-*.json manifests for build_diffusion_vocab and cache_diffusion_kilo.
4. Creates NTFS directory junctions data/animals_images_64x64 and data/animals_images-64.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ANIMALS_DIR = DATA_DIR / "animals_images"
SRC_IMAGES_DIR = ANIMALS_DIR / "images"
PROMPTS_FILE = ANIMALS_DIR / "prompts.txt"
DEST_IMAGES_DIR = ANIMALS_DIR / "images-64"

def clean_prompt_template(line: str) -> str:
    s = line.strip()
    if s.startswith(","):
        s = s[1:].strip()
    s = s.strip(' \t\r\n"\'\u201c\u201d')
    return s

def load_prompt_templates():
    if not PROMPTS_FILE.exists():
        raise FileNotFoundError(f"Prompts file not found at {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    templates = [clean_prompt_template(l) for l in lines]
    print(f"Loaded {len(templates)} prompt templates from {PROMPTS_FILE}")
    return templates

def resize_image(task):
    src_path, dst_path = task
    if dst_path.exists():
        return False
    try:
        with Image.open(src_path) as img:
            img_rgb = img.convert("RGB")
            resized = img_rgb.resize((64, 64), Image.Resampling.LANCZOS)
            resized.save(dst_path, "PNG")
        return True
    except Exception as e:
        print(f"Error resizing {src_path}: {e}", file=sys.stderr)
        return False

def main():
    print("=" * 60)
    print("=== Preprocessing animals_images -> 64x64 + Manifests ===")
    print("=" * 60)

    DEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    templates = load_prompt_templates()

    # Discover all source images
    print(f"Scanning {SRC_IMAGES_DIR}...")
    image_items = []  # (src_path, dst_filename, prompt)

    subdirs = sorted([d for d in os.listdir(SRC_IMAGES_DIR) if (SRC_IMAGES_DIR / d).is_dir()])
    print(f"Found {len(subdirs)} animal categories: {', '.join(subdirs)}")

    for animal in subdirs:
        animal_dir = SRC_IMAGES_DIR / animal
        png_files = sorted([f for f in os.listdir(animal_dir) if f.endswith(".png")])
        for f in png_files:
            parts = f.split("_")
            if not parts[0].isdigit():
                continue
            idx = int(parts[0])
            template = templates[(idx - 1) % len(templates)]
            prompt = template.replace("{mammal}", animal)
            src_path = animal_dir / f
            dst_filename = f
            image_items.append((src_path, dst_filename, prompt))

    print(f"Total images collected: {len(image_items)}")

    # Resize images
    print("\n[1/3] Resizing images to 64x64 using Lanczos...")
    t0 = time.time()
    resize_tasks = [
        (src_path, DEST_IMAGES_DIR / dst_filename)
        for (src_path, dst_filename, _) in image_items
    ]

    total_tasks = len(resize_tasks)
    completed = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as executor:
        for res in executor.map(resize_image, resize_tasks):
            completed += 1
            if completed % 1000 == 0 or completed == total_tasks:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  Processed {completed}/{total_tasks} images ({rate:.1f} imgs/s)...")

    elapsed = time.time() - t0
    print(f"Image resizing complete in {elapsed:.2f}s!")

    # Generate part-*.json manifests (1000 images per part)
    print("\n[2/3] Generating part-*.json manifests...")
    chunk_size = 1000
    num_parts = (len(image_items) + chunk_size - 1) // chunk_size

    for part_idx in range(num_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, len(image_items))
        chunk = image_items[start:end]

        manifest = {}
        for _, filename, prompt in chunk:
            manifest[filename] = {"p": prompt}

        part_filename = f"part-{part_idx + 1:06d}.json"
        part_path = DEST_IMAGES_DIR / part_filename
        with open(part_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"  Written {part_filename} with {len(manifest)} items")

    # Create junctions for alternate directory references
    print("\n[3/3] Setting up directory junctions...")
    junction_aliases = [
        DATA_DIR / "animals_images_64x64",
        DATA_DIR / "animals_images-64",
        ANIMALS_DIR / "images_64x64",
    ]
    for alias in junction_aliases:
        if not alias.exists():
            try:
                cmd = f'cmd /c mklink /J "{alias}" "{DEST_IMAGES_DIR}"'
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                print(f"  Created junction: {alias.name} -> images-64")
            except Exception as e:
                print(f"  Junction creation skipped for {alias.name}: {e}")
        else:
            print(f"  Directory/junction already exists: {alias.name}")

    print("\n=== Dataset preparation finished successfully! ===")

if __name__ == "__main__":
    main()
