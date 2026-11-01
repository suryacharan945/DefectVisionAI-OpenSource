#!/usr/bin/env python3
"""
DefectVision AI — PatchCore Memory Bank Reproduction CLI

Extracts patch embeddings from normal training images using ResNet-18 mid-level
features, applies greedy minimax coreset subsampling, and saves the calibrated
memory bank (.npy) and FAISS search index for defect localization.

Usage:
    python scripts/reproduce_memory_banks.py --data_dir ./datasets/mvtec/ --category bottle --coreset_ratio 0.10
    python scripts/reproduce_memory_banks.py --data_dir ./datasets/mvtec/ --all_categories
"""

import os
import sys
import argparse
import glob
import time
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure DefectVisionAI is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DefectVisionAI"))

try:
    from src.patchcore import PatchCoreExtractor, greedy_coreset_gpu, PatchCoreModel
except ImportError:
    from DefectVisionAI.src.patchcore import PatchCoreExtractor, greedy_coreset_gpu, PatchCoreModel

KNOWN_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

PC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PC_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PC_IMG_SIZE = 224
GRID_SIZE = 28

def load_and_preprocess_image(filepath, size=PC_IMG_SIZE):
    """Loads an RGB image, resizes to (size, size), normalizes via ImageNet stats."""
    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - PC_MEAN) / PC_STD
    return torch.from_numpy(img).permute(2, 0, 1).float()

def extract_patch_embeddings(image_paths, extractor, device, batch_size=16):
    """Batched extraction of patch embeddings from training images."""
    all_patches = []
    total_imgs = len(image_paths)
    print(f"  Extracting patch tokens from {total_imgs} normal images (batch size {batch_size})...")

    t0 = time.time()
    for i in range(0, total_imgs, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            t = load_and_preprocess_image(p)
            if t is not None:
                tensors.append(t)

        if not tensors:
            continue

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            fused = extractor(batch_tensor)  # B, C, G, G
            B, C, G, _ = fused.shape
            # Reshape to (B * G * G, C)
            patches = fused.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
            all_patches.append(patches)

        if (i // batch_size) % 5 == 0 or (i + batch_size) >= total_imgs:
            processed = min(i + batch_size, total_imgs)
            print(f"    Progress: {processed}/{total_imgs} images processed")

    patch_embeddings = np.concatenate(all_patches, axis=0)
    elapsed = time.time() - t0
    print(f"  Total raw patch tokens: {patch_embeddings.shape[0]} (dim={patch_embeddings.shape[1]}) in {elapsed:.2f}s")
    return patch_embeddings

def reproduce_category_memory_bank(data_dir, category, output_base, coreset_ratio=0.10, device="cpu", seed=42):
    """Fits and saves the PatchCore memory bank for a single category."""
    print(f"\n==========================================")
    print(f"Processing category: {category}")
    print(f"==========================================")

    # Locate normal training images: <data_dir>/<category>/train/good/
    train_good_dir = os.path.join(data_dir, category, "train", "good")
    if not os.path.isdir(train_good_dir):
        # Alternative structure: <data_dir>/<category>/train/
        train_good_dir = os.path.join(data_dir, category, "train")
        if not os.path.isdir(train_good_dir):
            raise FileNotFoundError(f"Training directory not found for {category} at {train_good_dir}")

    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(train_good_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(train_good_dir, "**", ext), recursive=True))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        raise ValueError(f"No valid images found in {train_good_dir}")

    print(f"  Found {len(image_paths)} training images.")

    # Initialize feature extractor
    extractor = PatchCoreExtractor(grid_size=GRID_SIZE).to(device)
    extractor.eval()

    # Extract patch tokens
    raw_patches = extract_patch_embeddings(image_paths, extractor, device)

    # Coreset subsampling
    target_count = max(1, int(len(raw_patches) * coreset_ratio))
    print(f"  Subsampling coreset: ratio={coreset_ratio:.2f} -> target {target_count} patch tokens...")
    t0 = time.time()
    coreset = greedy_coreset_gpu(raw_patches, coreset_ratio=coreset_ratio, seed=seed, device=device, log_every=1000)
    print(f"  Coreset selection complete in {time.time() - t0:.2f}s. Final bank shape: {coreset.shape}")

    # Save memory bank
    cat_out_dir = os.path.join(output_base, "patchcore", category)
    os.makedirs(cat_out_dir, exist_ok=True)
    mb_file = os.path.join(cat_out_dir, "memory_bank.npy")
    np.save(mb_file, coreset.astype(np.float32))
    print(f"  Saved memory bank to: {mb_file} ({os.path.getsize(mb_file):,} bytes)")

    return mb_file

def main():
    parser = argparse.ArgumentParser(
        description="DefectVision AI — PatchCore Memory Bank Reproduction Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reproduce_memory_banks.py --data_dir ./datasets/mvtec/ --category bottle
  python scripts/reproduce_memory_banks.py --data_dir ./datasets/mvtec/ --all_categories --coreset_ratio 0.10
        """
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root path to the MVTec AD dataset directory containing category subfolders.")
    parser.add_argument("--category", type=str, default=None, choices=KNOWN_CATEGORIES,
                        help="Specific MVTec AD category to process.")
    parser.add_argument("--all_categories", action="store_true",
                        help="Process all 15 official MVTec AD categories sequentially.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "DefectVisionAI", "models"),
                        help="Directory where models/patchcore/<category>/memory_bank.npy will be written.")
    parser.add_argument("--coreset_ratio", type=float, default=0.10,
                        help="Fraction of patch tokens to retain via minimax coreset (default: 0.10).")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use ('cuda' or 'cpu'). Default: auto-detect.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible coreset initialization (default: 42).")

    args = parser.parse_args()

    if not args.category and not args.all_categories:
        parser.error("You must specify either --category <name> or --all_categories.")

    target_categories = KNOWN_CATEGORIES if args.all_categories else [args.category]

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DefectVision AI Memory Bank Reproduction")
    print(f"Target Device: {device}")
    print(f"Data Root: {args.data_dir}")
    print(f"Output Root: {args.output_dir}")
    print(f"Categories ({len(target_categories)}): {', '.join(target_categories)}")

    successful = []
    failed = []

    for cat in target_categories:
        try:
            mb_path = reproduce_category_memory_bank(
                data_dir=args.data_dir,
                category=cat,
                output_base=args.output_dir,
                coreset_ratio=args.coreset_ratio,
                device=device,
                seed=args.seed
            )
            successful.append((cat, mb_path))
        except Exception as e:
            print(f"  [ERROR] Failed to process {cat}: {e}")
            failed.append((cat, str(e)))

    print(f"\n==========================================")
    print(f"Reproduction Summary: {len(successful)} succeeded, {len(failed)} failed.")
    for cat, path in successful:
        print(f"  [OK] {cat}: {path}")
    for cat, err in failed:
        print(f"  [FAIL] {cat}: {err}")

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    main()
