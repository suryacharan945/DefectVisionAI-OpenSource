#!/usr/bin/env python3
"""
DefectVision AI — Decision Threshold Calibration CLI

Computes empirical anomaly score distributions on nominal validation images,
deriving the 99.5th percentile detection threshold and baseline normalization
parameters for industrial inspection categories.

Usage:
    python scripts/calibrate_thresholds.py --data_dir ./datasets/mvtec/ --category bottle
    python scripts/calibrate_thresholds.py --data_dir ./datasets/mvtec/ --all_categories --percentile 99.5
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
import pandas as pd
import cv2
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DefectVisionAI"))

try:
    from src.patchcore import PatchCoreExtractor, PatchCoreModel
except ImportError:
    from DefectVisionAI.src.patchcore import PatchCoreExtractor, PatchCoreModel

KNOWN_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

PC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PC_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PC_IMG_SIZE = 224
GRID_SIZE = 28

def load_pc_tensor(filepath, size=PC_IMG_SIZE):
    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - PC_MEAN) / PC_STD
    return torch.from_numpy(img).permute(2, 0, 1).float()

def calibrate_category(data_dir, category, models_dir, percentile=99.5, device="cpu"):
    """Evaluates normal validation images for category and computes baseline statistics."""
    print(f"\n==========================================")
    print(f"Calibrating category: {category}")
    print(f"==========================================")

    # 1. Load memory bank
    mb_path = os.path.join(models_dir, "patchcore", category, "memory_bank.npy")
    if not os.path.exists(mb_path):
        raise FileNotFoundError(
            f"PatchCore memory bank not found at {mb_path}. "
            f"Please run scripts/reproduce_memory_banks.py first."
        )

    model = PatchCoreModel()
    model.load(mb_path, grid_size=GRID_SIZE)

    # 2. Find normal validation images (<data_dir>/<category>/test/good/)
    test_good_dir = os.path.join(data_dir, category, "test", "good")
    if not os.path.isdir(test_good_dir):
        # Fallback to train/good if test/good not available
        test_good_dir = os.path.join(data_dir, category, "train", "good")
        if not os.path.isdir(test_good_dir):
            raise FileNotFoundError(f"No nominal validation images found for {category}")

    image_paths = sorted(glob.glob(os.path.join(test_good_dir, "*.png")) +
                         glob.glob(os.path.join(test_good_dir, "*.jpg")))
    if not image_paths:
        raise ValueError(f"No image files found in {test_good_dir}")

    print(f"  Found {len(image_paths)} nominal validation images in {test_good_dir}")

    # 3. Score each image
    extractor = PatchCoreExtractor(grid_size=GRID_SIZE).to(device)
    extractor.eval()

    scores = []
    t0 = time.time()
    for idx, p in enumerate(image_paths):
        tensor = load_pc_tensor(p)
        if tensor is None:
            continue
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            fused = extractor(tensor)[0].cpu().numpy()
        _, img_score = model.score(fused)
        scores.append(float(img_score))

    scores = np.array(scores)
    mean_val = float(np.mean(scores))
    p_thresh = float(np.percentile(scores, percentile))
    elapsed = time.time() - t0

    print(f"  Processed {len(scores)} samples in {elapsed:.2f}s")
    print(f"  Baseline Mean: {mean_val:.4f}")
    print(f"  P{percentile} Threshold: {p_thresh:.4f}")

    return {
        "category": category,
        "baseline_mean": mean_val,
        "baseline_p99": p_thresh,
        "n_normal_samples": len(scores)
    }

def main():
    parser = argparse.ArgumentParser(
        description="DefectVision AI — Decision Threshold Calibration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate_thresholds.py --data_dir ./datasets/mvtec/ --category bottle
  python scripts/calibrate_thresholds.py --data_dir ./datasets/mvtec/ --all_categories --percentile 99.5
        """
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing category dataset folders.")
    parser.add_argument("--category", type=str, default=None, choices=KNOWN_CATEGORIES,
                        help="Specific category to calibrate.")
    parser.add_argument("--all_categories", action="store_true",
                        help="Calibrate all 15 official MVTec AD categories.")
    parser.add_argument("--models_dir", type=str, default=os.path.join(PROJECT_ROOT, "DefectVisionAI", "models"),
                        help="Directory containing trained memory banks.")
    parser.add_argument("--percentile", type=float, default=99.5,
                        help="Percentile threshold for boundary calibration (default: 99.5).")
    parser.add_argument("--output_csv", type=str,
                        default=os.path.join(PROJECT_ROOT, "DefectVisionAI", "reports", "step9_severity_baselines.csv"),
                        help="Target CSV file for baseline calibrations.")
    parser.add_argument("--device", type=str, default=None,
                        help="Computation device ('cuda' or 'cpu'). Default: auto-detect.")

    args = parser.parse_args()

    if not args.category and not args.all_categories:
        parser.error("You must specify either --category <name> or --all_categories.")

    target_categories = KNOWN_CATEGORIES if args.all_categories else [args.category]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load existing baseline CSV if present to allow selective updates
    results = {}
    if os.path.exists(args.output_csv):
        try:
            existing_df = pd.read_csv(args.output_csv)
            for _, row in existing_df.iterrows():
                results[row["category"]] = row.to_dict()
        except Exception:
            pass

    for cat in target_categories:
        try:
            res = calibrate_category(
                data_dir=args.data_dir,
                category=cat,
                models_dir=args.models_dir,
                percentile=args.percentile,
                device=device
            )
            results[cat] = res
        except Exception as e:
            print(f"  [ERROR] Calibration failed for {cat}: {e}")

    # Write updated CSV
    if results:
        df = pd.DataFrame(list(results.values()))
        df = df.sort_values("category").reset_index(drop=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved updated baseline calibration to: {args.output_csv}")
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
