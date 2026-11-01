"""
CPND Development-Only Benchmark & Ablation Runner (Optimized)
DefectVision AI — Phase 7B Experimental Evaluation

Evaluates candidate uncertainty metrics (Random, Top-Score, Distance-to-Threshold,
PND-only, Combined Joint) across multiple review budgets on authorized development data.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import cv2
import faiss
from typing import Dict, List, Tuple, Optional, Union

# Add paths
sys.path.insert(0, os.path.abspath("DefectVisionAI"))
from src.patchcore import PatchCoreExtractor
from research.cpnd.cpnd_core import (
    rank_and_select_budget,
    evaluate_selective_review
)

# Configuration
DEVICE = torch.device("cpu")
PC_IMG_SIZE = 224
PC_MEAN = np.array([0.485, 0.456, 0.406])
PC_STD = np.array([0.229, 0.224, 0.225])
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

# Selected development categories covering rigid objects, pharmaceuticals, and mechanical fasteners
DEV_CATEGORIES = [
    "bottle",      # Rigid transparent object (83 test images)
    "capsule",     # Rigid pharmaceutical object (132 test images)
    "metal_nut"    # Rigid metal mechanical part (115 test images)
]

def load_pc_tensor(filepath: str, size: int = PC_IMG_SIZE) -> Optional[torch.Tensor]:
    g_path = filepath.replace("/content/drive/MyDrive/DefectVisionAI", "G:\\My Drive\\DefectVisionAI").replace("/", "\\")
    if not os.path.exists(g_path):
        return None
    img = cv2.imread(g_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
    img = (img - PC_MEAN) / PC_STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def extract_dev_features_and_scores(
    categories: List[str] = DEV_CATEGORIES,
    k_list: List[int] = [3, 5, 10, 20],
    output_csv: str = "DefectVisionAI/research/cpnd/dev_extracted_scores.csv"
) -> pd.DataFrame:
    """
    Extracts PatchCore features and computes PND scores for development images
    using a single vectorized FAISS query (k=max(k_list)) per image.
    """
    baselines_df = pd.read_csv("DefectVisionAI/reports/step9_severity_baselines.csv")
    thresh_lookup = dict(zip(baselines_df['category'], baselines_df['baseline_p99']))

    step7_df = pd.read_csv("DefectVisionAI/logs/step7_patchcore_inference_log.csv")
    dev_subset = step7_df[step7_df['category'].isin(categories)].copy()
    total_imgs = len(dev_subset)
    print(f"Total development images to evaluate across {len(categories)} categories: {total_imgs}", flush=True)

    extractor = PatchCoreExtractor().to(DEVICE)
    extractor.eval()

    # Pre-build FAISS index for each category
    faiss_indices = {}
    max_k = max(k_list)
    for cat in categories:
        mb_path = f"DefectVisionAI/models/patchcore/{cat}/memory_bank.npy"
        if os.path.exists(mb_path):
            mb = np.ascontiguousarray(np.load(mb_path), dtype=np.float32)
            idx = faiss.IndexFlatL2(mb.shape[1])
            idx.add(mb)
            faiss_indices[cat] = idx

    records = []
    t_start = time.perf_counter()

    for idx, (_, row) in enumerate(dev_subset.iterrows()):
        cat = row['category']
        fp = row['filepath']
        true_label = str(row['true_label']).lower()
        tau = float(thresh_lookup.get(cat, 1.50))

        if cat not in faiss_indices:
            continue

        t_img = load_pc_tensor(fp)
        if t_img is None:
            continue

        with torch.no_grad():
            feat = extractor(t_img)[0].cpu().numpy()

        C, g1, g2 = feat.shape
        patches = np.ascontiguousarray(feat.reshape(C, g1 * g2).T, dtype=np.float32)

        # Single FAISS search for top-20 neighbors
        sq_dist, _ = faiss_indices[cat].search(patches, k=max_k)
        euclid_dist = np.sqrt(np.maximum(sq_dist, 0.0))

        # Standard PatchCore 1-NN score
        dist_1nn = euclid_dist[:, 0]
        argmax_idx = int(np.argmax(dist_1nn))
        score = float(dist_1nn[argmax_idx])

        rec = {
            "image_id": f"{cat}_{os.path.basename(fp)}",
            "category": cat,
            "filepath": fp,
            "true_label": true_label,
            "decision_threshold": tau,
            "anomaly_score": score,
            "predicted_label": "defective" if score > tau else "normal",
            "is_error": int(("defective" if score > tau else "normal") != true_label),
            "is_fn": int(true_label == "defective" and score <= tau),
            "is_fp": int(true_label == "normal" and score > tau),
            "u_dist": float(1.0 / (abs(score - tau) + 1e-5)),
        }

        # Multi-K and multi-aggregation dispersion calculations (vectorized NumPy)
        for k in k_list:
            d_k = euclid_dist[:, :k]
            means = np.mean(d_k, axis=1)
            stds = np.std(d_k, axis=1, ddof=0)
            pnd_k = stds / (means + 1e-7)

            rec[f"pnd_k{k}_argmax"] = float(pnd_k[argmax_idx])
            rec[f"pnd_k{k}_max"] = float(np.max(pnd_k))

            if k == 5:
                rec["u_pnd_k5_argmax"] = float(pnd_k[argmax_idx])
                rec["u_pnd_k5_max"] = float(np.max(pnd_k))
                k_top = max(1, int(len(pnd_k) * 0.05))
                top_pnd = np.partition(pnd_k, -k_top)[-k_top:]
                rec["u_pnd_k5_top5"] = float(np.mean(top_pnd))

        records.append(rec)
        if (idx + 1) % 25 == 0 or (idx + 1) == total_imgs:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            print(f"  Processed {idx + 1}/{total_imgs} images ({rate:.1f} img/s, {elapsed:.1f}s elapsed)...", flush=True)

    df_out = pd.DataFrame(records)

    # Category-level rank normalization for Distance and PND
    def rank_norm(s):
        return (s.rank() - 1.0) / max(1.0, len(s) - 1.0)

    df_out["u_dist_norm"] = df_out.groupby("category")["u_dist"].transform(rank_norm)
    df_out["u_pnd_norm"] = df_out.groupby("category")["u_pnd_k5_argmax"].transform(rank_norm)

    # Joint Uncertainty: 0.5 * u_dist_norm + 0.5 * u_pnd_norm
    df_out["u_joint"] = 0.5 * df_out["u_dist_norm"] + 0.5 * df_out["u_pnd_norm"]

    # Development-calibrated PND
    def dev_calibrated_pnd(group):
        norm_pnd = group.loc[group['true_label'] == 'normal', 'u_pnd_k5_argmax']
        p05 = norm_pnd.quantile(0.05) if len(norm_pnd) > 0 else group['u_pnd_k5_argmax'].min()
        p95 = norm_pnd.quantile(0.95) if len(norm_pnd) > 0 else group['u_pnd_k5_argmax'].max()
        denom = p95 - p05 if (p95 - p05) > 1e-6 else 1.0
        return np.clip((group['u_pnd_k5_argmax'] - p05) / denom, 0.0, 1.0)

    df_out["u_pnd_calibrated"] = df_out.groupby("category").apply(dev_calibrated_pnd).reset_index(level=0, drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Extraction complete! Saved {len(df_out)} scored records to {output_csv}", flush=True)
    return df_out


def run_bootstrap_ci(
    records: List[Dict],
    uncertainty_col: str,
    budget: float,
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """Computes bootstrap confidence intervals for ECR and DER at budget B."""
    rng = np.random.RandomState(seed)
    n = len(records)
    ecr_list = []
    der_list = []

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        sample = [records[i] for i in indices]
        reviewed, _ = rank_and_select_budget(sample, uncertainty_col, budget)
        res = evaluate_selective_review(sample, reviewed)
        ecr_list.append(res["error_capture_rate"])
        der_list.append(res["defect_escape_reduction"])

    return {
        "ecr_mean": float(np.mean(ecr_list)),
        "ecr_ci_low": float(np.percentile(ecr_list, 2.5)),
        "ecr_ci_high": float(np.percentile(ecr_list, 97.5)),
        "der_mean": float(np.mean(der_list)),
        "der_ci_low": float(np.percentile(der_list, 2.5)),
        "der_ci_high": float(np.percentile(der_list, 97.5))
    }


def run_full_benchmark(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Runs development comparison across baselines, budgets, and ablations."""
    records = df.to_dict(orient="records")
    total_imgs = len(df)
    total_errors = df["is_error"].sum()
    total_fn = df["is_fn"].sum()
    total_fp = df["is_fp"].sum()

    print(f"\n================ DEVELOPMENT BENCHMARK SUMMARY ================", flush=True)
    print(f"Total Images: {total_imgs}", flush=True)
    print(f"Total PatchCore Errors: {total_errors} ({total_fn} FN, {total_fp} FP)", flush=True)
    print(f"Baseline Accuracy: {(total_imgs - total_errors) / total_imgs * 100:.2f}%", flush=True)

    policies = {
        "Random Review (BL0)": "random_score",
        "Top Anomaly Score (BL1)": "anomaly_score",
        "Distance to Threshold (BL2)": "u_dist",
        "Patch Dispersion Only (CPND-Raw)": "u_pnd_k5_argmax",
        "Patch Dispersion (CPND-Calibrated)": "u_pnd_calibrated",
        "Joint Boundary + CPND (Proposed)": "u_joint"
    }

    np.random.seed(42)
    for r in records:
        r["random_score"] = np.random.uniform()

    # 1. Budget Results Table
    budget_rows = []
    for pol_name, pol_col in policies.items():
        for b in BUDGETS:
            reviewed, k = rank_and_select_budget(records, pol_col, b)
            metrics = evaluate_selective_review(records, reviewed)
            ci = run_bootstrap_ci(records, pol_col, b, n_resamples=500)
            budget_rows.append({
                "Policy": pol_name,
                "Budget": b,
                "Reviewed_K": k,
                "Total_Errors": total_errors,
                "Captured_Errors": metrics["captured_errors"],
                "Captured_FN": metrics["captured_fn"],
                "Captured_FP": metrics["captured_fp"],
                "ECR": round(metrics["error_capture_rate"] * 100, 2),
                "ECR_CI_Low": round(ci["ecr_ci_low"] * 100, 2),
                "ECR_CI_High": round(ci["ecr_ci_high"] * 100, 2),
                "DER": round(metrics["defect_escape_reduction"] * 100, 2),
                "DER_CI_Low": round(ci["der_ci_low"] * 100, 2),
                "DER_CI_High": round(ci["der_ci_high"] * 100, 2),
                "Review_Yield": round(metrics["review_yield"] * 100, 2)
            })

    budget_df = pd.DataFrame(budget_rows)

    # 2. Baseline Comparison Table at B = 10%
    b10_df = budget_df[budget_df["Budget"] == 0.10].copy()

    # 3. Ablation Studies Table (at B = 10%)
    ablation_policies = {
        "Ablation A: Distance-to-Threshold Only": "u_dist",
        "Ablation B: PND Only (k=5, argmax)": "u_pnd_k5_argmax",
        "Ablation C: Joint Distance + PND (alpha=0.5)": "u_joint",
        "Ablation D1: PND Only (k=3, argmax)": "pnd_k3_argmax",
        "Ablation D2: PND Only (k=10, argmax)": "pnd_k10_argmax",
        "Ablation D3: PND Only (k=20, argmax)": "pnd_k20_argmax",
        "Ablation E1: PND Aggregation: Argmax Patch": "u_pnd_k5_argmax",
        "Ablation E2: PND Aggregation: Max Patch": "u_pnd_k5_max",
        "Ablation E3: PND Aggregation: Mean Top-5%": "u_pnd_k5_top5",
        "Ablation F1: Raw PND (k=5)": "u_pnd_k5_argmax",
        "Ablation F2: Development Calibrated PND": "u_pnd_calibrated"
    }

    ablation_rows = []
    for abl_name, abl_col in ablation_policies.items():
        reviewed, k = rank_and_select_budget(records, abl_col, 0.10)
        metrics = evaluate_selective_review(records, reviewed)
        ci = run_bootstrap_ci(records, abl_col, 0.10, n_resamples=500)
        ablation_rows.append({
            "Ablation_Experiment": abl_name,
            "Reviewed_K": k,
            "Captured_Errors": metrics["captured_errors"],
            "Captured_FN": metrics["captured_fn"],
            "Captured_FP": metrics["captured_fp"],
            "ECR_at_10pct": round(metrics["error_capture_rate"] * 100, 2),
            "ECR_95CI": f"[{ci['ecr_ci_low']*100:.1f}%, {ci['ecr_ci_high']*100:.1f}%]",
            "DER_at_10pct": round(metrics["defect_escape_reduction"] * 100, 2),
            "DER_95CI": f"[{ci['der_ci_low']*100:.1f}%, {ci['der_ci_high']*100:.1f}%]",
            "Review_Yield": round(metrics["review_yield"] * 100, 2)
        })

    ablation_df = pd.DataFrame(ablation_rows)

    return b10_df, budget_df, ablation_df


if __name__ == "__main__":
    extracted_csv = "DefectVisionAI/research/cpnd/dev_extracted_scores.csv"
    if not os.path.exists(extracted_csv):
        print("Extracting features and scoring development partition...", flush=True)
        df_dev = extract_dev_features_and_scores(output_csv=extracted_csv)
    else:
        print(f"Loading existing extracted scores from {extracted_csv}...", flush=True)
        df_dev = pd.read_csv(extracted_csv)

    b10_df, budget_df, ablation_df = run_full_benchmark(df_dev)

    # Save deliverables
    out_dir = "research/phase7b_cpnd_development"
    os.makedirs(out_dir, exist_ok=True)

    b10_df.to_csv(os.path.join(out_dir, "DEVELOPMENT_BASELINE_COMPARISON.csv"), index=False)
    budget_df.to_csv(os.path.join(out_dir, "REVIEW_BUDGET_RESULTS.csv"), index=False)
    ablation_df.to_csv(os.path.join(out_dir, "ABLATION_RESULTS.csv"), index=False)

    print("\nResults successfully saved to research/phase7b_cpnd_development/!", flush=True)
    print("\n--- BASELINE COMPARISON AT 10% BUDGET ---", flush=True)
    print(b10_df[["Policy", "Reviewed_K", "Captured_Errors", "ECR", "DER", "Review_Yield"]].to_string(), flush=True)
    print("\n--- ABLATION COMPARISON AT 10% BUDGET ---", flush=True)
    print(ablation_df[["Ablation_Experiment", "Captured_Errors", "ECR_at_10pct", "DER_at_10pct"]].to_string(), flush=True)
