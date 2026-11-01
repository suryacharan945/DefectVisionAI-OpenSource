"""
DefectVision AI — Phase 7C Controlled Experiment Runner.
Executes density-normalized CPND, boundary-gating, partitioning, paired bootstrap,
and comprehensive ablations on the authorized 330-image development partition.
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
from typing import Dict, List, Tuple, Optional, Any

# Paths
sys.path.insert(0, os.path.abspath("DefectVisionAI"))
from src.patchcore import PatchCoreExtractor
from research.cpnd.cpnd_density_norm import (
    DensityNormalizedCPNDScorer,
    rank_and_select_budget,
    evaluate_selective_review
)

DEVICE = torch.device("cpu")
PC_IMG_SIZE = 224
PC_MEAN = np.array([0.485, 0.456, 0.406])
PC_STD = np.array([0.229, 0.224, 0.225])
BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
CATEGORIES = ["bottle", "capsule", "metal_nut"]


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


def paired_bootstrap_evaluation(
    records: List[Dict[str, Any]],
    baseline_key: str,
    candidate_key: str,
    budget: float,
    baseline_tier_key: Optional[str] = None,
    candidate_tier_key: Optional[str] = None,
    n_bootstrap: int = 2000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes paired bootstrap confidence intervals for:
      Delta_ECR = ECR_candidate - ECR_baseline
      Delta_DER = DER_candidate - DER_baseline
    """
    rng = np.random.default_rng(seed)
    N = len(records)
    
    # Baseline and candidate point estimates
    sel_base, _ = rank_and_select_budget(records, baseline_key, budget, gating_tier_key=baseline_tier_key)
    res_base = evaluate_selective_review(records, sel_base)
    
    sel_cand, _ = rank_and_select_budget(records, candidate_key, budget, gating_tier_key=candidate_tier_key)
    res_cand = evaluate_selective_review(records, sel_cand)

    ecr_base = res_base["error_capture_rate"]
    der_base = res_base["defect_escape_reduction"]
    ecr_cand = res_cand["error_capture_rate"]
    der_cand = res_cand["defect_escape_reduction"]

    delta_ecr = ecr_cand - ecr_base
    delta_der = der_cand - der_base

    delta_ecrs = []
    delta_ders = []
    ecr_cand_list = []
    der_cand_list = []

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(N, size=N, replace=True)
        sample = [records[i] for i in sample_indices]

        # Evaluate on sample
        s_base, _ = rank_and_select_budget(sample, baseline_key, budget, gating_tier_key=baseline_tier_key)
        m_base = evaluate_selective_review(sample, s_base)

        s_cand, _ = rank_and_select_budget(sample, candidate_key, budget, gating_tier_key=candidate_tier_key)
        m_cand = evaluate_selective_review(sample, s_cand)

        d_ecr = m_cand["error_capture_rate"] - m_base["error_capture_rate"]
        d_der = m_cand["defect_escape_reduction"] - m_base["defect_escape_reduction"]

        delta_ecrs.append(d_ecr)
        delta_ders.append(d_der)
        ecr_cand_list.append(m_cand["error_capture_rate"])
        der_cand_list.append(m_cand["defect_escape_reduction"])

    return {
        "ecr_cand": ecr_cand,
        "ecr_cand_ci_low": float(np.percentile(ecr_cand_list, 2.5)),
        "ecr_cand_ci_high": float(np.percentile(ecr_cand_list, 97.5)),
        "der_cand": der_cand,
        "der_cand_ci_low": float(np.percentile(der_cand_list, 2.5)),
        "der_cand_ci_high": float(np.percentile(der_cand_list, 97.5)),
        "delta_ecr": delta_ecr,
        "delta_ecr_ci_low": float(np.percentile(delta_ecrs, 2.5)),
        "delta_ecr_ci_high": float(np.percentile(delta_ecrs, 97.5)),
        "delta_der": delta_der,
        "delta_der_ci_low": float(np.percentile(delta_ders, 2.5)),
        "delta_der_ci_high": float(np.percentile(delta_ders, 97.5)),
        "defect_review_yield": res_cand["defect_review_yield"],
        "error_review_yield": res_cand["error_review_yield"],
        "reviewed_k": res_cand["reviewed_images"],
        "captured_errors": res_cand["captured_errors"],
        "captured_fn": res_cand["captured_fn"],
        "captured_fp": res_cand["captured_fp"]
    }


def main():
    print("================ DEFECTVISION AI: PHASE 7C CONTROLLED BENCHMARK ================", flush=True)
    t_start = time.perf_counter()

    # Step 1: Load pre-existing Phase 7B development predictions
    dev_csv = "DefectVisionAI/research/cpnd/dev_extracted_scores.csv"
    if not os.path.exists(dev_csv):
        raise FileNotFoundError(f"Missing {dev_csv}")
    
    df = pd.read_csv(dev_csv)
    print(f"Loaded {len(df)} development images across categories: {df['category'].unique().tolist()}", flush=True)

    # Check if pre-extracted Phase 7C scores already exist
    extracted_csv = "DefectVisionAI/research/cpnd/dev_extracted_scores_phase7c.csv"
    if os.path.exists(extracted_csv):
        print(f"\n--- Loading pre-extracted Phase 7C scores from {extracted_csv} ---", flush=True)
        extracted_df = pd.read_csv(extracted_csv)
        records = extracted_df.to_dict(orient="records")
    else:
        # Step 2: Initialize Density-Normalized Scorers for each category
        print("\n--- Initializing Density-Normalized Memory Bank Scorers ---", flush=True)
        scorers = {}
        for cat in CATEGORIES:
            mb_path = f"DefectVisionAI/models/patchcore/{cat}/memory_bank.npy"
            mb = np.ascontiguousarray(np.load(mb_path), dtype=np.float32)
            
            scorers[f"{cat}_k5_mad"] = DensityNormalizedCPNDScorer(mb, k=5, ref_k=5, normalization_type="median_mad")
            scorers[f"{cat}_k5_std"] = DensityNormalizedCPNDScorer(mb, k=5, ref_k=5, normalization_type="mean_std")
            scorers[f"{cat}_k10_mad"] = DensityNormalizedCPNDScorer(mb, k=10, ref_k=10, normalization_type="median_mad")
            print(f"  [{cat}] Memory Bank: {len(mb)} vectors. Precomputed nominal reference dispersions.", flush=True)

        # Step 3: Extract Patch Features and Compute Density-Normalized Scores
        print("\n--- Extracting Image Features and Scoring ---", flush=True)
        extractor = PatchCoreExtractor().to(DEVICE)
        extractor.eval()

        records = []
        for idx, row in df.iterrows():
            cat = row["category"]
            tau = float(row["decision_threshold"])
            fp = row["filepath"]
            
            t_img = load_pc_tensor(fp)
            if t_img is None:
                raise FileNotFoundError(f"Could not load image: {fp}")

            with torch.no_grad():
                feat = extractor(t_img)[0].numpy()
            C, g1, g2 = feat.shape
            patches = np.ascontiguousarray(feat.reshape(C, g1 * g2).T, dtype=np.float32)

            # Scorer lookups
            s_k5_mad = scorers[f"{cat}_k5_mad"]
            s_k5_std = scorers[f"{cat}_k5_std"]
            s_k10_mad = scorers[f"{cat}_k10_mad"]

            res_k5_mad = s_k5_mad.score_patches(patches, k=5)
            res_k5_std = s_k5_std.score_patches(patches, k=5)
            res_k10_mad = s_k10_mad.score_patches(patches, k=10)

            # Argmax patch index
            nn1 = res_k5_mad["nn1_distances"]
            argmax_idx = int(np.argmax(nn1))
            anomaly_score = float(nn1[argmax_idx])

            # Distance to threshold
            dist_to_thresh = float(abs(anomaly_score - tau))
            u_dist = -dist_to_thresh

            # Dispersions for argmax patch
            raw_pnd_k5 = float(res_k5_mad["raw_dispersion"][argmax_idx])
            norm_pnd_k5_mad = float(res_k5_mad["norm_dispersion"][argmax_idx])
            norm_pnd_k5_std = float(res_k5_std["norm_dispersion"][argmax_idx])

            raw_pnd_k10 = float(res_k10_mad["raw_dispersion"][argmax_idx])
            norm_pnd_k10_mad = float(res_k10_mad["norm_dispersion"][argmax_idx])

            # Boundary gating tiers (delta as fraction of threshold tau)
            in_band_005 = 1 if dist_to_thresh <= (0.05 * tau) else 0
            in_band_010 = 1 if dist_to_thresh <= (0.10 * tau) else 0
            in_band_015 = 1 if dist_to_thresh <= (0.15 * tau) else 0
            in_band_020 = 1 if dist_to_thresh <= (0.20 * tau) else 0

            # Soft Gaussian gating: exp(- dist^2 / (2 * delta^2)) * norm_pnd
            u_soft_010 = float(np.exp(- (dist_to_thresh ** 2) / (2.0 * ((0.10 * tau) ** 2) + 1e-7))) * norm_pnd_k5_mad
            u_soft_015 = float(np.exp(- (dist_to_thresh ** 2) / (2.0 * ((0.15 * tau) ** 2) + 1e-7))) * norm_pnd_k5_mad

            rec = {
                "image_id": row["image_id"],
                "category": cat,
                "filepath": fp,
                "true_label": row["true_label"],
                "decision_threshold": tau,
                "anomaly_score": anomaly_score,
                "predicted_label": row["predicted_label"],
                "is_error": int(row["is_error"]),
                "is_fn": int(row["is_fn"]),
                "is_fp": int(row["is_fp"]),
                "u_dist": u_dist,
                "raw_pnd_k5": raw_pnd_k5,
                "norm_pnd_k5_mad": norm_pnd_k5_mad,
                "norm_pnd_k5_std": norm_pnd_k5_std,
                "raw_pnd_k10": raw_pnd_k10,
                "norm_pnd_k10_mad": norm_pnd_k10_mad,
                "in_band_005": in_band_005,
                "in_band_010": in_band_010,
                "in_band_015": in_band_015,
                "in_band_020": in_band_020,
                "u_soft_010": u_soft_010,
                "u_soft_015": u_soft_015
            }
            records.append(rec)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
                elapsed = time.perf_counter() - t_start
                rate = (idx + 1) / elapsed
                print(f"  Processed {idx + 1}/{len(df)} images ({rate:.1f} img/s)...", flush=True)

        extracted_df = pd.DataFrame(records)
        extracted_df.to_csv(extracted_csv, index=False)
        print(f"Scoring extraction complete! Saved to {extracted_csv}", flush=True)

    # Step 4: Stratified Partitioning (50% Tune, 50% Internal Eval)
    print("\n--- Partitioning Development Data (50% Tune / 50% Internal Eval) ---", flush=True)
    rng = np.random.default_rng(42)
    # Stratify by (category, is_error)
    extracted_df["strata"] = extracted_df["category"] + "_" + extracted_df["is_error"].astype(str)
    tune_indices = []
    eval_indices = []

    for strata_name, group in extracted_df.groupby("strata"):
        g_indices = list(group.index)
        rng.shuffle(g_indices)
        half = len(g_indices) // 2
        tune_indices.extend(g_indices[:half])
        eval_indices.extend(g_indices[half:])

    tune_records = [records[i] for i in tune_indices]
    eval_records = [records[i] for i in eval_indices]

    tune_errors = sum(r["is_error"] for r in tune_records)
    eval_errors = sum(r["is_error"] for r in eval_records)
    print(f"  Partition 1 (Tune): {len(tune_records)} images, {tune_errors} errors.", flush=True)
    print(f"  Partition 2 (Eval): {len(eval_records)} images, {eval_errors} errors.", flush=True)

    # Step 5: Hyperparameter Tuning on Partition 1 (Tune Only)
    print("\n--- Hyperparameter Tuning on Partition 1 (Tune Only) ---", flush=True)
    delta_options = [
        ("delta_005", "in_band_005", "norm_pnd_k5_mad"),
        ("delta_010", "in_band_010", "norm_pnd_k5_mad"),
        ("delta_015", "in_band_015", "norm_pnd_k5_mad"),
        ("delta_020", "in_band_020", "norm_pnd_k5_mad"),
        ("soft_010", None, "u_soft_010"),
        ("soft_015", None, "u_soft_015")
    ]

    tune_results = []
    # Baseline on Tune
    sel_b, _ = rank_and_select_budget(tune_records, "u_dist", budget_fraction=0.10)
    res_b = evaluate_selective_review(tune_records, sel_b)
    print(f"  [Tune Baseline BL2 Distance]: ECR = {res_b['error_capture_rate']*100:.2f}%, Captured = {res_b['captured_errors']}/{tune_errors}")

    for name, tier_k, unc_k in delta_options:
        sel_c, _ = rank_and_select_budget(tune_records, unc_k, budget_fraction=0.10, gating_tier_key=tier_k)
        res_c = evaluate_selective_review(tune_records, sel_c)
        tune_results.append({
            "name": name,
            "tier_key": tier_k,
            "unc_key": unc_k,
            "captured_errors": res_c["captured_errors"],
            "ecr": res_c["error_capture_rate"]
        })
        print(f"  Candidate {name:12s}: ECR = {res_c['error_capture_rate']*100:.2f}%, Captured = {res_c['captured_errors']}/{tune_errors}")

    # Best candidate on Tune
    best_tune = max(tune_results, key=lambda x: x["ecr"])
    best_tier_key = best_tune["tier_key"]
    best_unc_key = best_tune["unc_key"]
    best_candidate_name = f"Boundary-Gated CPND ({best_tune['name']})"
    print(f"\nWinning Tuned Configuration: {best_candidate_name} (ECR={best_tune['ecr']*100:.2f}%)", flush=True)

    # Step 6: Internal Evaluation on Partition 2 (Held-Out)
    print("\n--- Internal Evaluation on Partition 2 (Held-Out Evaluation) ---", flush=True)
    eval_policies = [
        ("Distance-to-Threshold (BL2)", "u_dist", None),
        ("Raw CPND (k=5)", "raw_pnd_k5", None),
        ("Density-Normalized CPND (Ungated)", "norm_pnd_k5_mad", None),
        (best_candidate_name, best_unc_key, best_tier_key)
    ]

    for pol_name, unc_k, tier_k in eval_policies:
        sel, _ = rank_and_select_budget(eval_records, unc_k, budget_fraction=0.10, gating_tier_key=tier_k)
        ev_res = evaluate_selective_review(eval_records, sel)
        print(f"  {pol_name:38s}: ECR={ev_res['error_capture_rate']*100:.2f}%, DER={ev_res['defect_escape_reduction']*100:.2f}%, Errors={ev_res['captured_errors']}/{eval_errors}")

    # Step 7: Paired Bootstrap Comparison on Full Development Partition (330 Images)
    print("\n--- Computing Paired Bootstrap Comparison on Full Development Partition ---", flush=True)
    paired_rows = []
    target_policies = [
        ("Original Raw CPND (k=5)", "raw_pnd_k5", None),
        ("Density-Normalized CPND (Ungated)", "norm_pnd_k5_mad", None),
        ("Boundary-Gated Hard Tiered (delta=0.10*tau)", "norm_pnd_k5_mad", "in_band_010"),
        ("Boundary-Gated Hard Tiered (delta=0.15*tau)", "norm_pnd_k5_mad", "in_band_015"),
        ("Boundary-Gated Soft Gaussian (delta=0.10*tau)", "u_soft_010", None)
    ]

    for pol_name, unc_k, tier_k in target_policies:
        boot_res = paired_bootstrap_evaluation(
            records=records,
            baseline_key="u_dist",
            candidate_key=unc_k,
            budget=0.10,
            baseline_tier_key=None,
            candidate_tier_key=tier_k,
            n_bootstrap=2000,
            seed=42
        )
        paired_rows.append({
            "Candidate_Method": pol_name,
            "Budget": 0.10,
            "Reviewed_K": boot_res["reviewed_k"],
            "Captured_Errors": boot_res["captured_errors"],
            "Captured_FN": boot_res["captured_fn"],
            "Captured_FP": boot_res["captured_fp"],
            "Candidate_ECR": round(boot_res["ecr_cand"] * 100, 2),
            "Candidate_ECR_95CI": f"[{boot_res['ecr_cand_ci_low']*100:.1f}%, {boot_res['ecr_cand_ci_high']*100:.1f}%]",
            "Baseline_BL2_ECR": 29.03,  # Fixed baseline point estimate
            "Delta_ECR": round(boot_res["delta_ecr"] * 100, 2),
            "Delta_ECR_95CI": f"[{boot_res['delta_ecr_ci_low']*100:.1f}%, {boot_res['delta_ecr_ci_high']*100:.1f}%]",
            "Candidate_DER": round(boot_res["der_cand"] * 100, 2),
            "Candidate_DER_95CI": f"[{boot_res['der_cand_ci_low']*100:.1f}%, {boot_res['der_cand_ci_high']*100:.1f}%]",
            "Baseline_BL2_DER": 25.42,
            "Delta_DER": round(boot_res["delta_der"] * 100, 2),
            "Delta_DER_95CI": f"[{boot_res['delta_der_ci_low']*100:.1f}%, {boot_res['delta_der_ci_high']*100:.1f}%]",
            "Defect_Review_Yield": round(boot_res["defect_review_yield"] * 100, 2),
            "Error_Review_Yield": round(boot_res["error_review_yield"] * 100, 2)
        })

    paired_df = pd.DataFrame(paired_rows)
    os.makedirs("research/phase7c_density_normalized_cpnd", exist_ok=True)
    paired_df.to_csv("research/phase7c_density_normalized_cpnd/PAIRED_COMPARISON.csv", index=False)
    print("\n--- PAIRED COMPARISON AT 10% BUDGET ---", flush=True)
    print(paired_df[["Candidate_Method", "Captured_Errors", "Candidate_ECR", "Delta_ECR", "Delta_ECR_95CI", "Delta_DER", "Delta_DER_95CI"]].to_string(), flush=True)

    # Step 8: Multi-Budget Sweep across 1%, 2%, 5%, 10%, 15%, 20%
    print("\n--- Multi-Budget Sweep ---", flush=True)
    sweep_rows = []
    sweep_methods = [
        ("Distance-to-Threshold (BL2)", "u_dist", None),
        ("Original Raw CPND (k=5)", "raw_pnd_k5", None),
        ("Density-Normalized CPND (Ungated)", "norm_pnd_k5_mad", None),
        ("Boundary-Gated Hard Tiered (delta=0.10*tau)", "norm_pnd_k5_mad", "in_band_010"),
        ("Boundary-Gated Soft Gaussian (delta=0.10*tau)", "u_soft_010", None)
    ]

    for b in BUDGETS:
        for pol_name, unc_k, tier_k in sweep_methods:
            sel, k = rank_and_select_budget(records, unc_k, budget_fraction=b, gating_tier_key=tier_k)
            res = evaluate_selective_review(records, sel)
            sweep_rows.append({
                "Method": pol_name,
                "Budget": b,
                "Reviewed_K": k,
                "Captured_Errors": res["captured_errors"],
                "Captured_FN": res["captured_fn"],
                "Captured_FP": res["captured_fp"],
                "ECR": round(res["error_capture_rate"] * 100, 2),
                "DER": round(res["defect_escape_reduction"] * 100, 2),
                "Defect_Review_Yield": round(res["defect_review_yield"] * 100, 2),
                "Error_Review_Yield": round(res["error_review_yield"] * 100, 2)
            })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv("research/phase7c_density_normalized_cpnd/BUDGET_SWEEP.csv", index=False)
    print("Saved BUDGET_SWEEP.csv", flush=True)

    # Step 9: Ablation Study
    print("\n--- Ablation Study ---", flush=True)
    ablation_experiments = [
        ("Ablation 1A: Raw Dispersion (k=5)", "raw_pnd_k5", None),
        ("Ablation 1B: Density-Normalized Dispersion (k=5, MAD)", "norm_pnd_k5_mad", None),
        ("Ablation 2A: Boundary Gating Off (Ungated Norm PND)", "norm_pnd_k5_mad", None),
        ("Ablation 2B: Hard Tiered Gating (delta=0.10*tau)", "norm_pnd_k5_mad", "in_band_010"),
        ("Ablation 2C: Hard Tiered Gating (delta=0.15*tau)", "norm_pnd_k5_mad", "in_band_015"),
        ("Ablation 2D: Soft Gaussian Gating (delta=0.10*tau)", "u_soft_010", None),
        ("Ablation 3A: Neighborhood Size k=5 (Norm PND)", "norm_pnd_k5_mad", None),
        ("Ablation 3B: Neighborhood Size k=10 (Norm PND)", "norm_pnd_k10_mad", None),
        ("Ablation 4A: Normalization Robust Median/MAD", "norm_pnd_k5_mad", None),
        ("Ablation 4B: Normalization Parametric Mean/STD", "norm_pnd_k5_std", None)
    ]

    ablation_rows = []
    for ab_name, unc_k, tier_k in ablation_experiments:
        sel, k = rank_and_select_budget(records, unc_k, budget_fraction=0.10, gating_tier_key=tier_k)
        res = evaluate_selective_review(records, sel)
        ablation_rows.append({
            "Ablation_Experiment": ab_name,
            "Reviewed_K": k,
            "Captured_Errors": res["captured_errors"],
            "Captured_FN": res["captured_fn"],
            "Captured_FP": res["captured_fp"],
            "ECR_at_10pct": round(res["error_capture_rate"] * 100, 2),
            "DER_at_10pct": round(res["defect_escape_reduction"] * 100, 2),
            "Defect_Review_Yield": round(res["defect_review_yield"] * 100, 2),
            "Error_Review_Yield": round(res["error_review_yield"] * 100, 2)
        })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv("research/phase7c_density_normalized_cpnd/ABLATION_RESULTS.csv", index=False)
    print("\n--- ABLATION RESULTS AT 10% BUDGET ---", flush=True)
    print(ablation_df[["Ablation_Experiment", "Captured_Errors", "ECR_at_10pct", "DER_at_10pct", "Error_Review_Yield"]].to_string(), flush=True)

    total_time = time.perf_counter() - t_start
    print(f"\nAll Phase 7C experiments completed successfully in {total_time:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
