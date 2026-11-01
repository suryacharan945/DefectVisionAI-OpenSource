"""
analyze_phase8c_equivalence.py
Exhaustive empirical analysis and decision trace generation for Phase 8C MACBA Equivalence Audit.
Directly reproduces Phase 8B setup, calibration, and allocations.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression

SEED = 42

def fit_calibration_models(cal_df: pd.DataFrame) -> Dict[str, Any]:
    models = {}
    categories = cal_df["category"].unique().tolist()
    
    cal_priors = {}
    for cat in categories:
        sub = cal_df[cal_df["category"] == cat]
        cal_priors[cat] = float(sub["is_error"].mean())

    y_defect = (cal_df["true_label"].str.lower().isin(["defective", "anomaly", "1"])).astype(int).to_numpy()
    x_score = (cal_df["anomaly_score"] - cal_df["decision_threshold"]).to_numpy().reshape(-1, 1)

    defect_clf = LogisticRegression(C=1.0, random_state=SEED)
    defect_clf.fit(x_score, y_defect)
    models["defect_clf"] = defect_clf

    y_error = cal_df["is_error"].to_numpy().astype(int)
    x_dist = cal_df["dist_to_threshold"].to_numpy().reshape(-1, 1)

    error_clf = LogisticRegression(C=1.0, random_state=SEED)
    error_clf.fit(x_dist, y_error)
    models["error_clf"] = error_clf

    models["cal_priors"] = cal_priors
    return models


def run_equivalence_audit():
    data_csv = "DefectVisionAI/research/cpnd/dev_extracted_scores_phase7c.csv"
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Missing {data_csv}")

    df = pd.read_csv(data_csv)
    df["dist_to_threshold"] = (df["anomaly_score"] - df["decision_threshold"]).abs()
    df["defect_type"] = df["filepath"].apply(lambda x: x.split('/')[-2] if '/' in x else x.split('\\')[-2])
    
    # 1. Stratified Group-Aware Split into Calibration (50%) and Internal Evaluation (50%)
    RNG = np.random.default_rng(SEED)
    df["strata"] = df["category"] + "_" + df["defect_type"] + "_" + df["is_error"].astype(str)
    cal_indices = []
    eval_indices = []

    for _, grp in df.groupby("strata"):
        g_idx = list(grp.index)
        RNG.shuffle(g_idx)
        half = len(g_idx) // 2
        cal_indices.extend(g_idx[:half])
        eval_indices.extend(g_idx[half:])

    cal_df = df.loc[cal_indices].copy().reset_index(drop=True)
    eval_df = df.loc[eval_indices].copy().reset_index(drop=True)

    print(f"Calibration Fold: {len(cal_df)} images, {cal_df['is_error'].sum()} errors ({cal_df['is_fn'].sum()} FN, {cal_df['is_fp'].sum()} FP)")
    print(f"Evaluation Fold:  {len(eval_df)} images, {eval_df['is_error'].sum()} errors ({eval_df['is_fn'].sum()} FN, {eval_df['is_fp'].sum()} FP)")

    cal_models = fit_calibration_models(cal_df)

    # Predictions on eval_df
    x_eval_score = (eval_df["anomaly_score"] - eval_df["decision_threshold"]).to_numpy().reshape(-1, 1)
    p_defect = cal_models["defect_clf"].predict_proba(x_eval_score)[:, 1]

    x_eval_dist = eval_df["dist_to_threshold"].to_numpy().reshape(-1, 1)
    p_error_unsigned = cal_models["error_clf"].predict_proba(x_eval_dist)[:, 1]

    y_hat = (eval_df["anomaly_score"] >= eval_df["decision_threshold"]).astype(int).to_numpy()
    
    # Ground truth defect label (1 for defect, 0 for normal)
    gt_defect = (eval_df["true_label"].str.lower().isin(["defective", "anomaly", "1"])).astype(int).to_numpy()

    # Signed P(Error | X, y_hat)
    p_error_signed = np.where(y_hat == 0, p_defect, 1.0 - p_defect)

    eval_df["y_hat"] = y_hat
    eval_df["gt_defect"] = gt_defect
    eval_df["p_defect_est"] = p_defect
    eval_df["p_error_unsigned"] = p_error_unsigned
    eval_df["p_error_signed"] = p_error_signed

    N = len(eval_df)
    c_rev = 1.0
    p_sens = 1.0
    p_spec = 1.0

    cost_ratios = [1, 5, 20, 50, 100]
    budgets = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    primary_budget = 25

    records = eval_df.to_dict(orient="records")

    trace_rows = []
    summary_rows = []

    for lambda_val in cost_ratios:
        c_fp = 10.0
        c_fn = 10.0 * lambda_val

        # Precompute scores for each specimen
        for i in range(N):
            r = records[i]
            y_h = r["y_hat"]
            p_d = r["p_defect_est"]
            p_e_uns = r["p_error_unsigned"]
            p_e_sgn = r["p_error_signed"]

            # MACBA delta L
            if y_h == 0:
                delta_l = p_d * p_sens * c_fn - (1.0 - p_d) * (1.0 - p_spec) * c_fp - c_rev
                bl6_weight = c_fn
            else:
                delta_l = (1.0 - p_d) * p_spec * c_fp - p_d * (1.0 - p_sens) * c_fn - c_rev
                bl6_weight = c_fp

            r["delta_loss_est"] = delta_l
            r["bl6_score_phase8b"] = p_e_uns * bl6_weight
            r["bl6_score_proper"] = p_e_sgn * bl6_weight

        # Sort keys
        sorted_macba = sorted(range(N), key=lambda i: (-records[i]["delta_loss_est"] / c_rev, records[i]["image_id"]))
        sorted_bl6_8b = sorted(range(N), key=lambda i: (-records[i]["bl6_score_phase8b"], records[i]["image_id"]))
        sorted_bl6_proper = sorted(range(N), key=lambda i: (-records[i]["bl6_score_proper"], records[i]["image_id"]))

        macba_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_macba)}
        bl6_8b_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_bl6_8b)}
        bl6_proper_ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_bl6_proper)}

        # Evaluate across budgets
        for K in budgets:
            sel_macba = set(sorted_macba[:K])
            sel_bl6_8b = set(sorted_bl6_8b[:K])
            sel_bl6_proper = set(sorted_bl6_proper[:K])

            # Selection overlap
            overlap_8b = len(sel_macba.intersection(sel_bl6_8b))
            overlap_proper = len(sel_macba.intersection(sel_bl6_proper))

            # Realized cost calculations
            def calc_realized_cost(selected_indices):
                sel_set = set(selected_indices)
                total_cost = 0.0
                captured_fn = 0
                captured_fp = 0
                for i in range(N):
                    r = records[i]
                    gt = r["gt_defect"]
                    y_h = r["y_hat"]
                    if i in sel_set:
                        total_cost += c_rev
                        if y_h != gt:
                            if y_h == 0 and gt == 1:
                                captured_fn += 1
                            elif y_h == 1 and gt == 0:
                                captured_fp += 1
                    else:
                        if y_h == 0 and gt == 1:
                            total_cost += c_fn
                        elif y_h == 1 and gt == 0:
                            total_cost += c_fp
                return total_cost, captured_fn, captured_fp

            cost_unassisted, _, _ = calc_realized_cost([])
            cost_macba, fn_macba, fp_macba = calc_realized_cost(sorted_macba[:K])
            cost_bl6_8b, fn_bl6_8b, fp_bl6_8b = calc_realized_cost(sorted_bl6_8b[:K])
            cost_bl6_proper, fn_bl6_proper, fp_bl6_proper = calc_realized_cost(sorted_bl6_proper[:K])

            summary_rows.append({
                "lambda": lambda_val,
                "budget_K": K,
                "overlap_8b_count": overlap_8b,
                "overlap_8b_pct": 100.0 * overlap_8b / K,
                "overlap_proper_count": overlap_proper,
                "overlap_proper_pct": 100.0 * overlap_proper / K,
                "cost_unassisted": cost_unassisted,
                "cost_macba": cost_macba,
                "cost_bl6_8b": cost_bl6_8b,
                "cost_bl6_proper": cost_bl6_proper,
                "red_pct_macba": 100.0 * (cost_unassisted - cost_macba) / cost_unassisted,
                "red_pct_bl6_8b": 100.0 * (cost_unassisted - cost_bl6_8b) / cost_unassisted,
                "red_pct_bl6_proper": 100.0 * (cost_unassisted - cost_bl6_proper) / cost_unassisted,
                "captured_fn_macba": fn_macba,
                "captured_fn_bl6_8b": fn_bl6_8b,
                "captured_fn_bl6_proper": fn_bl6_proper
            })

            # For the decision trace export, record primary budget K=25
            if K == primary_budget:
                for i in range(N):
                    r = records[i]
                    is_in_macba = i in sel_macba
                    is_in_bl6_8b = i in sel_bl6_8b
                    is_in_bl6_proper = i in sel_bl6_proper
                    
                    discrepancy_reason = "Identical selection"
                    if is_in_macba != is_in_bl6_8b:
                        discrepancy_reason = "Phase 8B unsigned error_clf vs signed defect_clf"
                    
                    trace_rows.append({
                        "lambda_ratio": lambda_val,
                        "review_budget_k": K,
                        "image_id": r["image_id"],
                        "category": r["category"],
                        "anomaly_score": r["anomaly_score"],
                        "decision_threshold": r["decision_threshold"],
                        "dist_to_threshold": r["dist_to_threshold"],
                        "automated_prediction": r["y_hat"],
                        "ground_truth": r["gt_defect"],
                        "is_automated_error": int(r["y_hat"] != r["gt_defect"]),
                        "p_defect_calibrated": round(r["p_defect_est"], 6),
                        "p_error_unsigned_phase8b": round(r["p_error_unsigned"], 6),
                        "p_error_signed_proper": round(r["p_error_signed"], 6),
                        "expected_review_benefit_macba": round(r["delta_loss_est"], 4),
                        "bl6_score_phase8b": round(r["bl6_score_phase8b"], 4),
                        "bl6_score_proper": round(r["bl6_score_proper"], 4),
                        "review_capacity_cost": c_rev,
                        "macba_rank": macba_ranks[i],
                        "macba_selected": int(is_in_macba),
                        "bl6_phase8b_rank": bl6_8b_ranks[i],
                        "bl6_phase8b_selected": int(is_in_bl6_8b),
                        "bl6_proper_rank": bl6_proper_ranks[i],
                        "bl6_proper_selected": int(is_in_bl6_proper),
                        "discrepancy_note": discrepancy_reason
                    })

    df_summary = pd.DataFrame(summary_rows)
    df_trace = pd.DataFrame(trace_rows)

    os.makedirs("research/phase8c_macba_equivalence", exist_ok=True)
    df_trace.to_csv("research/phase8c_macba_equivalence/BL6_MACBA_DECISION_TRACE.csv", index=False)
    print("Exported BL6_MACBA_DECISION_TRACE.csv successfully.")

    # Print key summary findings
    print("\n=== SUMMARY OVERLAP AND REALIZED COSTS AT PRIMARY BUDGET K=25 ===")
    sub_k25 = df_summary[df_summary["budget_K"] == 25]
    print(sub_k25[["lambda", "overlap_8b_count", "overlap_8b_pct", "overlap_proper_count", "overlap_proper_pct", "red_pct_macba", "red_pct_bl6_8b", "red_pct_bl6_proper"]].to_string(index=False))

    print("\n=== SUMMARY ACROSS ALL BUDGETS AT LAMBDA = 1 ===")
    sub_l1 = df_summary[df_summary["lambda"] == 1]
    print(sub_l1[["budget_K", "overlap_8b_count", "overlap_8b_pct", "overlap_proper_count", "overlap_proper_pct", "red_pct_macba", "red_pct_bl6_8b", "red_pct_bl6_proper"]].to_string(index=False))

    print("\n=== SUMMARY ACROSS ALL BUDGETS AT LAMBDA = 5 ===")
    sub_l5 = df_summary[df_summary["lambda"] == 5]
    print(sub_l5[["budget_K", "overlap_8b_count", "overlap_8b_pct", "overlap_proper_count", "overlap_proper_pct", "red_pct_macba", "red_pct_bl6_8b", "red_pct_bl6_proper"]].to_string(index=False))

    print("\n=== SUMMARY ACROSS ALL BUDGETS AT LAMBDA = 20 ===")
    sub_l20 = df_summary[df_summary["lambda"] == 20]
    print(sub_l20[["budget_K", "overlap_8b_count", "overlap_8b_pct", "overlap_proper_count", "overlap_proper_pct", "red_pct_macba", "red_pct_bl6_8b", "red_pct_bl6_proper"]].to_string(index=False))

    # Also save df_summary for reference
    df_summary.to_csv("research/phase8c_macba_equivalence/SUMMARY_EQUIVALENCE_METRICS.csv", index=False)

if __name__ == "__main__":
    run_equivalence_audit()
