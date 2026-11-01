
import os
import pandas as pd
import numpy as np
from datetime import datetime

def build_batch_report(results_df, batch_id):
    """Aggregate per-image results into a batch-level summary dict."""
    total = len(results_df)
    defective = int((results_df["predicted_label"] == "defective").sum())
    normal = total - defective

    severity_counts = results_df["severity_level"].value_counts().to_dict()

    category_breakdown = []
    for category, group in results_df.groupby("category"):
        cat_total = len(group)
        cat_defective = int((group["predicted_label"] == "defective").sum())
        category_breakdown.append({
            "category": category,
            "total": cat_total,
            "defective": cat_defective,
            "defect_rate_pct": round(100 * cat_defective / cat_total, 2) if cat_total else 0.0,
        })

    report = {
        "batch_id": batch_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_inspected": total,
        "defective_count": defective,
        "normal_count": normal,
        "defect_rate_pct": round(100 * defective / total, 2) if total else 0.0,
        "severity_distribution": severity_counts,
        "category_breakdown": category_breakdown,
    }
    return report


def detect_category_from_filename_or_default(filepath, known_categories, default=None):
    """Best-effort category guess for uploaded images with no metadata:
    checks if any known category name appears in the filename/path."""
    lower_path = filepath.lower()
    for cat in known_categories:
        if cat.lower() in lower_path:
            return cat
    return default
