
import numpy as np

def normalize_anomaly_score(raw_score, baseline_mean, baseline_p99):
    """Maps a raw per-category anomaly score onto a comparable 0-100 scale
    using that category's OWN normal-image score distribution as the reference frame."""
    span = max(baseline_p99 - baseline_mean, 1e-6)
    normed = (raw_score - baseline_mean) / span
    normed = np.clip(normed, 0, 1.5)  # allow some headroom above p99 for genuinely severe cases
    return float(np.clip(normed * 100, 0, 100))


def compute_severity_score(anomaly_norm, defect_area_pct, confidence_pct,
                            w_anomaly=0.5, w_area=0.3, w_confidence=0.2):
    """Weighted fusion of three independent signals into one 0-100 severity score."""
    area_capped = min(defect_area_pct, 100.0)
    score = (w_anomaly * anomaly_norm) + (w_area * area_capped) + (w_confidence * confidence_pct)
    return float(np.clip(score, 0, 100))


def bucket_severity(score):
    if score < 34:
        return "Minor"
    elif score < 67:
        return "Moderate"
    else:
        return "Critical"
