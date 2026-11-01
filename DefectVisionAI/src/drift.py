
import numpy as np

def compute_reference_stats(embeddings):
    mean_vec = embeddings.mean(axis=0)
    centered = embeddings - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    top_direction = Vt[0]
    ref_projection = centered @ top_direction
    ref_scale = np.linalg.norm(centered, axis=1).mean()
    return {
        "mean_vec": mean_vec,
        "top_direction": top_direction,
        "ref_projection": ref_projection,
        "ref_scale": ref_scale,
        "ref_proj_mean": float(ref_projection.mean()),
        "ref_proj_std": float(ref_projection.std() + 1e-6),
    }


def embedding_distance_score(embeddings, ref_stats):
    dists = np.linalg.norm(embeddings - ref_stats["mean_vec"], axis=1)
    normalized = dists.mean() / max(ref_stats["ref_scale"], 1e-8)
    return float(normalized)


def gaussian_kl(mean_p, std_p, mean_q, std_q):
    """Closed-form KL(P || Q) between two 1-D Gaussians. Stable regardless of
    sample size, unlike histogram-based estimation which needs many samples
    per bin to avoid noise dominating the result."""
    std_p = max(std_p, 1e-6)
    std_q = max(std_q, 1e-6)
    return float(np.log(std_q / std_p) + (std_p**2 + (mean_p - mean_q)**2) / (2 * std_q**2) - 0.5)


def kl_divergence_score(embeddings, ref_stats, min_samples=5):
    """Parametric Gaussian KL divergence between the new stream's projection
    and the reference projection, both projected onto the reference's own
    top principal direction. Symmetrized (avg of both directions) since
    small-sample estimates can be direction-sensitive."""
    if len(embeddings) < min_samples:
        return 0.0  # too few samples to estimate a distribution meaningfully — don't false-alarm

    centered = embeddings - ref_stats["mean_vec"]
    prod_projection = centered @ ref_stats["top_direction"]

    prod_mean = float(prod_projection.mean())
    prod_std = float(prod_projection.std() + 1e-6)

    kl_pq = gaussian_kl(prod_mean, prod_std, ref_stats["ref_proj_mean"], ref_stats["ref_proj_std"])
    kl_qp = gaussian_kl(ref_stats["ref_proj_mean"], ref_stats["ref_proj_std"], prod_mean, prod_std)

    return float((kl_pq + kl_qp) / 2)  # symmetrized KL (Jeffreys divergence)


def drift_alert(embedding_dist_score, kl_score, dist_threshold=1.5, kl_threshold=0.5):
    triggered = (embedding_dist_score >= dist_threshold) or (kl_score >= kl_threshold)
    reasons = []
    if embedding_dist_score >= dist_threshold:
        reasons.append(f"embedding_distance={embedding_dist_score:.2f} (threshold={dist_threshold})")
    if kl_score >= kl_threshold:
        reasons.append(f"kl_divergence={kl_score:.2f} (threshold={kl_threshold})")
    return triggered, reasons
