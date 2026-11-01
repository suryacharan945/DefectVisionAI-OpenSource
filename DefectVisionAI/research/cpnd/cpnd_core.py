"""
Calibrated Patch-Neighborhood Dispersion (CPND) Module
DefectVision AI — Phase 7B Research Implementation

This module implements Patch-Neighborhood Dispersion (PND) and selective review
prioritization algorithms for unsupervised visual anomaly detection (PatchCore).
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional, Union


class CPNDScorer:
    """
    Computes Patch-Neighborhood Dispersion (PND) from nominal memory banks
    and generates uncertainty rankings under fixed human review budgets.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        eps: float = 1e-7,
        aggregation: str = "argmax_score",
        eps_boundary: float = 1e-5
    ):
        """
        Args:
            k_neighbors: Number of nearest neighbors to query in nominal memory bank (>= 2).
            eps: Numerical stability constant for dispersion calculation.
            aggregation: Patch-to-image aggregation rule:
                         'argmax_score' (patch with highest anomaly score),
                         'max_dispersion' (maximum PND across all patches),
                         'mean_top5pct' (average of top 5% highest dispersion patches),
                         'p99_dispersion' (99th percentile of patch dispersion).
            eps_boundary: Numerical stability constant for distance-to-threshold inversion.
        """
        if k_neighbors < 2:
            raise ValueError(f"k_neighbors must be >= 2 for standard deviation calculation, got {k_neighbors}")
        if aggregation not in ["argmax_score", "max_dispersion", "mean_top5pct", "p99_dispersion"]:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        self.k_neighbors = k_neighbors
        self.eps = eps
        self.aggregation = aggregation
        self.eps_boundary = eps_boundary
        self.index: Optional[faiss.IndexFlatL2] = None
        self.memory_bank: Optional[np.ndarray] = None
        self.dim: Optional[int] = None
        self.grid_size: int = 28

    def fit(self, memory_bank: np.ndarray, grid_size: int = 28) -> "CPNDScorer":
        """
        Builds the FAISS L2 index over nominal memory bank embeddings.
        Args:
            memory_bank: Array of shape (M, D) with float32 patch embeddings.
            grid_size: Spatial grid size (default 28 for 224x224 input with stride 8).
        """
        if memory_bank.ndim != 2:
            raise ValueError(f"Expected 2D array (M, D), got shape {memory_bank.shape}")
        if memory_bank.shape[0] < self.k_neighbors:
            raise ValueError(
                f"Memory bank size ({memory_bank.shape[0]}) must be >= k_neighbors ({self.k_neighbors})"
            )

        self.memory_bank = np.ascontiguousarray(memory_bank, dtype=np.float32)
        self.dim = self.memory_bank.shape[1]
        self.grid_size = grid_size
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.memory_bank)
        return self

    def compute_patch_pnd(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 1-NN anomaly distance and K-NN dispersion for a collection of patches.

        Args:
            patches: Array of shape (P, D) where P = g * g is number of patches.

        Returns:
            distances_1nn: Euclidean distance to nearest neighbor for each patch (P,).
            pnd_scores: Patch-neighborhood dispersion for each patch (P,).
        """
        if self.index is None:
            raise RuntimeError("CPNDScorer has not been fitted with a memory bank.")
        if patches.ndim != 2 or patches.shape[1] != self.dim:
            raise ValueError(f"Expected patches shape (P, {self.dim}), got {patches.shape}")

        patches = np.ascontiguousarray(patches, dtype=np.float32)

        # Query top-K neighbors from FAISS (returns squared L2 distances)
        sq_distances, indices = self.index.search(patches, k=self.k_neighbors)

        # Convert squared L2 distances to Euclidean distances safely
        euclid_distances = np.sqrt(np.maximum(sq_distances, 0.0))

        # 1-NN distance for standard PatchCore score
        distances_1nn = euclid_distances[:, 0]

        # Compute mean and standard deviation across K neighbors
        means = np.mean(euclid_distances, axis=1)
        stds = np.std(euclid_distances, axis=1, ddof=0)

        # Dispersion = std / (mean + eps)
        # Stable under zero-distance (when patch is identical to a memory bank item, std=0, mean=0 -> 0.0)
        pnd_scores = stds / (means + self.eps)

        return distances_1nn, pnd_scores

    def score_patch_tensor(self, patch_embedding: np.ndarray) -> Dict[str, float]:
        """
        Scores an image from its feature map tensor.

        Args:
            patch_embedding: Array of shape (C, g, g) or (C, g*g).

        Returns:
            Dict containing:
                'anomaly_score': Standard PatchCore maximum patch distance.
                'pnd_score': Image-level aggregated dispersion score.
                'argmax_patch_idx': Index of the patch with maximum anomaly score.
        """
        if patch_embedding.ndim == 3:
            C, g1, g2 = patch_embedding.shape
            patches = patch_embedding.reshape(C, g1 * g2).T
        elif patch_embedding.ndim == 2:
            patches = patch_embedding
        else:
            raise ValueError(f"Unexpected patch_embedding shape: {patch_embedding.shape}")

        distances_1nn, pnd_scores = self.compute_patch_pnd(patches)

        argmax_idx = int(np.argmax(distances_1nn))
        anomaly_score = float(distances_1nn[argmax_idx])

        # Aggregate patch PND to image level according to chosen aggregation rule
        if self.aggregation == "argmax_score":
            image_pnd = float(pnd_scores[argmax_idx])
        elif self.aggregation == "max_dispersion":
            image_pnd = float(np.max(pnd_scores))
        elif self.aggregation == "mean_top5pct":
            k_top = max(1, int(len(pnd_scores) * 0.05))
            top_pnd = np.partition(pnd_scores, -k_top)[-k_top:]
            image_pnd = float(np.mean(top_pnd))
        elif self.aggregation == "p99_dispersion":
            image_pnd = float(np.percentile(pnd_scores, 99))
        else:
            image_pnd = float(pnd_scores[argmax_idx])

        return {
            "anomaly_score": anomaly_score,
            "pnd_score": image_pnd,
            "argmax_patch_idx": argmax_idx
        }

    def compute_boundary_distance_uncertainty(self, anomaly_score: float, threshold: float) -> float:
        """
        Computes the distance-to-threshold uncertainty score.
        Samples closest to the decision threshold tau receive highest uncertainty:
            u_dist = 1.0 / (|score - threshold| + eps_boundary)
        """
        dist = abs(anomaly_score - threshold)
        return float(1.0 / (dist + self.eps_boundary))

    def compute_joint_uncertainty(
        self,
        pnd_score: float,
        anomaly_score: float,
        threshold: float,
        alpha: float = 0.5,
        pnd_norm: float = 1.0,
        dist_norm: float = 1.0
    ) -> float:
        """
        Combines boundary proximity and patch dispersion into a joint uncertainty metric:
            u_joint = alpha * (u_dist / dist_norm) + (1 - alpha) * (u_pnd / pnd_norm)
        """
        u_dist = self.compute_boundary_distance_uncertainty(anomaly_score, threshold)
        norm_dist = u_dist / (dist_norm + 1e-8)
        norm_pnd = pnd_score / (pnd_norm + 1e-8)
        return float(alpha * norm_dist + (1.0 - alpha) * norm_pnd)


def rank_and_select_budget(
    records: List[Dict],
    uncertainty_key: str,
    budget_fraction: float,
    descending: bool = True,
    tie_breaker_key: str = "image_id"
) -> Tuple[List[Dict], int]:
    """
    Ranks items by an uncertainty score and selects the top K = floor(B * N) for human review.

    Args:
        records: List of dictionaries representing inspection items.
        uncertainty_key: Key in record dictionary to sort by.
        budget_fraction: Review budget B in (0.0, 1.0].
        descending: Sort descending (highest uncertainty first).
        tie_breaker_key: Secondary deterministic sort key to ensure deterministic order.

    Returns:
        reviewed_records: Subset of records selected for review.
        k_count: Number of records selected (floor(B * N)).
    """
    if not (0.0 < budget_fraction <= 1.0):
        raise ValueError(f"budget_fraction must be in (0.0, 1.0], got {budget_fraction}")

    N = len(records)
    if N == 0:
        return [], 0

    K = max(1, int(np.floor(budget_fraction * N)))

    # Stable 2-stage sort: first sort ascending by tie-breaker key
    sorted_records = sorted(records, key=lambda r: str(r.get(tie_breaker_key, "")))
    # Then sort by uncertainty key (stable sort preserves alphabetical tie-breaking)
    sorted_records.sort(key=lambda r: float(r[uncertainty_key]), reverse=descending)

    return sorted_records[:K], K


def evaluate_selective_review(
    all_records: List[Dict],
    reviewed_records: List[Dict],
    true_label_key: str = "true_label",
    pred_label_key: str = "predicted_label",
    positive_label: str = "defective",
    negative_label: str = "normal"
) -> Dict[str, float]:
    """
    Computes selective review performance metrics:
    - Error Capture Rate (ECR): fraction of total baseline misclassifications audited and corrected.
    - Defect Escape Reduction (DER): fraction of baseline false negatives audited and corrected.
    - Review Defect Yield (DY): fraction of audited items that are true defectives.
    """
    total_n = len(all_records)
    reviewed_n = len(reviewed_records)

    # Base classification error counts
    base_errors = 0
    base_fn = 0
    base_fp = 0
    base_defects = 0

    for r in all_records:
        y_true = str(r[true_label_key]).lower()
        y_pred = str(r[pred_label_key]).lower()
        is_pos = (y_true == positive_label)
        is_pred_pos = (y_pred == positive_label)

        if is_pos:
            base_defects += 1
        if is_pos and not is_pred_pos:
            base_fn += 1
            base_errors += 1
        elif not is_pos and is_pred_pos:
            base_fp += 1
            base_errors += 1

    # Audited error counts
    captured_errors = 0
    captured_fn = 0
    captured_fp = 0
    captured_defects = 0

    for r in reviewed_records:
        y_true = str(r[true_label_key]).lower()
        y_pred = str(r[pred_label_key]).lower()
        is_pos = (y_true == positive_label)
        is_pred_pos = (y_pred == positive_label)

        if is_pos:
            captured_defects += 1
        if is_pos and not is_pred_pos:
            captured_fn += 1
            captured_errors += 1
        elif not is_pos and is_pred_pos:
            captured_fp += 1
            captured_errors += 1

    ecr = float(captured_errors / base_errors) if base_errors > 0 else 0.0
    der = float(captured_fn / base_fn) if base_fn > 0 else 0.0
    yield_rate = float(captured_defects / reviewed_n) if reviewed_n > 0 else 0.0

    return {
        "total_images": total_n,
        "reviewed_images": reviewed_n,
        "base_errors": base_errors,
        "base_fn": base_fn,
        "base_fp": base_fp,
        "captured_errors": captured_errors,
        "captured_fn": captured_fn,
        "captured_fp": captured_fp,
        "error_capture_rate": ecr,
        "defect_escape_reduction": der,
        "review_yield": yield_rate
    }
