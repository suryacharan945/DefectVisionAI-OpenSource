"""
Density-Normalized Patch-Neighborhood Dispersion (CPND) Module.
DefectVision AI — Phase 7C Controlled Revision.

Implements:
1. Exact k-NN search via FAISS.
2. Self-neighbor-excluded nominal training dispersion precomputation.
3. Robust local reference estimation (median/MAD and mean/std).
4. Local density-normalized patch dispersion.
5. Boundary-gated ranking policies (hard band & Gaussian soft gating).
6. Disambiguated metrics evaluation (Defect Yield vs Error Yield).
"""

import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional, Union, Any


class DensityNormalizedCPNDScorer:
    """
    Computes Patch-Neighborhood Dispersion normalized by local nominal reference statistics.
    """

    def __init__(
        self,
        memory_bank: np.ndarray,
        k: int = 5,
        ref_k: int = 5,
        normalization_type: str = "median_mad",
        epsilon: float = 1e-7
    ):
        """
        Args:
            memory_bank: Nominal training coreset [M, D].
            k: Number of nearest neighbors for query patch dispersion.
            ref_k: Number of nearest neighbors for nominal reference dispersion.
            normalization_type: 'median_mad' (robust) or 'mean_std'.
            epsilon: Numerical stabilizer for zero distance / zero variance.
        """
        if memory_bank.ndim != 2:
            raise ValueError(f"memory_bank must be 2D [M, D], got shape {memory_bank.shape}")
        if memory_bank.shape[0] < max(k + 1, ref_k + 1):
            raise ValueError(f"memory_bank size {memory_bank.shape[0]} too small for k={k}, ref_k={ref_k}")
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")

        self.memory_bank = np.ascontiguousarray(memory_bank, dtype=np.float32)
        self.num_vectors, self.dim = self.memory_bank.shape
        self.k = int(k)
        self.ref_k = int(ref_k)
        self.normalization_type = normalization_type
        self.epsilon = float(epsilon)

        # Build FAISS exact L2 index
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.memory_bank)

        # Precompute nominal training dispersion with self-neighbor excluded
        self._precompute_nominal_dispersions()

    def _precompute_nominal_dispersions(self) -> None:
        """
        Queries ref_k + 1 neighbors for every vector in memory bank.
        Drops neighbor 0 (the vector itself) to eliminate self-neighbor contamination.
        """
        # Query ref_k + 1 neighbors
        sq_dist, indices = self.index.search(self.memory_bank, self.ref_k + 1)
        
        # Discard column 0 (self-neighbor distance = 0)
        sq_dist = sq_dist[:, 1:]
        indices = indices[:, 1:]

        euclid_dist = np.sqrt(np.maximum(sq_dist, 0.0))
        mu = np.mean(euclid_dist, axis=1)
        sigma = np.std(euclid_dist, axis=1, ddof=0)
        
        # Nominal dispersion for all training vectors [M]
        self.nominal_dispersions = sigma / (mu + self.epsilon)

    def score_patches(
        self,
        patches: np.ndarray,
        k: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculates raw 1-NN anomaly distances, raw dispersion, and density-normalized dispersion
        for all patches in an image.

        Args:
            patches: Query patch features [P, D].
            k: Optional k override; defaults to self.k.

        Returns:
            Dictionary containing:
                'nn1_distances': [P] PatchCore 1-NN Euclidean distances
                'raw_dispersion': [P] Raw coefficient of variation
                'norm_dispersion': [P] Local-density normalized dispersion
                'ref_median': [P] Local nominal median dispersion
                'ref_mad': [P] Local nominal MAD dispersion
        """
        if patches.ndim != 2 or patches.shape[1] != self.dim:
            raise ValueError(f"Expected patches shape [P, {self.dim}], got {patches.shape}")

        P = patches.shape[0]
        if P == 0:
            return {
                "nn1_distances": np.empty(0, dtype=np.float32),
                "raw_dispersion": np.empty(0, dtype=np.float32),
                "norm_dispersion": np.empty(0, dtype=np.float32),
                "ref_median": np.empty(0, dtype=np.float32),
                "ref_mad": np.empty(0, dtype=np.float32)
            }

        query_k = int(k if k is not None else self.k)
        patches_contig = np.ascontiguousarray(patches, dtype=np.float32)

        # Query k nearest neighbors
        sq_dist, indices = self.index.search(patches_contig, query_k)
        euclid_dist = np.sqrt(np.maximum(sq_dist, 0.0))

        # 1-NN PatchCore distance
        nn1 = euclid_dist[:, 0]

        # Raw patch dispersion
        mu = np.mean(euclid_dist, axis=1)
        sigma = np.std(euclid_dist, axis=1, ddof=0)
        raw_disp = sigma / (mu + self.epsilon)

        # Local nominal reference statistics
        # indices shape: [P, query_k]
        # Gather nominal dispersions of the retrieved neighbors: [P, query_k]
        nbr_nominal_disps = self.nominal_dispersions[indices]

        if self.normalization_type == "median_mad":
            ref_center = np.median(nbr_nominal_disps, axis=1)
            # MAD with scale factor 1.4826 for asymptotic normal consistency
            abs_dev = np.abs(nbr_nominal_disps - ref_center[:, np.newaxis])
            ref_scale = np.median(abs_dev, axis=1) * 1.4826
            # Numerical stabilizer floor
            ref_scale = np.maximum(ref_scale, 1e-5)
            norm_disp = (raw_disp - ref_center) / ref_scale
        elif self.normalization_type == "mean_std":
            ref_center = np.mean(nbr_nominal_disps, axis=1)
            ref_scale = np.std(nbr_nominal_disps, axis=1, ddof=0)
            ref_scale = np.maximum(ref_scale, 1e-5)
            norm_disp = (raw_disp - ref_center) / ref_scale
        else:
            raise ValueError(f"Unknown normalization_type: {self.normalization_type}")

        return {
            "nn1_distances": nn1,
            "raw_dispersion": raw_disp,
            "norm_dispersion": norm_disp,
            "ref_median": ref_center,
            "ref_mad": ref_scale
        }

    def score_image(
        self,
        patches: np.ndarray,
        threshold: float,
        boundary_delta: float = 0.15
    ) -> Dict[str, float]:
        """
        Scores an image, computing image-level PatchCore anomaly score, raw dispersion,
        density-normalized dispersion, boundary distance, and boundary-gated rankings.

        Args:
            patches: Image patch features [P, D].
            threshold: Category decision threshold tau.
            boundary_delta: Half-width of decision boundary region.

        Returns:
            Dictionary of scalar metrics.
        """
        patch_res = self.score_patches(patches)
        nn1 = patch_res["nn1_distances"]
        if len(nn1) == 0:
            return {
                "anomaly_score": 0.0,
                "u_dist": 0.0,
                "raw_pnd": 0.0,
                "norm_pnd": 0.0,
                "in_boundary_band": 0.0,
                "u_gated_soft": 0.0
            }

        # Argmax patch index corresponds to PatchCore image anomaly score
        argmax_idx = int(np.argmax(nn1))
        anomaly_score = float(nn1[argmax_idx])

        # Distance to decision threshold: |S(X) - tau|
        dist_to_thresh = float(abs(anomaly_score - threshold))
        u_dist = -dist_to_thresh  # Closer to threshold -> higher uncertainty score

        # Dispersion of the argmax patch
        raw_pnd = float(patch_res["raw_dispersion"][argmax_idx])
        norm_pnd = float(patch_res["norm_dispersion"][argmax_idx])

        # Boundary gating
        in_band = 1.0 if dist_to_thresh <= boundary_delta else 0.0
        
        # Gaussian soft gating: exp( - (S(X) - tau)^2 / (2 * delta^2) ) * norm_pnd
        soft_gate = float(np.exp(- (dist_to_thresh ** 2) / (2.0 * (boundary_delta ** 2) + 1e-7)))
        u_gated_soft = soft_gate * norm_pnd

        return {
            "anomaly_score": anomaly_score,
            "dist_to_threshold": dist_to_thresh,
            "u_dist": u_dist,
            "raw_pnd": raw_pnd,
            "norm_pnd": norm_pnd,
            "in_boundary_band": in_band,
            "u_gated_soft": u_gated_soft
        }


def rank_and_select_budget(
    records: List[Dict[str, Any]],
    uncertainty_key: str,
    budget_fraction: float,
    gating_tier_key: Optional[str] = None,
    secondary_key: str = "image_id"
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Selects top floor(B * N) records based on uncertainty score.
    If gating_tier_key is provided, performs tiered sorting:
      Tier 1 (gating_tier_key == 1): sorted by uncertainty_key descending.
      Tier 2 (gating_tier_key == 0): sorted by distance to threshold ascending.
    Ties broken deterministically by secondary_key ascending.
    """
    if not (0.0 < budget_fraction <= 1.0):
        raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction}")

    N = len(records)
    if N == 0:
        return [], 0

    K = int(np.floor(budget_fraction * N))
    if K == 0 and budget_fraction > 0:
        K = 1

    # Stage 1: Sort by secondary key ascending for determinism
    sorted_stage1 = sorted(records, key=lambda x: str(x.get(secondary_key, "")))

    if gating_tier_key is not None:
        # Tiered sorting:
        # Tier 1 (in band) comes first (tier=1), then Tier 0 (tier=0).
        # Within Tier 1: uncertainty_key descending.
        # Within Tier 0: secondary fallback (e.g. u_dist descending).
        def sort_tuple(r):
            tier = int(r.get(gating_tier_key, 0))
            if tier == 1:
                # Primary: tier=1, secondary: uncertainty descending
                return (1, float(r.get(uncertainty_key, -1e9)))
            else:
                # Primary: tier=0, secondary: u_dist descending
                return (0, float(r.get("u_dist", -1e9)))

        sorted_records = sorted(sorted_stage1, key=sort_tuple, reverse=True)
    else:
        # Standard sort by uncertainty descending
        sorted_records = sorted(
            sorted_stage1,
            key=lambda x: float(x.get(uncertainty_key, 0.0)),
            reverse=True
        )

    selected = sorted_records[:K]
    return selected, K


def evaluate_selective_review(
    all_records: List[Dict[str, Any]],
    reviewed_records: List[Dict[str, Any]],
    true_label_key: str = "true_label",
    pred_label_key: str = "predicted_label",
    positive_label: str = "anomaly"
) -> Dict[str, Union[int, float]]:
    """
    Evaluates selective review performance with unambiguous yield metrics:
      - Error_Capture_Rate (ECR): captured_errors / base_errors
      - Defect_Escape_Reduction (DER): captured_fn / base_fn
      - Defect_Review_Yield: captured_defects / reviewed_n
      - Error_Review_Yield: captured_errors / reviewed_n
    """
    total_n = len(all_records)
    reviewed_n = len(reviewed_records)

    pos_names = {"anomaly", "defective", "defect", "1", "true", positive_label.lower()}

    def is_defective(r):
        return str(r.get(true_label_key, "")).lower() in pos_names

    def get_error_type(r):
        if "is_fn" in r and "is_fp" in r:
            return bool(r["is_fn"]), bool(r["is_fp"]), bool(r["is_error"])
        y_true = str(r.get(true_label_key, "")).lower() in pos_names
        y_pred = str(r.get(pred_label_key, "")).lower() in pos_names
        is_fn = y_true and not y_pred
        is_fp = not y_true and y_pred
        is_err = is_fn or is_fp
        return is_fn, is_fp, is_err

    base_errors = 0
    base_fn = 0
    base_fp = 0
    base_defects = 0

    for r in all_records:
        is_fn, is_fp, is_err = get_error_type(r)
        if is_defective(r):
            base_defects += 1
        if is_fn:
            base_fn += 1
        if is_fp:
            base_fp += 1
        if is_err:
            base_errors += 1

    captured_errors = 0
    captured_fn = 0
    captured_fp = 0
    captured_defects = 0

    for r in reviewed_records:
        is_fn, is_fp, is_err = get_error_type(r)
        if is_defective(r):
            captured_defects += 1
        if is_fn:
            captured_fn += 1
            captured_errors += 1
        elif is_fp:
            captured_fp += 1
            captured_errors += 1

    ecr = float(captured_errors / base_errors) if base_errors > 0 else 0.0
    der = float(captured_fn / base_fn) if base_fn > 0 else 0.0
    defect_yield = float(captured_defects / reviewed_n) if reviewed_n > 0 else 0.0
    error_yield = float(captured_errors / reviewed_n) if reviewed_n > 0 else 0.0

    return {
        "total_images": total_n,
        "reviewed_images": reviewed_n,
        "base_errors": base_errors,
        "base_fn": base_fn,
        "base_fp": base_fp,
        "base_defects": base_defects,
        "captured_errors": captured_errors,
        "captured_fn": captured_fn,
        "captured_fp": captured_fp,
        "captured_defects": captured_defects,
        "error_capture_rate": ecr,
        "defect_escape_reduction": der,
        "defect_review_yield": defect_yield,
        "error_review_yield": error_yield
    }
