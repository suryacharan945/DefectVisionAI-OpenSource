"""
Unit and Integration Tests for Density-Normalized CPND Module.
DefectVision AI — Phase 7C.
"""

import unittest
import numpy as np
import os
import sys

# Ensure local DefectVisionAI is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DefectVisionAI"))

from research.cpnd.cpnd_density_norm import (
    DensityNormalizedCPNDScorer,
    rank_and_select_budget,
    evaluate_selective_review
)


class TestDensityNormalizedCPND(unittest.TestCase):
    """Verifies correctness, numerical stability, and deterministic execution of Density-Normalized CPND."""

    def setUp(self):
        np.random.seed(42)
        self.dim = 64
        self.num_vectors = 200
        # Synthetic nominal memory bank
        self.memory_bank = np.random.randn(self.num_vectors, self.dim).astype(np.float32)

    def test_self_neighbor_exclusion(self):
        """Test that memory bank self-neighbor (distance = 0) is excluded from reference statistics."""
        k = 5
        scorer = DensityNormalizedCPNDScorer(self.memory_bank, k=k, ref_k=k)
        # All nominal dispersions should be positive and non-zero
        self.assertEqual(len(scorer.nominal_dispersions), self.num_vectors)
        self.assertTrue(np.all(scorer.nominal_dispersions > 0.0))
        self.assertFalse(np.any(np.isnan(scorer.nominal_dispersions)))

    def test_normalization_types(self):
        """Test both median_mad and mean_std normalization options."""
        patches = np.random.randn(10, self.dim).astype(np.float32)
        
        scorer_mad = DensityNormalizedCPNDScorer(self.memory_bank, k=5, normalization_type="median_mad")
        res_mad = scorer_mad.score_patches(patches)
        self.assertIn("norm_dispersion", res_mad)
        self.assertEqual(len(res_mad["norm_dispersion"]), 10)
        self.assertFalse(np.any(np.isnan(res_mad["norm_dispersion"])))

        scorer_std = DensityNormalizedCPNDScorer(self.memory_bank, k=5, normalization_type="mean_std")
        res_std = scorer_std.score_patches(patches)
        self.assertIn("norm_dispersion", res_std)
        self.assertEqual(len(res_std["norm_dispersion"]), 10)
        self.assertFalse(np.any(np.isnan(res_std["norm_dispersion"])))

    def test_zero_distance_stability(self):
        """Test that identical query vectors (distance = 0) do not cause division by zero or NaN."""
        # Query identical vectors to the first 3 vectors in memory bank
        query_identical = self.memory_bank[:3].copy()
        scorer = DensityNormalizedCPNDScorer(self.memory_bank, k=5)
        res = scorer.score_patches(query_identical)
        
        self.assertFalse(np.any(np.isnan(res["raw_dispersion"])))
        self.assertFalse(np.any(np.isnan(res["norm_dispersion"])))
        self.assertFalse(np.any(np.isinf(res["raw_dispersion"])))
        self.assertFalse(np.any(np.isinf(res["norm_dispersion"])))

    def test_numerical_determinism(self):
        """Test that repeated runs produce bitwise identical output."""
        patches = np.random.randn(15, self.dim).astype(np.float32)
        scorer1 = DensityNormalizedCPNDScorer(self.memory_bank, k=5)
        res1 = scorer1.score_patches(patches)

        scorer2 = DensityNormalizedCPNDScorer(self.memory_bank, k=5)
        res2 = scorer2.score_patches(patches)

        np.testing.assert_array_equal(res1["norm_dispersion"], res2["norm_dispersion"])
        np.testing.assert_array_equal(res1["raw_dispersion"], res2["raw_dispersion"])

    def test_image_scoring_and_argmax(self):
        """Test image scoring extracts the argmax patch properties correctly."""
        patches = np.random.randn(20, self.dim).astype(np.float32)
        # Artificially make patch 7 an extreme outlier
        patches[7] = self.memory_bank[0] + 100.0

        scorer = DensityNormalizedCPNDScorer(self.memory_bank, k=5)
        img_res = scorer.score_image(patches, threshold=2.0, boundary_delta=0.5)

        patch_res = scorer.score_patches(patches)
        expected_argmax = int(np.argmax(patch_res["nn1_distances"]))
        self.assertEqual(expected_argmax, 7)
        self.assertAlmostEqual(img_res["anomaly_score"], float(patch_res["nn1_distances"][7]), places=4)
        self.assertAlmostEqual(img_res["raw_pnd"], float(patch_res["raw_dispersion"][7]), places=4)
        self.assertAlmostEqual(img_res["norm_pnd"], float(patch_res["norm_dispersion"][7]), places=4)

    def test_boundary_gated_tier_sorting(self):
        """Test that tiered boundary-gated ranking strictly prioritizes Tier 1 (in-band) items."""
        records = [
            {"image_id": "img1", "in_band": 0, "norm_pnd": 5.0, "u_dist": -2.0},  # out of band
            {"image_id": "img2", "in_band": 1, "norm_pnd": 0.5, "u_dist": -0.05}, # in band, low pnd
            {"image_id": "img3", "in_band": 1, "norm_pnd": 2.0, "u_dist": -0.08}, # in band, high pnd
            {"image_id": "img4", "in_band": 0, "norm_pnd": 10.0, "u_dist": -1.5}  # out of band
        ]

        # Select budget 50% (2 items)
        selected, k = rank_and_select_budget(
            records,
            uncertainty_key="norm_pnd",
            budget_fraction=0.5,
            gating_tier_key="in_band",
            secondary_key="image_id"
        )
        self.assertEqual(k, 2)
        # Both selected should be Tier 1 (in_band == 1), and img3 (pnd=2.0) before img2 (pnd=0.5)
        self.assertEqual(selected[0]["image_id"], "img3")
        self.assertEqual(selected[1]["image_id"], "img2")

    def test_deterministic_tie_breaking(self):
        """Test that identical scores break ties deterministically by secondary key (alphabetical)."""
        records = [
            {"image_id": "img_Z", "score": 1.0},
            {"image_id": "img_A", "score": 1.0},
            {"image_id": "img_M", "score": 1.0}
        ]
        selected, k = rank_and_select_budget(
            records,
            uncertainty_key="score",
            budget_fraction=0.66,
            secondary_key="image_id"
        )
        self.assertEqual(k, 1)
        self.assertEqual(selected[0]["image_id"], "img_A")

    def test_yield_metrics_disambiguation(self):
        """Test that Defect Review Yield and Error Review Yield are correctly distinguished."""
        all_records = [
            {"image_id": "1", "true_label": "anomaly", "predicted_label": "anomaly"}, # TP (defect, no error)
            {"image_id": "2", "true_label": "anomaly", "predicted_label": "good"},    # FN (defect + error)
            {"image_id": "3", "true_label": "good", "predicted_label": "good"},       # TN (no defect, no error)
            {"image_id": "4", "true_label": "good", "predicted_label": "anomaly"}     # FP (no defect + error)
        ]
        
        # Auditor reviews image 1 (TP) and image 2 (FN)
        reviewed = [all_records[0], all_records[1]]
        metrics = evaluate_selective_review(all_records, reviewed)

        # Base counts: total=4, errors=2 (FN=1, FP=1), defects=2 (TP=1, FN=1)
        self.assertEqual(metrics["base_errors"], 2)
        self.assertEqual(metrics["base_defects"], 2)
        # Captured: defects = 2 (TP + FN), errors = 1 (FN)
        self.assertEqual(metrics["captured_defects"], 2)
        self.assertEqual(metrics["captured_errors"], 1)

        # Defect Yield = 2 defects / 2 reviewed = 1.0 (100%)
        self.assertEqual(metrics["defect_review_yield"], 1.0)
        # Error Yield = 1 error / 2 reviewed = 0.5 (50%)
        self.assertEqual(metrics["error_review_yield"], 0.5)
        # ECR = 1 error captured / 2 base errors = 0.5 (50%)
        self.assertEqual(metrics["error_capture_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
