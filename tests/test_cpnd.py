"""
Unit and Integration Tests for CPND Module
DefectVision AI — Phase 7B Verification Suite
Uses Python standard library unittest for zero extra dependencies.
"""

import unittest
import numpy as np
import tempfile
import csv
import os
import sys

# Ensure DefectVisionAI is importable
sys.path.insert(0, os.path.abspath("DefectVisionAI"))

from research.cpnd.cpnd_core import (
    CPNDScorer,
    rank_and_select_budget,
    evaluate_selective_review
)


class TestCPNDCorrectness(unittest.TestCase):

    def test_faiss_squared_to_euclidean_conversion(self):
        """Test that FAISS squared L2 distances are correctly converted to Euclidean distances."""
        dim = 4
        memory_bank = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        scorer = CPNDScorer(k_neighbors=3).fit(memory_bank)

        # Query at [1.0, 0.0, 0.0, 0.0]
        query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        d1nn, pnd = scorer.compute_patch_pnd(query)

        self.assertTrue(np.isclose(d1nn[0], 1.0, atol=1e-5))

        # Expected distances: [1.0, 1.0, 3.0]
        expected_mean = 5.0 / 3.0
        expected_std = np.sqrt(8.0 / 9.0)
        expected_pnd = expected_std / (expected_mean + 1e-7)

        self.assertTrue(np.isclose(pnd[0], expected_pnd, atol=1e-4))

    def test_k_nearest_neighbor_ordering(self):
        """Test that FAISS returns neighbors in strictly ascending distance order."""
        dim = 8
        np.random.seed(42)
        memory_bank = np.random.randn(50, dim).astype(np.float32)
        scorer = CPNDScorer(k_neighbors=5).fit(memory_bank)

        queries = np.random.randn(10, dim).astype(np.float32)
        sq_dist, _ = scorer.index.search(queries, k=5)
        euclid_dist = np.sqrt(np.maximum(sq_dist, 0.0))

        for row in euclid_dist:
            diffs = np.diff(row)
            self.assertTrue(np.all(diffs >= -1e-6), f"Neighbor distances not sorted: {row}")

    def test_zero_distance_stability(self):
        """Test that identical query vectors (distance = 0) do not cause division by zero or NaN."""
        dim = 4
        memory_bank = np.zeros((10, dim), dtype=np.float32)
        scorer = CPNDScorer(k_neighbors=4, eps=1e-7).fit(memory_bank)

        query = np.zeros((1, dim), dtype=np.float32)
        d1nn, pnd = scorer.compute_patch_pnd(query)

        self.assertFalse(np.isnan(pnd[0]))
        self.assertFalse(np.isinf(pnd[0]))
        self.assertTrue(np.isclose(d1nn[0], 0.0, atol=1e-7))
        self.assertTrue(np.isclose(pnd[0], 0.0, atol=1e-5))

    def test_numerical_determinism(self):
        """Test that repeated runs produce bitwise identical output."""
        dim = 16
        np.random.seed(12345)
        memory_bank = np.random.randn(100, dim).astype(np.float32)
        query = np.random.randn(5, dim).astype(np.float32)

        scorer1 = CPNDScorer(k_neighbors=5).fit(memory_bank)
        d1, p1 = scorer1.compute_patch_pnd(query)

        scorer2 = CPNDScorer(k_neighbors=5).fit(memory_bank)
        d2, p2 = scorer2.compute_patch_pnd(query)

        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_array_equal(p1, p2)

    def test_feature_and_memory_bank_dimensions(self):
        """Test dimension checking and validation."""
        scorer = CPNDScorer(k_neighbors=3)

        with self.assertRaises(ValueError):
            scorer.fit(np.zeros((10,), dtype=np.float32))

        with self.assertRaises(ValueError):
            scorer.fit(np.zeros((2, 10), dtype=np.float32))

        scorer.fit(np.zeros((10, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            scorer.compute_patch_pnd(np.zeros((5, 12), dtype=np.float32))

    def test_fixed_budget_selection_count(self):
        """Test that rank_and_select_budget selects exactly floor(B * N)."""
        records = [{"image_id": f"img_{i:03d}", "uncertainty": i / 100.0} for i in range(100)]

        for b in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
            reviewed, k = rank_and_select_budget(records, "uncertainty", b)
            expected_k = max(1, int(np.floor(b * 100)))
            self.assertEqual(k, expected_k)
            self.assertEqual(len(reviewed), expected_k)

    def test_deterministic_tie_handling(self):
        """Test that identical uncertainty scores break ties deterministically using secondary key."""
        records = [
            {"image_id": "img_C", "uncertainty": 0.5},
            {"image_id": "img_A", "uncertainty": 0.5},
            {"image_id": "img_D", "uncertainty": 0.5},
            {"image_id": "img_B", "uncertainty": 0.5},
        ]

        reviewed, k = rank_and_select_budget(records, "uncertainty", budget_fraction=0.5, descending=True)
        self.assertEqual(k, 2)
        self.assertEqual(reviewed[0]["image_id"], "img_A")
        self.assertEqual(reviewed[1]["image_id"], "img_B")

    def test_threshold_consistency(self):
        """Test that distance-to-threshold is maximal at score == threshold."""
        scorer = CPNDScorer(k_neighbors=3)
        th = 1.50
        u_at_thresh = scorer.compute_boundary_distance_uncertainty(1.50, th)
        u_near = scorer.compute_boundary_distance_uncertainty(1.51, th)
        u_far = scorer.compute_boundary_distance_uncertainty(2.50, th)

        self.assertGreater(u_at_thresh, u_near)
        self.assertGreater(u_near, u_far)

    def test_category_specific_calibration(self):
        """Test that joint uncertainty correctly handles category-specific thresholds and normalization."""
        scorer = CPNDScorer(k_neighbors=5)
        # Bottle threshold 1.4138 vs capsule threshold 2.3988
        u_bottle = scorer.compute_joint_uncertainty(
            pnd_score=0.40, anomaly_score=1.4140, threshold=1.4138, alpha=0.5, pnd_norm=1.0, dist_norm=1e5
        )
        u_capsule = scorer.compute_joint_uncertainty(
            pnd_score=0.40, anomaly_score=2.3990, threshold=2.3988, alpha=0.5, pnd_norm=1.0, dist_norm=1e5
        )
        self.assertTrue(u_bottle > 0)
        self.assertTrue(u_capsule > 0)
        self.assertTrue(abs(u_bottle - u_capsule) < 0.05)

    def test_pnd_not_identical_to_anomaly_score(self):
        """Test that PND ranking is mathematically distinct from raw anomaly score ranking."""
        dim = 8
        np.random.seed(999)
        memory_bank = np.random.randn(200, dim).astype(np.float32)
        scorer = CPNDScorer(k_neighbors=10).fit(memory_bank)

        test_patches = np.random.randn(50, dim).astype(np.float32)
        d1nn, pnd = scorer.compute_patch_pnd(test_patches)

        corr = np.corrcoef(d1nn, pnd)[0, 1]
        self.assertTrue(abs(corr) < 0.95, f"PND is trivially collinear with anomaly score (r={corr:.4f})")

    def test_score_export_and_reload_parity(self):
        """Test CSV export and reload parity."""
        records = [
            {"image_id": "bottle_001", "category": "bottle", "score": 1.2345, "pnd": 0.4567},
            {"image_id": "bottle_002", "category": "bottle", "score": 2.3456, "pnd": 0.5678}
        ]

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", newline="") as f:
            temp_path = f.name
            writer = csv.DictWriter(f, fieldnames=["image_id", "category", "score", "pnd"])
            writer.writeheader()
            writer.writerows(records)

        reloaded = []
        with open(temp_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reloaded.append({
                    "image_id": row["image_id"],
                    "category": row["category"],
                    "score": float(row["score"]),
                    "pnd": float(row["pnd"])
                })
        os.remove(temp_path)

        self.assertEqual(len(reloaded), len(records))
        for orig, rel in zip(records, reloaded):
            self.assertEqual(orig["image_id"], rel["image_id"])
            self.assertTrue(np.isclose(orig["score"], rel["score"]))
            self.assertTrue(np.isclose(orig["pnd"], rel["pnd"]))

    def test_empty_or_invalid_inputs(self):
        """Test defensive behavior on invalid inputs."""
        scorer = CPNDScorer(k_neighbors=5)
        with self.assertRaises(RuntimeError):
            scorer.compute_patch_pnd(np.zeros((5, 10), dtype=np.float32))

        with self.assertRaises(ValueError):
            CPNDScorer(k_neighbors=1)

        with self.assertRaises(ValueError):
            CPNDScorer(aggregation="invalid_method")

        with self.assertRaises(ValueError):
            rank_and_select_budget([], "uncertainty", budget_fraction=1.5)


if __name__ == "__main__":
    unittest.main()
