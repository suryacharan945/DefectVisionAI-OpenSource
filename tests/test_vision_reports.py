import unittest
import os
import csv
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestDefectVisionReports(unittest.TestCase):

    def test_fair_comparison_report_exists_and_valid(self):
        csv_path = os.path.join(PROJECT_ROOT, "DefectVisionAI", "reports", "step8_model_comparison_FAIR.csv")
        self.assertTrue(os.path.exists(csv_path), f"Report not found at {csv_path}")
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            
        self.assertGreaterEqual(len(reader), 4, "Should evaluate at least 4 architectures")
        models = [r["Model"] for r in reader]
        self.assertTrue(any("EfficientNet" in m for m in models))
        self.assertIn("PatchCore", models)
        self.assertIn("PaDiM", models)
        self.assertIn("Autoencoder", models)
        
        for row in reader:
            auroc = float(row["Mean Category AUROC"])
            self.assertGreater(auroc, 0.5, "AUROC must be better than random chance")
            self.assertLessEqual(auroc, 1.0, "AUROC cannot exceed 1.0")
            images = int(row["Images Evaluated"])
            self.assertEqual(images, 856, "Benchmark must cover standard 856 MVTec AD test images")

    def test_frontend_data_integrity(self):
        json_path = os.path.join(PROJECT_ROOT, "frontend", "public", "data", "models_comparison.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, (list, dict))

if __name__ == "__main__":
    unittest.main()
