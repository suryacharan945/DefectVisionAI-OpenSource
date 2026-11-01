
import pandas as pd
import numpy as np

class PatternAnalyzer:
    def __init__(self, severity_log, per_category_auroc_matrix=None):
        self.severity_log = severity_log
        self.auroc_matrix = per_category_auroc_matrix

    def severity_by_category(self):
        defective = self.severity_log[self.severity_log["true_label"] == "defective"]
        stats = defective.groupby("category")["severity_score"].agg(["mean", "count"]).round(2)
        stats = stats.sort_values("mean", ascending=False)
        return stats.reset_index().to_dict("records")

    def defect_type_severity_ranking(self, category=None):
        defective = self.severity_log[self.severity_log["true_label"] == "defective"]
        if category:
            defective = defective[defective["category"].str.lower() == category.lower()]
        if len(defective) == 0:
            return []
        stats = defective.groupby("defect_type")["severity_score"].agg(["mean", "count"]).round(2)
        stats = stats.sort_values("mean", ascending=False)
        return stats.reset_index().to_dict("records")

    def confidence_severity_correlation(self):
        """Checks whether the model's classifier confidence tracks with
        anomaly-based severity -- large disagreement would flag cases
        worth manual review."""
        defective = self.severity_log[self.severity_log["true_label"] == "defective"]
        corr = defective["confidence_pct"].corr(defective["severity_score"])
        return {"correlation": round(float(corr), 3),
                "n_samples": len(defective)}

    def low_confidence_high_severity_cases(self, conf_threshold=60, sev_threshold=70):
        """Surfaces genuinely interesting cases: high anomaly/area severity
        but low classifier confidence -- worth flagging for review since
        the two signals disagree."""
        defective = self.severity_log[self.severity_log["true_label"] == "defective"]
        flagged = defective[
            (defective["confidence_pct"] < conf_threshold) &
            (defective["severity_score"] > sev_threshold)
        ]
        return {
            "count": len(flagged),
            "pct_of_defective": round(100 * len(flagged) / len(defective), 2),
            "example_categories": flagged["category"].value_counts().head(3).to_dict(),
        }

    def weakest_detection_categories(self, top_n=3):
        """Which categories the DETECTION MODELS struggle with most --
        grounded in real Step 8 AUROC numbers, not fabricated."""
        if self.auroc_matrix is None:
            return None
        avg_per_category = self.auroc_matrix.mean(axis=1).sort_values()
        return avg_per_category.head(top_n).round(4).reset_index().rename(
            columns={"index": "category", 0: "avg_auroc"}
        ).to_dict("records")

    def area_severity_relationship(self):
        """Does defect area% actually predict severity, or is severity
        driven more by anomaly score / confidence? Real correlation check."""
        defective = self.severity_log[self.severity_log["true_label"] == "defective"]
        corr = defective["defect_area_pct"].corr(defective["severity_score"])
        return {"correlation": round(float(corr), 3), "n_samples": len(defective)}
