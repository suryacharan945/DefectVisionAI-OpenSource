
import pandas as pd
import os

class DashboardAnalytics:
    def __init__(self, logs_dir, reports_dir):
        self.logs_dir = logs_dir
        self.reports_dir = reports_dir
        self._cache = {}

    def _load(self, name, path):
        if name not in self._cache:
            self._cache[name] = pd.read_csv(path)
        return self._cache[name]

    def severity_log(self):
        return self._load("severity", os.path.join(self.logs_dir, "step9_severity_log.csv"))

    def drift_log(self):
        return self._load("drift", os.path.join(self.logs_dir, "step12_drift_log_v3.csv"))

    def model_comparison(self):
        return self._load("comparison", os.path.join(self.reports_dir, "step8_model_comparison_FAIR.csv"))

    def active_learning_sensitivity(self):
        return self._load("al_sens", os.path.join(self.reports_dir, "step11_threshold_sensitivity.csv"))

    def most_common_defect(self, category=None):
        df = self.severity_log()
        df = df[df["true_label"] == "defective"]
        if category:
            df = df[df["category"].str.lower() == category.lower()]
        if len(df) == 0:
            return None
        counts = df["defect_type"].value_counts()
        return {"defect_type": counts.index[0], "count": int(counts.iloc[0]), "total_defective": len(df)}

    def defect_rate_by_category(self):
        df = self.severity_log()
        rates = df.groupby("category")["true_label"].apply(
            lambda x: round(100 * (x == "defective").sum() / len(x), 2)
        ).to_dict()
        return rates

    def severity_summary(self):
        df = self.severity_log()
        defective = df[df["true_label"] == "defective"]
        return {
            "mean_severity_defective": round(defective["severity_score"].mean(), 2),
            "pct_critical": round(100 * (defective["severity_level"] == "Critical").sum() / len(defective), 2),
            "pct_moderate": round(100 * (defective["severity_level"] == "Moderate").sum() / len(defective), 2),
            "pct_minor": round(100 * (defective["severity_level"] == "Minor").sum() / len(defective), 2),
        }

    def model_ranking(self):
        df = self.model_comparison()
        df = df.sort_values("Mean Category AUROC", ascending=False)
        return df[["Model", "Mean Category AUROC", "F1"]].to_dict("records")

    def drift_summary(self):
        df = self.drift_log()
        summary = {}
        for stream in df["stream"].unique():
            subset = df[df["stream"] == stream]
            alerted = int(subset["drift_alert"].sum())
            total = len(subset)
            rate_pct = round(100 * alerted / total, 1) if total else 0
            # Explicit interpretation label -- removes ambiguity for the LLM,
            # since raw alert counts alone were being misread as "drift present"
            # regardless of whether the rate was 6.7% or 100%.
            if rate_pct <= 10:
                status = "minimal / expected false-positive rate, no real drift"
            elif rate_pct <= 60:
                status = "moderate drift detected"
            else:
                status = "strong, systematic drift detected"
            summary[stream] = {
                "alerted": alerted, "total": total,
                "alert_rate_pct": rate_pct, "status": status,
            }
        return summary

    def active_learning_efficiency(self, review_budget_pct=10):
        df = self.active_learning_sensitivity()
        closest = df.iloc[(df["pct_of_dataset"] - review_budget_pct).abs().argsort()[:1]]
        return closest.to_dict("records")[0] if len(closest) > 0 else None

    def worst_performing_category(self, model_name="PatchCore"):
        matrix_path = os.path.join(self.reports_dir, "step8_per_category_auroc_matrix.csv")
        if not os.path.exists(matrix_path):
            return None
        matrix = pd.read_csv(matrix_path, index_col=0)
        if model_name not in matrix.columns:
            return None
        worst = matrix[model_name].idxmin()
        return {"category": worst, "auroc": round(matrix[model_name].min(), 4)}
