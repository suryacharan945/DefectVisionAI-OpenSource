
import numpy as np

def uncertainty_score(prob_defective):
    """
    Distance from the decision boundary (0.5), inverted to a 0-1 uncertainty scale.
    prob_defective near 0.5 -> uncertainty near 1.0 (model is genuinely unsure).
    prob_defective near 0.0 or 1.0 -> uncertainty near 0.0 (model is confident).
    """
    return 1.0 - abs(2 * prob_defective - 1.0)


def entropy_score(prob_defective, eps=1e-8):
    """Binary entropy, an alternative/complementary uncertainty measure."""
    p = np.clip(prob_defective, eps, 1 - eps)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def select_for_review(df, uncertainty_col="uncertainty", threshold=0.5, max_samples=None):
    """Select rows whose uncertainty exceeds threshold, most-uncertain first."""
    flagged = df[df[uncertainty_col] >= threshold].sort_values(uncertainty_col, ascending=False)
    if max_samples is not None:
        flagged = flagged.head(max_samples)
    return flagged


def simulate_human_verification(row, true_label_col="true_label", pred_label_col="predicted_label"):
    """
    Stand-in for a real reviewer UI: in production this would be a human
    clicking 'confirm' or 'correct' on the prediction. Here, ground truth
    plays that role so the pipeline can be validated end-to-end offline.
    Returns whether the model's original prediction was correct, and the
    verified (corrected) label to use for future retraining.
    """
    was_correct = row[true_label_col] == row[pred_label_col]
    verified_label = row[true_label_col]
    return was_correct, verified_label
