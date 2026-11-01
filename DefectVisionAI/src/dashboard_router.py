
import re

INTENT_PATTERNS = [
    ("most_common_defect", [r"most common defect", r"which defect.*(most|common)", r"top defect"]),
    ("defect_rate_by_category", [r"defect rate", r"which categor(y|ies).*(most|worst|highest) defect"]),
    ("severity_summary", [r"severity", r"critical.*(defect|percent)", r"how severe"]),
    ("model_ranking", [r"best model", r"which model", r"model.*(compar|perform|rank)", r"patchcore.*vs", r"auroc"]),
    ("drift_summary", [r"drift", r"distribution shift", r"production.*(shift|change)"]),
    ("active_learning_efficiency", [r"active learning", r"review budget", r"labeling effort", r"uncertain"]),
    ("worst_performing_category", [r"worst.*(categor|perform)", r"weakest categor"]),
]

def route_intent(question):
    q = question.lower()
    for intent, patterns in INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return None


def extract_category(question, known_categories):
    q = question.lower()
    for cat in known_categories:
        if cat.lower() in q:
            return cat
    return None


def extract_model_name(question, known_models):
    q = question.lower()
    for model in known_models:
        # match on first word of model name too (e.g. "patchcore" matches "PatchCore")
        if model.lower() in q or model.lower().split()[0] in q:
            return model
    return None
