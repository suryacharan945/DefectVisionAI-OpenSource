
import re

DRIFT_MARKERS = [
    r"\n\n",
    r"```",
    r"\[\{[\"\']?name",
    r"\{[\"\']name[\"\']\s*:",
    r"Answer Choices",
    r"\bHey[,!]",
    r"\bYo,",
    r"\bI need (?:a|to|you)\b",
    r"\bI'd like\b",
    r"\bI would like\b",
    r"\bI want (?:a|to)\b",
    r"\bCould you\b",
    r"\bCan you\b",
    r"\bPlease (?:help|write|create)\b",
    r"\b(?:Lua|Python|Ruby|JavaScript|Node\.js) (?:script|class|method)\b",
    r"\bDockerfile\b",
    r"\bLuhn algorithm\b",
    r"\bpry['\"]? gem\b",
    r"\baverage of \d+",
    r"^[A-Z][a-z]+ (?:can normally|is (?:running|driving|walking|considering)|was)\b",  # word-problem openers
    r"\bA [A-Z][a-z]+ (?:farmer|driver|student|manager|company|team) (?:is|was|has)\b",  # "A poultry farmer is..."
]

def clean_explanation(text, max_sentences=3):
    if not isinstance(text, str):
        return text

    cleaned = text.strip()

    # Scan sentence-by-sentence rather than just searching raw text —
    # catches drift that starts partway through a sentence list.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    kept = []
    for sent in sentences:
        is_drift = any(re.search(pat, sent) for pat in DRIFT_MARKERS)
        if is_drift:
            break
        kept.append(sent)
        if len(kept) >= max_sentences:
            break

    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"[\[\{]\s*$", "", cleaned).strip()

    return cleaned
