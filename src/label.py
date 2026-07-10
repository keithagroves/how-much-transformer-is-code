"""Ground-truth labels from the small model (ministral-3:3b).

This model IS the predictor we are reverse-engineering: its single-word output
token is the constrained "token prediction" our surrogate must replicate.
"""
import json, sys, requests
from corpus import TEXTS

MODEL = "ministral-3:3b"
LABELS = {"positive", "negative", "neutral"}
PROMPT = (
    "Classify the sentiment of the text as exactly one word: "
    "positive, negative, or neutral. Reply with only that one word.\n\n"
    'Text: "{text}"\nSentiment:'
)

def classify(text):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": MODEL, "stream": False,
        "options": {"temperature": 0},
        "prompt": PROMPT.format(text=text),
    }, timeout=120)
    raw = r.json()["response"].strip().lower()
    for lab in LABELS:                      # tolerate stray punctuation/casing
        if lab in raw:
            return lab
    return "?" + raw

def main():
    out = []
    for i, t in enumerate(TEXTS):
        lab = classify(t)
        out.append({"text": t, "label": lab})
        print(f"[{i+1:>2}/{len(TEXTS)}] {lab:<9} {t[:56]}", file=sys.stderr)
    with open("labels.json", "w") as f:
        json.dump(out, f, indent=2)
    from collections import Counter
    print("\nlabel counts:", dict(Counter(d["label"] for d in out)), file=sys.stderr)

if __name__ == "__main__":
    main()
