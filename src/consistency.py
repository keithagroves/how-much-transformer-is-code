"""Experiment A: separate ministral label-noise from surrogate error.

Sample ministral K times per held-out sentence at temperature. Its self-agreement
= how confident/stable its label is. Then ask: do the frozen surrogate's errors
concentrate on ministral's LOW-agreement (ambiguous) sentences? And does fidelity
recover on the high-agreement subset?
"""
import json, numpy as np, requests
from collections import Counter
import model

MODEL = "ministral-3:3b"
K = 7
LABELS = {"positive", "negative", "neutral"}
PROMPT = ('Classify the sentiment of the text as exactly one word: positive, '
          'negative, or neutral. Reply with only that one word.\n\nText: "{}"\nSentiment:')

def sample(text):
    votes = []
    for _ in range(K):
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": MODEL, "stream": False,
            "options": {"temperature": 0.8, "seed": np.random.randint(1 << 30)},
            "prompt": PROMPT.format(text),
        }, timeout=120)
        raw = r.json()["response"].strip().lower()
        votes.append(next((l for l in LABELS if l in raw), "?"))
    return votes

data = json.load(open("labels_test.json"))
E = np.load("emb_test.npy")
pred = model.predict(E)

recs = []
for i, d in enumerate(data):
    v = sample(d["text"])
    c = Counter(v)
    maj, n = c.most_common(1)[0]
    recs.append({"text": d["text"], "t0": d["label"], "maj": maj,
                 "agree": n / K, "votes": dict(c), "pred": pred[i]})
    print(f"[{i+1:>2}/{len(data)}] agree={n}/{K} maj={maj:<8} pred={pred[i]:<8} {d['text'][:44]}")

json.dump(recs, open("consistency.json", "w"), indent=2)
maj = np.array([r["maj"] for r in recs])
agree = np.array([r["agree"] for r in recs])
pr = np.array([r["pred"] for r in recs])

print("\n=== fidelity vs ministral self-agreement ===")
for lo, hi, name in [(0.0, 1.01, "all"), (6/7, 1.01, ">=6/7 (confident)"),
                     (1.0, 1.01, "7/7 (unanimous)"), (0.0, 6/7, "<6/7 (ambiguous)")]:
    m = (agree >= lo) & (agree < hi)
    if m.any():
        print(f"  {name:<20} n={m.sum():>2}  fidelity(maj)={ (pr[m]==maj[m]).mean():.3f}")

print("\n=== does agreement differ by class? (ministral's neutral = its uncertainty?) ===")
for c in ["negative", "neutral", "positive"]:
    m = maj == c
    if m.any():
        print(f"  maj={c:<9} n={m.sum():>2}  mean agreement={agree[m].mean():.3f}")

# how often does the temperature-majority differ from the original temp-0 label?
t0 = np.array([r["t0"] for r in recs])
print(f"\nmajority != original temp-0 label on {(maj!=t0).sum()}/{len(recs)} sentences "
      f"(ministral is not even self-consistent there)")
