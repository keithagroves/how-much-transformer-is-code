"""Score the FROZEN surrogate on the author-independent held-out set.
Pipeline: glm/gemma-authored texts -> ministral labels -> qwen embeddings ->
frozen model.predict. Reports fidelity, per-class axis structure, disagreements.
"""
import json, numpy as np
from collections import Counter
from label import classify          # ministral ground-truth
from embed import embed             # qwen embeddings
import model                        # frozen surrogate (fit on original 72)

texts = json.load(open("test_texts.json"))
rows = []
for i, t in enumerate(texts):
    lab = classify(t)
    if lab.startswith("?"):         # unparseable ministral output -> skip
        print(f"  skip (bad label {lab!r}): {t}")
        continue
    rows.append({"text": t, "label": lab, "emb": embed(t)})
    print(f"[{i+1:>2}/{len(texts)}] {lab:<9} {t[:52]}")

y = np.array([r["label"] for r in rows])
E = np.vstack([r["emb"] for r in rows]).astype(np.float32)
np.save("emb_test.npy", E)
json.dump([{"text": r["text"], "label": r["label"]} for r in rows],
          open("labels_test.json", "w"), indent=2)

pred = model.predict(E)
fid = (pred == y).mean()
print(f"\nheld-out label balance: {dict(Counter(y))}")
print(f"=== FROZEN surrogate fidelity on unseen, author-independent data = "
      f"{fid:.3f}  ({(pred==y).sum()}/{len(y)}) ===")
print(f"    (original LOO-CV was 0.944; self train 0.972)")

pol, ev = model.scores(E)
print("\n=== does the 2-axis structure hold on new data? (mean +/- std) ===")
for c in ["negative", "neutral", "positive"]:
    m = y == c
    if m.any():
        print(f"  {c:<9} n={m.sum():>2}  polarity {pol[m].mean():+.3f}+/-{pol[m].std():.3f}"
              f"   evaluativeness {ev[m].mean():+.3f}+/-{ev[m].std():.3f}")

print("\n=== disagreements (frozen surrogate vs ministral) ===")
for i in np.where(pred != y)[0]:
    print(f"  said {pred[i]:<8} truth {y[i]:<8} pol={pol[i]:+.3f} ev={ev[i]:+.3f} | {rows[i]['text']}")
