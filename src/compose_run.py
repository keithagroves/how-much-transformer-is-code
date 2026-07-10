"""Label the compositional corpus with ministral (base model) + embed with qwen.
Then the first diagnostic: does ministral flip sentiment under negation? If not,
the task is not compositional for it and the whole test is moot.
"""
import json, numpy as np, sys
from label import classify
from embed import embed

rows = json.load(open("compose.json"))
E = []
for i, r in enumerate(rows):
    r["label"] = classify(r["text"])
    E.append(embed(r["text"]))
    print(f"[{i+1:>2}/{len(rows)}] lex={r['lex']:+d} neg={r['neg']} {r['label']:<8} {r['text']}", file=sys.stderr)

np.save("compose_emb.npy", np.vstack(E).astype(np.float32))
json.dump(rows, open("compose_labeled.json", "w"), indent=2)

# --- cross-tab: ministral label by (lex, neg) cell ---
from collections import Counter
print("\n=== ministral label distribution per cell ===")
print(f"  {'cell':<22}{'positive':>10}{'negative':>10}{'neutral':>9}")
for lex in (+1, -1):
    for neg in (0, 1):
        c = Counter(r["label"] for r in rows if r["lex"] == lex and r["neg"] == neg)
        name = f"lex={'+' if lex>0 else '-'} neg={neg} " + \
               ({(1,0):'(great)',(1,1):'(not great)',(-1,0):'(terrible)',(-1,1):'(not terrible)'}[(lex,neg)])
        print(f"  {name:<22}{c.get('positive',0):>10}{c.get('negative',0):>10}{c.get('neutral',0):>9}")
