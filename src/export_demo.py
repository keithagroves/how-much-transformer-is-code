"""Export the real models (QA rulebook + template automaton + tone scores)
as JSON for the self-contained browser demo."""
import json, re, sys
from collections import defaultdict, Counter

# ---- QA rulebook (same build as qa_distill eval, MAXN=7) ----
MAXN = 7
pairs = json.load(open("qa_pairs.json"))
tok = lambda s: re.findall(r"[a-z0-9]+|[^\w\s]", s.lower())
stream = []
for p in pairs:
    stream += ["q", ":"] + tok(p["q"]) + ["a", ":"] + tok(p["a"]) + ["<end>"]
rules = defaultdict(Counter)
for i in range(len(stream)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: rules[tuple(stream[i-o+1:i])][stream[i]] += 1
qa_rb = {" ".join(k): c.most_common(1)[0][0] for k, c in rules.items()}
print(f"QA rules: {len(qa_rb):,}", file=sys.stderr)

# ---- template automaton from nlg_final ----
import nlg_final as NF
tpl, trans, tone = {}, {}, {}
from scales import Scale
urg = Scale("urgency")
for s in NF.templates:
    tl = sorted(NF.templates[s].items(), key=lambda kv: -kv[1])
    tpl[s] = [{"t": list(t), "c": c} for t, c in tl]
    idx = {t: i for i, (t, _) in enumerate(tl)}
    tr = {}
    for prev, nxt in NF.trans[s].items():
        pk = "START" if prev == "<START>" else str(idx.get(prev, "OTHER"))
        if pk == "OTHER": continue
        tr[pk] = {str(idx[t]): c for t, c in nxt.items() if t in idx}
    trans[s] = tr
    texts = [" ".join(t) for t, _ in tl]
    tone[s] = [round(float(x), 2) for x in urg.score(texts)]
    print(f"{s}: {len(tpl[s])} templates scored", file=sys.stderr)

json.dump({"qa": {"rb": qa_rb, "maxn": MAXN},
           "nlg": {"templates": tpl, "trans": trans, "tone": tone},
           "meta": {"qa_rules": len(qa_rb), "n_templates": sum(len(v) for v in tpl.values()),
                    "qa_pairs": len(pairs)}},
          open("demo_data.json", "w"))
print("wrote demo_data.json")
