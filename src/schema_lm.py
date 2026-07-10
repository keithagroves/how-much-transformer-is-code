"""Schema-conditioned rulebook: rules keyed (schema, context) with fallback to
plain context. The conditioning state is discrete, known, and semantically
massive (three sublanguages) -- the configuration where conditioning should
finally pay, unlike fuzzy open-text states.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 5

def classify(doc):
    d = doc.lower()
    w = sum(k in d for k in ("temperatures", "winds", "degrees", "cloudy", "residents"))
    p = sum(k in d for k in ("features", "costs", "ships", "weighs", "comes in"))
    r = sum(k in d for k in ("defeated", "score", "scored", "half", "match"))
    return max((w, "weather"), (p, "product"), (r, "recap"))[1]

docs = [d.strip() for d in open("structured_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 10]
schemas = [classify(d) for d in docs]
print("schema mix:", dict(Counter(schemas)), file=sys.stderr)
tok = lambda d: re.findall(r"[a-z0-9]+|[^\w\s]", d.lower())
n = len(docs)
tr_docs, te_docs = list(zip(docs[:int(n*.8)], schemas[:int(n*.8)])), \
                   list(zip(docs[int(n*.9):], schemas[int(n*.9):]))

plain, cond = defaultdict(Counter), defaultdict(Counter)
for d, s in tr_docs:
    ts = tok(d)
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0:
                k = tuple(ts[i-o+1:i])
                plain[k][ts[i]] += 1
                cond[(s, k)][ts[i]] += 1
plain = {k: c.most_common(1)[0][0] for k, c in plain.items() if sum(c.values()) >= MIN_RULE}
cond  = {k: c.most_common(1)[0][0] for k, c in cond.items()  if sum(c.values()) >= MIN_RULE}
default = "."
print(f"plain rules={len(plain):,}  conditioned={len(cond):,}", file=sys.stderr)

def predict(ts, i, s, use_cond):
    for o in range(MAXN, 1, -1):
        k = tuple(ts[max(0, i-o+1):i])
        if len(k) != o-1: continue
        if use_cond and (s, k) in cond: return cond[(s, k)]
        if k in plain: return plain[k]
    return default

win = loss = pl_hits = cd_hits = total = 0
for d, s in te_docs:
    ts = tok(d)
    for i in range(3, len(ts)):
        p0 = predict(ts, i, s, False); p1 = predict(ts, i, s, True)
        pl_hits += p0 == ts[i]; cd_hits += p1 == ts[i]; total += 1
        if p1 != p0:
            if p1 == ts[i]: win += 1
            elif p0 == ts[i]: loss += 1
print(f"\n=== schema conditioning (test n={total:,}) ===")
print(f"  plain rulebook      : {pl_hits/total:.3f}")
print(f"  schema-conditioned  : {cd_hits/total:.3f}  ({(cd_hits-pl_hits)/total:+.3f})")
from scipy.stats import binomtest
print(f"  McNemar: {win} wins / {loss} losses  (p={binomtest(win, win+loss).pvalue if win+loss else 1:.2e})")

# ---- drift demo: same seed, schema-locked generation ----
gc = defaultdict(Counter)
for d, s in tr_docs:
    ts = tok(d)
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: gc[(s, tuple(ts[i-o+1:i]))][ts[i]] += 1
grules = {k: c.most_common(5) for k, c in gc.items() if sum(c.values()) >= 3}
rng = np.random.default_rng(11)
def gen(seed, s, nn=38, temp=0.6):
    ts = tok(seed)
    for _ in range(nn):
        cands = None
        for o in range(MAXN, 1, -1):
            k = (s, tuple(ts[-(o-1):]))
            if k in grules: cands = grules[k]; break
        if not cands: cands = [(".", 1)]
        w = [x for x, _ in cands]; p = np.array([c for _, c in cands], float)**(1/temp); p /= p.sum()
        ts.append(w[rng.choice(len(w), p=p)])
    return " ".join(ts)
print("\n=== schema-locked generation (same seed 'it will') ===")
for s in ("weather", "product", "recap"):
    print(f"  [{s:<7}] {gen('it will', s)!r}")
