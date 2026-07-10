"""Structured vs open-domain distillability, matched corpus sizes.

Open-domain reference points (ministral stories, doc-split, MIN_RULE=5):
  37.8k tok = 0.157 | 75.6k = 0.169   (top-1, fixed test)

Same recipe on the structured corpus, plus rule coverage and a sample
generation with trace to show what structure buys.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 5

docs = [d.strip() for d in open("structured_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 10]
tok = lambda d: re.findall(r"[a-z0-9]+|[^\w\s]", d.lower())
n = len(docs)
train = [t for d in docs[:int(n*.8)] for t in tok(d)]
test  = [t for d in docs[int(n*.9):] for t in tok(d)]
print(f"docs={n} train={len(train):,} test={len(test):,} "
      f"type/token={len(set(train))/len(train):.4f}", file=sys.stderr)

pairs = [(test[max(0,i-3):i], test[i]) for i in range(3, len(test))]

def build_eval(tr):
    cc = defaultdict(Counter)
    for i in range(len(tr)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: cc[tuple(tr[i-o+1:i])][tr[i]] += 1
    rules = {k: c.most_common(1)[0][0] for k, c in cc.items() if sum(c.values()) >= MIN_RULE}
    dflt = Counter(tr).most_common(1)[0][0]
    hits = cov = 0
    for ctx, truth in pairs:
        p, fired = dflt, False
        for o in range(MAXN, 1, -1):
            k = tuple(ctx[-(o-1):])
            if k in rules: p, fired = rules[k], True; break
        hits += p == truth; cov += fired
    return len(rules), hits/len(pairs), cov/len(pairs), rules, dflt

print(f"\n{'train_tok':>10}{'rules':>8}{'top1':>8}{'coverage':>10}   (open-domain ref: 38k=0.157, 76k=0.169)")
for frac in (0.5, 1.0):
    tr = train[:int(len(train)*frac)]
    r, a, c, rules, dflt = build_eval(tr)
    print(f"{len(tr):>10,}{r:>8,}{a:>8.3f}{c:>10.1%}")

# sample generation with the full structured rulebook
rng = np.random.default_rng(4)
cc = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: cc[tuple(train[i-o+1:i])][train[i]] += 1
grules = {k: c.most_common(6) for k, c in cc.items() if sum(c.values()) >= 3}
def sample(seed, nn=30, temp=0.6):
    ts = tok(seed)
    for _ in range(nn):
        cands = None
        for o in range(MAXN, 1, -1):
            k = tuple(ts[-(o-1):])
            if k in grules: cands = grules[k]; break
        if not cands: cands = [(".", 1)]
        w = [x for x,_ in cands]; p = np.array([c for _,c in cands], float)**(1/temp); p/=p.sum()
        ts.append(w[rng.choice(len(w), p=p)])
    return " ".join(ts)
print("\n=== structured generation samples ===")
for s in ("temperatures will reach", "the turning point came when", "it features"):
    print(f"  {sample(s)!r}")
