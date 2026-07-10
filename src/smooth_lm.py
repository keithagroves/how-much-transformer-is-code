"""Geometric fallback: compose predictions for UNSEEN contexts from the
nearest rule-bearing contexts in distributional space.

From one SVD of PPMI(next-token, prev-word):
  O = U*sqrt(S)  -- tokens as prediction TARGETS (what precedes them)
  C = V*sqrt(S)  -- tokens as CONTEXTS (what follows them)

A context = recency-weighted mean of its tokens' C-vectors. Where no lexical
rule fires (the dictionary cliff, currently -> default ','), find kNN rule
contexts, blend their target vectors, read out nearest O tokens.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter
from scipy import sparse
from scipy.sparse.linalg import svds

MAXN = 4; MIN = 10; KDIM = 256; KNN = 25
W_RECENCY = np.array([0.25, 0.5, 1.0])          # weights for last-3 tokens

def load(p):
    t = open(p, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    b = t.rfind("*** END");  t = t[:b] if b != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

austen = load("austen_corpus.txt")
c80, c90 = int(len(austen)*0.8), int(len(austen)*0.9)
train = austen[:c80]
for pid in (98, 1400, 766, 1260, 768, 145, 4276, 599, 2701):
    train += load(f"pg{pid}.txt")
test = austen[c90:]

freq = Counter(train)
vocab = sorted(w for w, c in freq.items() if c >= 5)
tid = {t: i for i, t in enumerate(vocab)}; V = len(vocab)

cooc = defaultdict(int)
for i in range(1, len(train)):
    w, p = train[i], train[i-1]
    if w in tid and p in tid: cooc[(tid[w], tid[p])] += 1
r = np.array([k[0] for k in cooc]); c = np.array([k[1] for k in cooc])
x = np.array(list(cooc.values()), dtype=np.float64)
M = sparse.coo_matrix((x, (r, c)), shape=(V, V)).tocsr()
row_s = np.asarray(M.sum(1)).ravel(); col_s = np.asarray(M.sum(0)).ravel(); N = M.sum()
d = M.tocoo(); pmi = np.log((d.data * N) / (row_s[d.row] * col_s[d.col])); keep = pmi > 0
P = sparse.coo_matrix((pmi[keep], (d.row[keep], d.col[keep])), shape=(V, V)).tocsr()
U, S, Vt = svds(P, k=KDIM)
O = (U * np.sqrt(S)).astype(np.float32)
C = (Vt.T * np.sqrt(S)).astype(np.float32)
O /= (np.linalg.norm(O, axis=1, keepdims=True) + 1e-9)
C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
print("O and C embeddings ready", file=sys.stderr)

# ---- rules with targets ----
ctx_counts = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: ctx_counts[tuple(train[i-o+1:i])][train[i]] += 1
rules, targets, rule_ctx_vec = {}, [], []
def cvec(ctx_tokens):
    vs, ws = [], []
    for t, w in zip(ctx_tokens[-3:], W_RECENCY[-len(ctx_tokens[-3:]):]):
        if t in tid: vs.append(C[tid[t]]); ws.append(w)
    if not vs: return None
    v = np.average(vs, axis=0, weights=ws)
    n = np.linalg.norm(v)
    return (v/n).astype(np.float32) if n > 0 else None
for k, cc in ctx_counts.items():
    tot = sum(cc.values())
    if tot < MIN: continue
    v = np.zeros(KDIM, np.float32); m = 0
    for w, n2 in cc.items():
        if w in tid: v += n2 * O[tid[w]]; m += n2
    nv = np.linalg.norm(v)
    if not m or nv == 0: continue
    cv = cvec(list(k))
    if cv is None: continue
    rules[k] = len(targets)
    targets.append(v/nv); rule_ctx_vec.append(cv)
T = np.stack(targets); RC = np.stack(rule_ctx_vec)
default = Counter(train).most_common(1)[0][0]
tok_rule = {k: None for k in rules}
for k in rules:  tok_rule[k] = ctx_counts[k].most_common(1)[0][0]
print(f"rules with targets: {len(rules):,}", file=sys.stderr)

# ---- eval on the DICTIONARY CLIFF: positions where no rule fires ----
cliff, served = [], 0
for i in range(3, len(test)):
    hit = False
    for o in range(MAXN, 1, -1):
        k = tuple(test[max(0, i-o+1):i])
        if len(k) == o-1 and k in rules: hit = True; break
    if hit: served += 1
    else: cliff.append(i)
print(f"test positions: {served+len(cliff):,}; cliff (no rule): {len(cliff):,} "
      f"({len(cliff)/(served+len(cliff)):.1%})", file=sys.stderr)

base_hits = geo_hits = top10_hits = evaluable = 0
for i in cliff:
    truth = test[i]
    base_hits += (default == truth)
    q = cvec(test[max(0, i-3):i])
    if q is None or truth not in tid: continue
    evaluable += 1
    sims = RC @ q
    nn = np.argpartition(-sims, KNN)[:KNN]
    w = np.clip(sims[nn], 0, None)
    tgt = (T[nn] * w[:, None]).sum(0)
    n2 = np.linalg.norm(tgt)
    if n2 == 0: continue
    tgt /= n2
    scores = O @ tgt
    best = int(scores.argmax())
    geo_hits += (vocab[best] == truth)
    top10 = np.argpartition(-scores, 10)[:10]
    top10_hits += (tid[truth] in set(top10.tolist()))

print(f"\n=== dictionary-cliff positions (n={len(cliff):,}, evaluable {evaluable:,}) ===")
print(f"  default token ({default!r})     : {base_hits/len(cliff):.3f}")
print(f"  geometric composed rules      : {geo_hits/evaluable:.3f}   top10={top10_hits/evaluable:.3f}")
total = served + len(cliff)
print(f"\noverall top-1 delta if adopted: {(geo_hits - base_hits)/total:+.4f}")
