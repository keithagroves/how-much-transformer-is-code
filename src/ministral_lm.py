"""In-domain rebuild: the full apparatus on ministral-generated text.

Train on ministral's own output, test on held-out ministral output --
distilling the model that generated the data. Builds token rulebook +
distributional space + vector rulebook; reports all metrics side by side.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter
from scipy import sparse
from scipy.sparse.linalg import svds

MAXN = 4; MIN_RULE = 5; MIN_VOCAB = 3; KDIM = 256   # thresholds scaled to 150k corpus

toks = re.findall(r"[a-z]+|[^\w\s]",
                  open("ministral_corpus.txt", encoding="utf-8").read()
                  .replace("<|doc|>", " ").lower())
n = len(toks)
train, val, test = toks[:int(n*.8)], toks[int(n*.8):int(n*.9)], toks[int(n*.9):]
tt = len(set(train))/len(train)
print(f"tokens={n:,} (train {len(train):,})  type/token={tt:.4f}", file=sys.stderr)

freq = Counter(train)
vocab = sorted(w for w, c in freq.items() if c >= MIN_VOCAB)
tid = {t: i for i, t in enumerate(vocab)}; V = len(vocab)
print(f"vocab={V:,}", file=sys.stderr)

# ---- distributional space ----
cooc = defaultdict(int)
for i in range(1, len(train)):
    w, p = train[i], train[i-1]
    if w in tid and p in tid: cooc[(tid[w], tid[p])] += 1
r = np.array([k[0] for k in cooc]); c = np.array([k[1] for k in cooc])
x = np.array(list(cooc.values()), dtype=np.float64)
M = sparse.coo_matrix((x, (r, c)), shape=(V, V)).tocsr()
rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel(); N = M.sum()
d = M.tocoo(); pmi = np.log((d.data*N)/(rs[d.row]*cs[d.col])); keep = pmi > 0
P = sparse.coo_matrix((pmi[keep], (d.row[keep], d.col[keep])), shape=(V, V)).tocsr()
k_svd = min(KDIM, V-1)
U, S, _ = svds(P, k=k_svd)
O = (U*np.sqrt(S)).astype(np.float32)
O /= (np.linalg.norm(O, axis=1, keepdims=True)+1e-9)

# ---- rules: token argmax + vector target ----
cc = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: cc[tuple(train[i-o+1:i])][train[i]] += 1
rule_id, tokp, targets = {}, [], []
for k, cnt in cc.items():
    tot = sum(cnt.values())
    if tot < MIN_RULE: continue
    v = np.zeros(k_svd, np.float32); m = 0
    for w, n2 in cnt.items():
        if w in tid: v += n2*O[tid[w]]; m += n2
    nv = np.linalg.norm(v)
    if not m or nv == 0: continue
    rule_id[k] = len(tokp)
    tokp.append(cnt.most_common(1)[0][0]); targets.append(v/nv)
T = np.stack(targets)
default = Counter(train).most_common(1)[0][0]
print(f"rules={len(rule_id):,}", file=sys.stderr)

def fire(seg, i):
    for o in range(MAXN, 1, -1):
        k = tuple(seg[max(0, i-o+1):i])
        if len(k) == o-1 and k in rule_id: return rule_id[k]
    return None

pos = [(i, fire(test, i)) for i in range(3, len(test))]
served = sum(1 for _, j in pos if j is not None)
pos_v = [(i, j) for i, j in pos if j is not None and test[i] in tid]

# token rulebook on ALL positions (default where no rule)
tok_hits = sum((tokp[j] if j is not None else default) == test[i] for i, j in pos)
print(f"\n=== ministral in-domain (test n={len(pos):,}, rule coverage {served/len(pos):.0%}) ===")
print(f"  token rulebook exact (all positions): {tok_hits/len(pos):.3f}")

Tm = np.stack([T[j] for _, j in pos_v])
truth = np.array([tid[test[i]] for i, _ in pos_v])
tokm = np.array([tid.get(tokp[j], -1) for _, j in pos_v])
ex_v = ex_t = r10 = 0; preds = []
for s in range(0, len(pos_v), 4096):
    sims = Tm[s:s+4096] @ O.T
    best = sims.argmax(1); preds.append(best)
    top10 = np.argpartition(-sims, min(10, V-1), axis=1)[:, :10]
    tr = truth[s:s+4096]
    ex_v += (best == tr).sum(); r10 += (top10 == tr[:, None]).any(1).sum()
    ex_t += (tokm[s:s+4096] == tr).sum()
m = len(pos_v); preds = np.concatenate(preds)
print(f"  (rule-fired, in-vocab subset n={m:,})")
print(f"  token rulebook   exact={ex_t/m:.3f}")
print(f"  vector rulebook  exact={ex_v/m:.3f}  top10={r10/m:.3f}  distinct={len(set(preds.tolist()))}")

print("\n=== regions (ministral space) ===")
for probe in [("she","could","not"), ("i","am"), (",","and"), ("of","the"), ("he","said")]:
    if probe in rule_id:
        v = T[rule_id[probe]]
        idx = np.argsort(-(O @ v))[:6]
        print(f"  after {' '.join(probe)!r}: " + ", ".join(vocab[j] for j in idx))
