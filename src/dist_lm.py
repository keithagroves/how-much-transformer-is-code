"""Vector rulebook v2: point into a DISTRIBUTIONAL space (organized by
predictability), not qwen semantic space (organized by meaning).

Output embeddings: token x preceding-word PPMI matrix -> truncated SVD (k=256),
built from the training corpus only. A token's neighbors = tokens that follow
the same contexts. Rules store count-weighted mean of next-token output
embeddings; predict = nearest output embeddings to the target.

Scoreboard (same eval protocol as vector_lm.py):
  token rulebook: exact 0.195 | qwen-space vectors: 0.143, 440 distinct (hub soup)
"""
import re, sys, numpy as np
from collections import defaultdict, Counter
from scipy import sparse
from scipy.sparse.linalg import svds

MAXN = 4; MIN = 10; KDIM = 256

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
val, test = austen[c80:c90], austen[c90:]

# ---- vocabulary: all tokens freq>=5 (words AND punctuation) ----
freq = Counter(train)
vocab = sorted(w for w, c in freq.items() if c >= 5)
tid = {t: i for i, t in enumerate(vocab)}
V = len(vocab)
print(f"train={len(train):,}  vocab={V:,}", file=sys.stderr)

# ---- distributional output embeddings: PPMI(token, prev-word) -> SVD ----
rows, cols, vals = [], [], []
cooc = defaultdict(int)
for i in range(1, len(train)):
    w, p = train[i], train[i-1]
    if w in tid and p in tid: cooc[(tid[w], tid[p])] += 1
r = np.array([k[0] for k in cooc]); c = np.array([k[1] for k in cooc])
x = np.array(list(cooc.values()), dtype=np.float64)
M = sparse.coo_matrix((x, (r, c)), shape=(V, V)).tocsr()
# PPMI
row_s = np.asarray(M.sum(1)).ravel(); col_s = np.asarray(M.sum(0)).ravel(); N = M.sum()
data = M.tocoo()
pmi = np.log((data.data * N) / (row_s[data.row] * col_s[data.col]))
keep = pmi > 0
P = sparse.coo_matrix((pmi[keep], (data.row[keep], data.col[keep])), shape=(V, V)).tocsr()
print(f"PPMI nnz={P.nnz:,}; running SVD k={KDIM}...", file=sys.stderr)
U, S, _ = svds(P, k=KDIM)
O = U * np.sqrt(S)                                     # (V, KDIM)
O = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-9)
O = O.astype(np.float32)
print("output embeddings ready", file=sys.stderr)

# sanity: neighbors should be same-slot tokens
def near(w, k=6):
    v = O[tid[w]]; idx = np.argsort(-(O @ v))[1:k+1]
    return [vocab[j] for j in idx]
print("\n=== distributional neighborhoods (sanity) ===")
for w in ("and", "she", "said", "house", "beautiful", ","):
    if w in tid: print(f"  {w:<10} -> {', '.join(near(w))}")

# ---- rules: target vector in O-space ----
ctx_counts = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: ctx_counts[tuple(train[i-o+1:i])][train[i]] += 1
contexts, tok_rule, targets = [], [], []
for k, cc in ctx_counts.items():
    tot = sum(cc.values())
    if tot < MIN: continue
    contexts.append(k); tok_rule.append(cc.most_common(1)[0][0])
    v = np.zeros(KDIM, np.float32); m = 0
    for w, n2 in cc.items():
        if w in tid: v += n2 * O[tid[w]]; m += n2
    nv = np.linalg.norm(v)
    targets.append(v/nv if m and nv > 0 else None)
rule_id = {k: j for j, k in enumerate(contexts)}
print(f"\nrules: {len(contexts):,}", file=sys.stderr)

def fire(seg, i):
    for o in range(MAXN, 1, -1):
        k = tuple(seg[max(0, i-o+1):i])
        if len(k) == o-1 and k in rule_id: return rule_id[k]
    return None

pos = [(i, fire(test, i)) for i in range(3, len(test))]
pos = [(i, j) for i, j in pos if j is not None and targets[j] is not None and test[i] in tid]
T = np.stack([targets[j] for _, j in pos])
truth = np.array([tid[test[i]] for i, _ in pos])
tokp = np.array([tid.get(tok_rule[j], -1) for _, j in pos])

ex_v = ex_t = r10 = 0; preds = []
for s in range(0, len(pos), 4096):
    sims = T[s:s+4096] @ O.T
    best = sims.argmax(1); preds.append(best)
    top10 = np.argpartition(-sims, 10, axis=1)[:, :10]
    tr = truth[s:s+4096]
    ex_v += (best == tr).sum(); r10 += (top10 == tr[:, None]).any(1).sum()
    ex_t += (tokp[s:s+4096] == tr).sum()
preds = np.concatenate(preds)
n = len(pos)
print(f"\n=== test (n={n:,}) ===")
print(f"  token rulebook          exact={ex_t/n:.3f}")
print(f"  DIST-space vector rule  exact={ex_v/n:.3f}  top10={r10/n:.3f}  "
      f"distinct-preds={len(set(preds.tolist()))}")
print("\n=== rules as regions (dist space) ===")
for probe in [("she","could","not"), ("i","am"), (",","and"), ("of","the")]:
    if probe in rule_id and targets[rule_id[probe]] is not None:
        v = targets[rule_id[probe]]
        idx = np.argsort(-(O @ v))[:6]
        print(f"  after {' '.join(probe)!r}: " + ", ".join(vocab[j] for j in idx))
