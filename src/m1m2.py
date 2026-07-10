"""M1 + M2 from plan.md.

M1: bucketed A0 baseline -- exact top-1 by fired-rule strength bucket
    (none / thin 3-9 / medium 10-49 / strong 50+).
M2: A1 polynomial context map -- order-aware per-position vectors + low-rank
    quadratic interactions, ridge-fit (closed form) to target vectors in the
    distributional space; bucketed eval; routed A0+A1 system.

Hygiene: dedup docs, DOC-level 80/10/10 split, space+rules from train only,
lambda tuned on val, one-shot test.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter
from scipy import sparse
from scipy.sparse.linalg import svds

MAXN = 4; MIN_RULE = 3; MIN_VOCAB = 3; KDIM = 256
rng = np.random.default_rng(0)

# ---------- corpus: dedup + doc-level split ----------
docs = [d.strip() for d in open("ministral_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 50]
def shingles(doc):
    t = re.findall(r"[a-z]+", doc.lower())
    return set(hash(tuple(t[i:i+8])) for i in range(0, max(1, len(t)-8), 3))
kept, sigs = [], []
for d in docs:
    s = shingles(d)
    if not any(len(s & t)/max(1, len(s | t)) > 0.30 for t in sigs):
        kept.append(d); sigs.append(s)
print(f"docs {len(docs)} -> {len(kept)} after dedup", file=sys.stderr)
tok = lambda d: re.findall(r"[a-z]+|[^\w\s]", d.lower())
n = len(kept)
train = [t for d in kept[:int(n*.8)] for t in tok(d)]
val   = [t for d in kept[int(n*.8):int(n*.9)] for t in tok(d)]
test  = [t for d in kept[int(n*.9):] for t in tok(d)]
print(f"train={len(train):,} val={len(val):,} test={len(test):,}", file=sys.stderr)

# ---------- distributional space (train only) ----------
freq = Counter(train)
vocab = sorted(w for w, c in freq.items() if c >= MIN_VOCAB)
tid = {t: i for i, t in enumerate(vocab)}; V = len(vocab)
cooc = defaultdict(int)
for i in range(1, len(train)):
    w, p = train[i], train[i-1]
    if w in tid and p in tid: cooc[(tid[w], tid[p])] += 1
r_ = np.array([k[0] for k in cooc]); c_ = np.array([k[1] for k in cooc])
x_ = np.array(list(cooc.values()), dtype=np.float64)
M = sparse.coo_matrix((x_, (r_, c_)), shape=(V, V)).tocsr()
rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel(); Nt = M.sum()
dd = M.tocoo(); pmi = np.log((dd.data*Nt)/(rs[dd.row]*cs[dd.col])); kp = pmi > 0
P = sparse.coo_matrix((pmi[kp], (dd.row[kp], dd.col[kp])), shape=(V, V)).tocsr()
U, S, Vt = svds(P, k=KDIM)
O = (U*np.sqrt(S)); O /= (np.linalg.norm(O, axis=1, keepdims=True)+1e-9)
C = (Vt.T*np.sqrt(S)); C /= (np.linalg.norm(C, axis=1, keepdims=True)+1e-9)
O = O.astype(np.float32); C = C.astype(np.float32)
print(f"space ready V={V:,}", file=sys.stderr)

# ---------- A0 rulebook (train only) ----------
cc = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: cc[tuple(train[i-o+1:i])][train[i]] += 1
rules = {k: (sum(c.values()), c.most_common(1)[0][0]) for k, c in cc.items()
         if sum(c.values()) >= MIN_RULE}
default = Counter(train).most_common(1)[0][0]
print(f"rules={len(rules):,}", file=sys.stderr)

def fire(seg, i):
    for o in range(MAXN, 1, -1):
        k = tuple(seg[max(0, i-o+1):i])
        if len(k) == o-1 and k in rules: return rules[k]
    return None

def bucket(tot):
    if tot is None: return "none"
    return "thin(3-9)" if tot < 10 else ("med(10-49)" if tot < 50 else "strong(50+)")

# ---------- A1 features: linear per-position + low-rank quadratic ----------
R1, R2, R3 = (rng.standard_normal((KDIM, KDIM)).astype(np.float32)/np.sqrt(KDIM)
              for _ in range(3))
def phi(seg, i):
    cs_ = []
    for j in (1, 2, 3):
        w = seg[i-j] if i-j >= 0 else None
        cs_.append(C[tid[w]] if w in tid else np.zeros(KDIM, np.float32))
    c1, c2, c3 = cs_
    q12 = (R1 @ c1) * (R2 @ c2); q13 = (R1 @ c1) * (R3 @ c3); q23 = (R2 @ c2) * (R3 @ c3)
    return np.concatenate([c1, c2, c3, q12, q13, q23, [1.0]]).astype(np.float32)

D = 6*KDIM + 1
# accumulate normal equations over train
XtX = np.zeros((D, D), np.float64); XtY = np.zeros((D, KDIM), np.float64)
B = 8192; buf, tgt = [], []
def flush():
    global XtX, XtY, buf, tgt
    if not buf: return
    Xb = np.stack(buf); Yb = np.stack(tgt)
    XtX += Xb.T.astype(np.float64) @ Xb.astype(np.float64)
    XtY += Xb.T.astype(np.float64) @ Yb.astype(np.float64)
    buf, tgt = [], []
cnt = 0
for i in range(3, len(train)):
    if train[i] not in tid: continue
    buf.append(phi(train, i)); tgt.append(O[tid[train[i]]]); cnt += 1
    if len(buf) >= B: flush()
flush()
print(f"A1 normal equations from {cnt:,} examples", file=sys.stderr)

def solve(lam):
    A = XtX + lam*np.eye(D)
    return np.linalg.solve(A, XtY).astype(np.float32)

def eval_poly(W, seg, idxs):
    hits = 0
    for s in range(0, len(idxs), 4096):
        chunk = idxs[s:s+4096]
        X = np.stack([phi(seg, i) for i in chunk])
        T = X @ W
        T /= (np.linalg.norm(T, axis=1, keepdims=True)+1e-9)
        best = (T @ O.T).argmax(1)
        hits += sum(vocab[b] == seg[i] for b, i in zip(best, chunk))
    return hits/len(idxs)

# tune lambda on val (in-vocab positions)
val_idx = [i for i in range(3, len(val)) if val[i] in tid]
sub = val_idx[::max(1, len(val_idx)//8000)]
best_lam, best_acc = None, -1
for lam in (1.0, 10.0, 100.0, 1000.0):
    W = solve(lam)
    a = eval_poly(W, val, sub)
    print(f"  lambda={lam:<7} val_top1={a:.4f}", file=sys.stderr)
    if a > best_acc: best_acc, best_lam = a, lam
W = solve(best_lam)
print(f"chosen lambda={best_lam}", file=sys.stderr)

# ---------- bucketed test eval: A0, A1, routed ----------
buckets = defaultdict(lambda: {"n":0, "a0":0, "a1":0})
test_idx = [i for i in range(3, len(test))]
for s in range(0, len(test_idx), 4096):
    chunk = test_idx[s:s+4096]
    X = np.stack([phi(test, i) for i in chunk])
    T = X @ W; T /= (np.linalg.norm(T, axis=1, keepdims=True)+1e-9)
    best = (T @ O.T).argmax(1)
    for b, i in zip(best, chunk):
        fr = fire(test, i)
        bk = bucket(fr[0] if fr else None)
        a0 = (fr[1] if fr else default) == test[i]
        a1 = vocab[b] == test[i]
        d = buckets[bk]; d["n"] += 1; d["a0"] += a0; d["a1"] += a1

print("\n=== M1/M2: bucketed test results (A0 rules vs A1 polynomial) ===")
print(f"{'bucket':<13}{'n':>8}{'A0 rules':>10}{'A1 poly':>10}{'routed':>9}")
tot_n = tot_a0 = tot_rt = 0
for bk in ("strong(50+)", "med(10-49)", "thin(3-9)", "none"):
    d = buckets[bk]
    if d["n"] == 0: continue
    rt = d["a0"] if bk in ("strong(50+)", "med(10-49)") else d["a1"]  # route: rules if med+, poly if thin/none
    print(f"{bk:<13}{d['n']:>8,}{d['a0']/d['n']:>10.3f}{d['a1']/d['n']:>10.3f}{rt/d['n']:>9.3f}")
    tot_n += d["n"]; tot_a0 += d["a0"]; tot_rt += rt
print(f"{'OVERALL':<13}{tot_n:>8,}{tot_a0/tot_n:>10.3f}{'':>10}{tot_rt/tot_n:>9.3f}")
