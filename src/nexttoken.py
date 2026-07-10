"""Next-token predictor, v0 = resumable kNN datastore.

Distill ministral's greedy next token from a qwen embedding of the prefix.
The store is a growing library of (prefix_embedding -> next_token); accuracy is
expected to climb as the store grows. Run `python3 nexttoken.py build [CAP]`
repeatedly to accumulate, then `python3 nexttoken.py eval`.
"""
import json, os, sys, numpy as np, requests
from collections import Counter

STORE_JSON, STORE_EMB = "store.json", "store_emb.npy"

# ---------- data: build prefixes from the sentence pool ----------
def sentence_pool():
    pool = list(json.load(open("labels.json")) and [d["text"] for d in json.load(open("labels.json"))])
    if os.path.exists("test_texts.json"):
        pool += json.load(open("test_texts.json"))
    return pool

def prefixes_from(sentences):
    out = []
    for s in sentences:
        w = s.split()
        for i in range(2, len(w)):            # need >=2 words of context
            out.append(" ".join(w[:i]))
    # dedup, stable order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

# ---------- model calls ----------
def next_token(prefix):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False, "raw": True,
        "options": {"temperature": 0, "num_predict": 1}, "prompt": prefix,
    }, timeout=60)
    return r.json().get("response", "")

def embed(text):
    r = requests.post("http://localhost:11434/api/embed",
                      json={"model": "qwen3-embedding:0.6b", "input": text}, timeout=60)
    return np.asarray(r.json()["embeddings"][0], dtype=np.float32)

# ---------- store ----------
def load_store():
    if os.path.exists(STORE_JSON):
        recs = json.load(open(STORE_JSON))
        emb = np.load(STORE_EMB) if os.path.exists(STORE_EMB) else np.zeros((0,1024),np.float32)
        return recs, emb
    return [], np.zeros((0,1024), np.float32)

def build(cap):
    recs, emb = load_store()
    have = {r["prefix"] for r in recs}
    todo = [p for p in prefixes_from(sentence_pool()) if p not in have]
    np.random.seed(0); np.random.shuffle(todo)
    todo = todo[:cap]
    new_e = []
    for i, p in enumerate(todo):
        tok = next_token(p)
        new_e.append(embed(p))
        recs.append({"prefix": p, "token": tok})
        if (i+1) % 25 == 0 or i+1 == len(todo):
            print(f"  +{i+1}/{len(todo)} store={len(recs)}  e.g. {p!r} -> {tok!r}", file=sys.stderr)
    emb = np.vstack([emb, np.vstack(new_e)]) if new_e else emb
    json.dump(recs, open(STORE_JSON, "w"))
    np.save(STORE_EMB, emb)
    print(f"store now holds {len(recs)} (prefix -> token) pairs", file=sys.stderr)

# ---------- kNN evaluation ----------
def knn_predict(train_e, train_t, q, k, topn=1):
    sims = train_e @ q
    idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
    votes = Counter()
    for j in idx:
        votes[train_t[j]] += sims[j]           # similarity-weighted vote
    return [t for t, _ in votes.most_common(topn)]

def eval(k=8):
    recs, emb = load_store()
    if len(recs) < 20:
        print("store too small; run build first"); return
    tok = np.array([r["token"] for r in recs])
    n = len(recs); rng = np.random.default_rng(1)
    perm = rng.permutation(n); cut = int(n*0.8)
    tr, te = perm[:cut], perm[cut:]
    freq_base = Counter(tok[tr]).most_common(1)[0][0]
    base = (tok[te] == freq_base).mean()

    def acc_at(train_idx):
        te_e = emb[te]; c1 = c3 = 0
        for q, truth in zip(te_e, tok[te]):
            top = knn_predict(emb[train_idx], tok[train_idx], q, k, topn=3)
            c1 += top[0] == truth
            c3 += truth in top
        return c1/len(te), c3/len(te)

    print(f"store={n}  train={len(tr)} test={len(te)}  |vocab|={len(set(tok))}")
    print(f"  most-frequent-token baseline (top-1): {base:.3f}")
    print(f"\n  === accuracy vs training-store size (the 'over time' curve) ===")
    print(f"  {'train_n':>8}{'top1':>8}{'top3':>8}")
    for frac in (0.1, 0.25, 0.5, 1.0):
        m = max(10, int(len(tr)*frac))
        a1, a3 = acc_at(tr[:m])
        print(f"  {m:>8}{a1:>8.3f}{a3:>8.3f}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    else:
        eval()
