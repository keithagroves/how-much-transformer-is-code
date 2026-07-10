"""Semantic-credit evaluation: when the rule program misses the exact next
token, was it semantically close? (e.g. predicted 'said' when truth was
'replied').

For a sample of test positions, embed predicted word and true word with
qwen3-embedding (batched, cached in word_emb.json-npy pair) and score cosine.
Reports exact accuracy, semantic-accuracy at thresholds, mean similarity, and
example near-misses so the thresholds can be eyeballed.

  usage: python3 semantic_eval.py [n_sample]
"""
import json, os, re, sys, numpy as np, requests
from collections import defaultdict

MAXN = 4
CACHE_WORDS, CACHE_EMB = "word_cache.json", "word_cache.npy"

def load_tokens(path="austen_corpus.txt"):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

def load_rules(path="rules.txt"):
    by_order = defaultdict(dict); default = "the"
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("@default"): default = line.split("=>")[1].strip(); continue
        if not line or line.startswith("#"): continue
        head, cands = line.split("=>", 1)
        _c, order, ctx = head.split("|", 2)
        by_order[int(order)][tuple(ctx.split())] = [c.strip() for c in cands.split(" :: ")]
    return by_order, default

def predict(by_order, default, ctx):
    for order in range(MAXN, 1, -1):
        key = tuple(ctx[-(order-1):])
        if key in by_order[order]: return by_order[order][key][0]
    return default

# ---- batched, cached word embeddings ----
def embed_batch(words):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b", "input": words}, timeout=300)
    return np.asarray(r.json()["embeddings"], dtype=np.float32)

def get_embeddings(words):
    words = sorted(set(words))
    if os.path.exists(CACHE_WORDS):
        cached = json.load(open(CACHE_WORDS)); emb = np.load(CACHE_EMB)
    else:
        cached, emb = [], np.zeros((0, 1024), np.float32)
    idx = {w: i for i, w in enumerate(cached)}
    todo = [w for w in words if w not in idx]
    for i in range(0, len(todo), 64):
        batch = todo[i:i+64]
        emb = np.vstack([emb, embed_batch(batch)])
        cached.extend(batch)
        print(f"  embedded {min(i+64, len(todo))}/{len(todo)} new words", file=sys.stderr)
    if todo:
        json.dump(cached, open(CACHE_WORDS, "w")); np.save(CACHE_EMB, emb)
    idx = {w: i for i, w in enumerate(cached)}
    return {w: emb[idx[w]] / np.linalg.norm(emb[idx[w]]) for w in words}

if __name__ == "__main__":
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    toks = load_tokens()
    test = toks[int(len(toks)*0.9):]
    pairs = [(test[max(0, i-3):i], test[i]) for i in range(3, len(test))]
    rng = np.random.default_rng(2)
    sample = [pairs[i] for i in rng.choice(len(pairs), n_sample, replace=False)]

    by_order, default = load_rules()
    rows = [(predict(by_order, default, ctx), truth, ctx) for ctx, truth in sample]

    # only embed real words (skip punctuation-vs-word comparisons: those are just wrong)
    is_word = lambda w: re.fullmatch(r"[a-z]+", w) is not None
    need = {w for p, t, _ in rows for w in (p, t) if is_word(w)}
    print(f"embedding {len(need)} unique words (cache warm after first run)...", file=sys.stderr)
    E = get_embeddings(need)

    sims, exact, results = [], 0, []
    for p, t, ctx in rows:
        if p == t:
            exact += 1; sims.append(1.0); continue
        if is_word(p) and is_word(t):
            s = float(E[p] @ E[t])
        else:
            s = 0.0                       # punctuation mismatch = no semantic credit
        sims.append(s); results.append((s, p, t, ctx))
    sims = np.array(sims)

    n = len(rows)
    print(f"\nsample={n}  |  exact top-1 = {exact/n:.3f}")
    for th in (0.80, 0.70, 0.60):
        print(f"  exact-or-semantic(cos>={th:.2f}) = {(sims >= th).mean():.3f}")
    print(f"  mean similarity of prediction to truth = {sims.mean():.3f}")
    print(f"  mean similarity on MISSES only          = {np.array([r[0] for r in results]).mean():.3f}")

    results.sort(key=lambda r: -r[0])
    print("\n=== best near-misses (predicted != truth, high cosine) ===")
    for s, p, t, ctx in results[:12]:
        print(f"  cos={s:.3f}  pred={p!r:<14} truth={t!r:<14} after '{' '.join(ctx)}'")
