"""Semantic credit, done honestly: only CONTENT words get similarity credit.
Function words / punctuation are grammar -- there, exact match or nothing.
(v1 showed qwen packs function words into one tight cluster: 'and'~'on' at
cos 0.95, which is orthographic soup, not synonymy.)
"""
import json, re, sys, numpy as np
from collections import defaultdict
from semantic_eval import (load_tokens, load_rules, predict, get_embeddings, MAXN)
from generalize import TOK2POS          # closed-class function-word lists

is_word = lambda w: re.fullmatch(r"[a-z]+", w) is not None
def kind(w):
    if not is_word(w): return "punct"
    return "func" if w in TOK2POS else "content"

n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
toks = load_tokens()
test = toks[int(len(toks)*0.9):]
pairs = [(test[max(0, i-3):i], test[i]) for i in range(3, len(test))]
rng = np.random.default_rng(2)
sample = [pairs[i] for i in rng.choice(len(pairs), n_sample, replace=False)]
by_order, default = load_rules()
rows = [(predict(by_order, default, ctx), truth, ctx) for ctx, truth in sample]

need = {w for p, t, _ in rows for w in (p, t) if is_word(w)}
E = get_embeddings(need)

# split test positions by the KIND of the true next token
buckets = defaultdict(list)
for p, t, ctx in rows:
    buckets[kind(t)].append((p, t, ctx))

print(f"sample={len(rows)}  truth-kind mix: " +
      "  ".join(f"{k}:{len(v)/len(rows):.0%}" for k, v in sorted(buckets.items())))

print("\n=== exact top-1 by kind of true token ===")
for k in ("punct", "func", "content"):
    b = buckets[k]
    print(f"  {k:<8} n={len(b):>4}  exact={sum(p==t for p,t,_ in b)/len(b):.3f}")

# semantic credit only where it's meaningful: truth AND prediction both content
b = buckets["content"]
exact = sum(p == t for p, t, _ in b)
sims, misses = [], []
for p, t, ctx in b:
    if p == t: sims.append(1.0); continue
    s = float(E[p] @ E[t]) if kind(p) == "content" else 0.0
    sims.append(s); misses.append((s, p, t, ctx))
sims = np.array(sims)
print(f"\n=== content-word positions only (n={len(b)}) ===")
print(f"  exact                      = {exact/len(b):.3f}")
for th in (0.80, 0.70, 0.60):
    print(f"  exact-or-semantic(>= {th:.2f}) = {(sims >= th).mean():.3f}")

misses.sort(key=lambda r: -r[0])
print("\n=== best content-word near-misses ===")
for s, p, t, ctx in misses[:12]:
    print(f"  cos={s:.3f}  pred={p!r:<13} truth={t!r:<13} after '{' '.join(ctx)}'")
