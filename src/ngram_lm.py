"""A real next-token predictor: word-level n-gram backoff LM.
Predicts the actual next token in held-out text. Accuracy provably climbs with
training size -- the 'reach accuracy over time' engine, done the way it works.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4                                   # up to 4-gram (3 tokens of context)

def load_tokens(path="pg1342.txt"):
    text = open(path, encoding="utf-8", errors="ignore").read()
    # drop Gutenberg boilerplate
    a = text.find("*** START"); b = text.rfind("*** END")
    if a != -1: text = text[text.find("\n", a)+1:]
    if b != -1: text = text[:b]
    toks = re.findall(r"[a-z]+|[^\w\s]", text.lower())
    return toks

def build(tokens):
    tables = [defaultdict(Counter) for _ in range(MAXN)]     # order 1..MAXN
    for i in range(len(tokens)):
        for order in range(1, MAXN+1):
            if i-order+1 < 1: continue
            ctx = tuple(tokens[i-order+1:i]); nxt = tokens[i]
            tables[order-1][ctx][nxt] += 1
    return tables

def predict(tables, ctx, topn=3):
    for order in range(MAXN, 0, -1):        # stupid backoff: highest seen order wins
        key = tuple(ctx[-(order-1):]) if order > 1 else ()
        c = tables[order-1].get(key)
        if c:
            return [t for t, _ in c.most_common(topn)], order
    return [], 0

def evaluate(train_tok, test_ctx_next, label):
    tables = build(train_tok)
    c1 = c3 = 0; used = Counter()
    for ctx, truth in test_ctx_next:
        top, order = predict(tables, ctx)
        used[order] += 1
        if top:
            c1 += top[0] == truth
            c3 += truth in top
    n = len(test_ctx_next)
    print(f"  {label:>10}  top1={c1/n:.3f}  top3={c3/n:.3f}   "
          f"(backoff order used: " +
          " ".join(f"{o}:{used[o]/n:.0%}" for o in (4,3,2,1)) + ")")
    return c1/n, c3/n

if __name__ == "__main__":
    toks = load_tokens(sys.argv[1] if len(sys.argv) > 1 else "pg1342.txt")
    split = int(len(toks)*0.9)
    train_all, test = toks[:split], toks[split:]
    # fixed test set of (context, next) pairs
    test_pairs = [(test[max(0,i-3):i], test[i]) for i in range(3, len(test))]
    print(f"corpus={len(toks)} tokens  vocab={len(set(toks))}  "
          f"train={len(train_all)} test_positions={len(test_pairs)}")
    print("\n=== accuracy vs training size (the 'over time' curve) ===")
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        m = int(len(train_all)*frac)
        evaluate(train_all[:m], test_pairs, f"{m} tok")
