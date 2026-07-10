"""The refactorability / accuracy frontier.

A 'rule' = one deterministic  context -> most-likely-next-token  mapping.
The full LM is ~hundreds of thousands of these. Question: if we keep only the
top-K rules (by training frequency) as a legible program + a default fallback,
what top-1 accuracy do we get? This is the small-program vs accuracy tradeoff.
"""
import re, sys
from collections import defaultdict, Counter

MAXN = 4
def load_tokens(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START");  t = t[t.find("\n",a)+1:] if a!=-1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

toks = load_tokens(sys.argv[1] if len(sys.argv)>1 else "austen_corpus.txt")
split = int(len(toks)*0.9); train, test = toks[:split], toks[split:]

# collect every context (orders 2..4) -> next-token counts, from train
ctx_counts = defaultdict(Counter)
for i in range(len(train)):
    for order in range(2, MAXN+1):
        if i-order+1 < 0: continue
        ctx_counts[tuple(train[i-order+1:i])][train[i]] += 1

# a rule = (context -> argmax token); its strength = count of that context
rules = {}
for ctx, c in ctx_counts.items():
    tok, cnt = c.most_common(1)[0]
    rules[ctx] = (tok, sum(c.values()))
ranked = sorted(rules.items(), key=lambda kv: -kv[1][1])   # by context frequency
default = Counter(train).most_common(1)[0][0]

test_pairs = [(tuple(test[max(0,i-3):i]), test[i]) for i in range(3, len(test))]

def accuracy_with_top_k(k):
    kept = dict((ctx, tok) for ctx,(tok,_) in ranked[:k])
    correct = 0
    for ctx, truth in test_pairs:
        pred = default
        for order in range(MAXN, 1, -1):          # highest-order KEPT rule wins
            key = tuple(ctx[-(order-1):])
            if key in kept: pred = kept[key]; break
        correct += pred == truth
    return correct/len(test_pairs)

print(f"corpus={len(toks)} tok  total distinct rules={len(rules):,}  test={len(test_pairs):,}")
print(f"  default-only (0 rules): {accuracy_with_top_k(0):.3f}")
print(f"\n  {'#rules kept':>12}{'top1':>8}{'% of full':>10}")
full = accuracy_with_top_k(len(ranked))
for k in (50, 200, 1000, 5000, 20000, 100000, len(ranked)):
    a = accuracy_with_top_k(k)
    print(f"  {k:>12,}{a:>8.3f}{a/full*100:>9.0f}%")
