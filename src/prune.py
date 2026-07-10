"""Prune the rule program: keep only rules that are RELIABLE, decided on a
validation split (never the test set).

reliability = count (how often the context was seen)  AND
              purity (fraction of the time it took its top token).

We tune (min_count, min_purity) on validation, then apply to a model built on
all non-test data and report held-out TEST accuracy + rule-count reduction.
"""
import re, sys, itertools
from collections import defaultdict, Counter

MAXN = 4
def load_tokens(path="austen_corpus.txt"):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

def build(tokens):
    ctx = defaultdict(Counter)
    for i in range(len(tokens)):
        for order in range(2, MAXN+1):
            if i-order+1 >= 0:
                ctx[tuple(tokens[i-order+1:i])][tokens[i]] += 1
    # context -> (top_token, count, purity)
    out = {}
    for k, c in ctx.items():
        tok, tc = c.most_common(1)[0]; tot = sum(c.values())
        out[k] = (tok, tot, tc/tot)
    return out

def prune(rules, cmin, pmin):
    return {k: v[0] for k, v in rules.items() if v[1] >= cmin and v[2] >= pmin}

def score(kept, default, pairs):
    c = 0
    for ctx, truth in pairs:
        pred = default
        for order in range(MAXN, 1, -1):
            key = tuple(ctx[-(order-1):])
            if key in kept: pred = kept[key]; break
        c += pred == truth
    return c/len(pairs)

toks = load_tokens()
n = len(toks)
build_tok = toks[:int(n*0.8)]
val_tok   = toks[int(n*0.8):int(n*0.9)]
test_tok  = toks[int(n*0.9):]
default = Counter(build_tok).most_common(1)[0][0]
pairs = lambda seg: [(seg[max(0,i-3):i], seg[i]) for i in range(3, len(seg))]
val_pairs, test_pairs = pairs(val_tok), pairs(test_tok)

rules_b = build(build_tok)
print(f"full rules (build set): {len(rules_b):,}")
print("\n=== tuning (min_count, min_purity) on validation ===")
best = (-1, None)
for cmin, pmin in itertools.product((1,2,3,5,10), (0.0,0.3,0.5,0.7)):
    kept = prune(rules_b, cmin, pmin)
    a = score(kept, default, val_pairs)
    if a > best[0]: best = (a, (cmin, pmin, len(kept)))
    print(f"  count>={cmin:<2} purity>={pmin:<3}  rules={len(kept):>7,}  val_top1={a:.3f}")
_, (cmin, pmin, _) = best
print(f"  -> best on val: count>={cmin}, purity>={pmin}")

# final model on all non-test data, apply chosen policy, report on TEST
rules_f = build(build_tok + val_tok)
full_test = score({k:v[0] for k,v in rules_f.items()}, default, test_pairs)
kept_f = prune(rules_f, cmin, pmin)
pruned_test = score(kept_f, default, test_pairs)
print(f"\n=== held-out TEST ===")
print(f"  full  ({len(rules_f):>7,} rules): top1={full_test:.3f}")
print(f"  pruned({len(kept_f):>7,} rules): top1={pruned_test:.3f}   "
      f"({len(kept_f)/len(rules_f)*100:.0f}% of rules, {pruned_test-full_test:+.3f} top1)")
