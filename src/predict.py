"""Run and score the rule program in rules.txt.

The program is the text file: predict() consults the highest-order matching rule.
Edit rules.txt and re-run  `python3 predict.py`  to see accuracy change, or
`python3 predict.py "some text prefix"` to see it predict live.

  usage: python3 predict.py               # score rules.txt on held-out test
         python3 predict.py "i was very"  # show top-3 prediction + which rule fired
"""
import re, sys
from collections import defaultdict

MAXN = 4
def load_tokens(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START");  t = t[t.find("\n", a)+1:] if a != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

def load_rules(path="rules.txt"):
    by_order = defaultdict(dict)      # order -> {context_tuple: [candidates]}
    default = "the"
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("@default"):
            default = line.split("=>")[1].strip(); continue
        if not line or line.startswith("#"): continue
        head, cands = line.split("=>", 1)
        _cnt, order, ctx = head.split("|", 2)
        by_order[int(order)][tuple(ctx.split())] = [c.strip() for c in cands.split(" :: ")]
    return by_order, default

def predict(by_order, default, context, topn=3):
    for order in range(MAXN, 1, -1):                  # highest-order rule wins
        key = tuple(context[-(order-1):])
        if key in by_order[order]:
            cand = by_order[order][key]
            return cand[:topn], order
    return [default], 1

if __name__ == "__main__":
    by_order, default = load_rules()
    n_rules = sum(len(v) for v in by_order.values())

    if len(sys.argv) > 1:                              # live prediction on a prefix
        ctx = re.findall(r"[a-z]+|[^\w\s]", sys.argv[1].lower())
        top, order = predict(by_order, default, ctx)
        print(f"context: {' '.join(ctx)!r}")
        print(f"  top-3: {top}   (fired: order-{order} rule)")
        sys.exit()

    toks = load_tokens("austen_corpus.txt")            # score on held-out tail
    test = toks[int(len(toks)*0.9):]
    pairs = [(test[max(0,i-3):i], test[i]) for i in range(3, len(test))]
    c1 = c3 = 0; fired = defaultdict(int)
    for ctx, truth in pairs:
        top, order = predict(by_order, default, ctx); fired[order] += 1
        c1 += top[0] == truth
        c3 += truth in top
    n = len(pairs)
    print(f"rules.txt: {n_rules:,} rules   test={n:,}")
    print(f"  top1={c1/n:.3f}  top3={c3/n:.3f}   "
          f"(order fired: " + " ".join(f"{o}:{fired[o]/n:.0%}" for o in (4,3,2,1)) + ")")
