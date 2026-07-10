"""Materialize the next-token predictor as an EDITABLE rule file.

Writes rules.txt: the top-K context->token rules (pruned of the overfitting
tail), sorted by frequency. This file IS the program -- predict.py runs from it,
so you can delete / reorder / rewrite rules and re-score.

  usage: python3 extract_rules.py [corpus.txt] [MIN_COUNT]

MIN_COUNT defaults to 10 -- the validation-tuned prune policy (see prune.py):
keep a rule only if its context was seen >= MIN_COUNT times. (Purity filtering
was tried and HURT: the best rules are high-traffic hedges like ',' -> 'and'
that are frequent but not pure.)
"""
import re, sys
from collections import defaultdict, Counter

MAXN = 4
def load_tokens(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START");  t = t[t.find("\n", a)+1:] if a != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

corpus = sys.argv[1] if len(sys.argv) > 1 else "austen_corpus.txt"
MIN_COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 10
toks = load_tokens(corpus)
train = toks[:int(len(toks)*0.9)]

ctx = defaultdict(Counter)
for i in range(len(train)):
    for order in range(2, MAXN+1):
        if i-order+1 >= 0:
            ctx[tuple(train[i-order+1:i])][train[i]] += 1

# keep rules whose context was seen >= MIN_COUNT times; top-3 candidates each
rules = []
for k, c in ctx.items():
    tot = sum(c.values())
    if tot < MIN_COUNT: continue
    cand = [t for t, _ in c.most_common(3)]
    rules.append((tot, len(k)+1, k, cand))
rules.sort(key=lambda r: -r[0])
default = Counter(train).most_common(1)[0][0]

with open("rules.txt", "w", encoding="utf-8") as f:
    f.write(f"# next-token rule program  |  corpus={corpus}  |  {len(rules)} rules\n")
    f.write(f"# format:  COUNT | ORDER | context tokens  =>  cand1 :: cand2 :: cand3\n")
    f.write(f"# runner uses the HIGHEST-order matching rule; edit/delete/reorder freely.\n")
    f.write(f"@default => {default}\n")
    for cnt, order, k, cand in rules:
        f.write(f"{cnt:>7} | {order} | {' '.join(k)}  =>  {' :: '.join(cand)}\n")

print(f"wrote rules.txt: {len(rules)} rules from {len(train):,} training tokens "
      f"(corpus vocab {len(set(toks)):,}); default token = {default!r}")
