"""Does register-matching improve next-token choice? Honest head-to-head.

On fixed-test positions where the fired rule offers >=2 content-word candidates
that are in lexicon.tsv:
  baseline : pick candidate[0] (frequency order)
  tone     : pick candidate whose lexicon score-vector is nearest the context
             register (context = previous 12 tokens, embedded, scored per scale)
"""
import re, sys, numpy as np
from collections import defaultdict
from scales import registry, Scale, _embed
from semantic_eval import load_tokens, load_rules, MAXN

names = sorted(registry())
scales = {n: Scale(n) for n in names}

LEX = {}
with open("lexicon.tsv", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")[1:]
    for line in f:
        p = line.rstrip("\n").split("\t")
        LEX[p[0]] = np.array([float(x) for x in p[1:]])
order_idx = [header.index(n) for n in names]

by, dflt = load_rules("rules_big.txt")
toks = load_tokens()
test = toks[int(len(toks)*0.9):]

# find eligible positions: fired rule has >=2 lexicon-covered content candidates
elig = []
for i in range(12, len(test)):
    ctx = test[i-3:i]
    cands = None
    for o in range(MAXN, 1, -1):
        k = tuple(ctx[-(o-1):])
        if k in by[o]: cands = by[o][k]; break
    if not cands: continue
    inlex = [c for c in cands if c in LEX]
    if len(inlex) >= 2:
        elig.append((i, inlex, test[i]))
print(f"eligible positions (rule offers >=2 content candidates): {len(elig)}", file=sys.stderr)

rng = np.random.default_rng(3)
sample = [elig[j] for j in rng.choice(len(elig), min(400, len(elig)), replace=False)]

# batch-embed the 12-token context windows
windows = [" ".join(test[i-12:i]) for i, _, _ in sample]
W = []
for j in range(0, len(windows), 32):
    W.append(_embed(windows[j:j+32]))
    print(f"  embedded {min(j+32,len(windows))}/{len(windows)} contexts", file=sys.stderr)
W = np.vstack(W)
REG = np.stack([((W @ scales[n].axis) - scales[n].mu) / scales[n].sd for n in names], axis=1)

base = tone = 0
flips_right, flips_wrong = [], []
for (i, cands, truth), reg in zip(sample, REG):
    b = cands[0]
    t = min(cands, key=lambda c: float(np.linalg.norm(LEX[c] - reg)))
    base += b == truth
    tone += t == truth
    if t != b:
        if t == truth: flips_right.append((cands, truth, " ".join(test[i-6:i])))
        elif b == truth: flips_wrong.append((cands, truth, " ".join(test[i-6:i])))
n = len(sample)
print(f"\nn={n} positions with a real choice")
print(f"  frequency-order top-1 : {base/n:.3f}")
print(f"  tone-match top-1      : {tone/n:.3f}")
print(f"  flips: {len(flips_right)} fixed, {len(flips_wrong)} broken by tone-matching")
for tag, lst in (("FIXED", flips_right[:5]), ("BROKEN", flips_wrong[:5])):
    print(f"\n  {tag} examples:")
    for cands, truth, w in lst:
        print(f"    cands={cands} truth={truth!r} | ...{w}")
