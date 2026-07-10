"""Materialize the graded lexicon: every content word appearing as a candidate
in the rulebook, scored on every registered scale. Output lexicon.tsv --
the editable tone-knowledge artifact of the program.
"""
import re, sys, numpy as np
from collections import defaultdict
from scales import registry, Scale, _embed
from generalize import TOK2POS

is_content = lambda w: re.fullmatch(r"[a-z]{3,}", w) and w not in TOK2POS

# vocab = content words among rulebook candidates
vocab = set()
for line in open("rules_big.txt", encoding="utf-8"):
    if line.startswith(("#", "@")) or "=>" not in line: continue
    for c in line.split("=>", 1)[1].split(" :: "):
        c = c.strip()
        if is_content(c): vocab.add(c)
vocab = sorted(vocab)
print(f"content vocab from rulebook candidates: {len(vocab)} words", file=sys.stderr)

names = sorted(registry())
scales = {n: Scale(n) for n in names}
print(f"scales: {', '.join(names)}", file=sys.stderr)

# embed vocab once (batched), then score = projections
V = []
for i in range(0, len(vocab), 64):
    V.append(_embed(vocab[i:i+64]))
    print(f"  embedded {min(i+64,len(vocab))}/{len(vocab)}", file=sys.stderr)
V = np.vstack(V)

cols = {n: ((V @ scales[n].axis) - scales[n].mu) / scales[n].sd for n in names}
with open("lexicon.tsv", "w", encoding="utf-8") as f:
    f.write("word\t" + "\t".join(names) + "\n")
    for i, w in enumerate(vocab):
        f.write(w + "\t" + "\t".join(f"{cols[n][i]:+.2f}" for n in names) + "\n")
print(f"wrote lexicon.tsv: {len(vocab)} words x {len(names)} scales", file=sys.stderr)

# spot check: extremes per scale
print("\n=== extremes per scale (top-3 / bottom-3) ===")
for n in names:
    order = np.argsort(-cols[n])
    top = [vocab[j] for j in order[:3]]; bot = [vocab[j] for j in order[-3:]]
    print(f"  {n:<13} high: {', '.join(top):<32} low: {', '.join(bot)}")
