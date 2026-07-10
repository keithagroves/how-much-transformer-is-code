"""Near-duplicate document filter. Splits the corpus on <|doc|>, computes
hashed 8-gram shingle signatures, drops any doc whose Jaccard overlap with an
earlier KEPT doc exceeds the threshold. Writes ministral_dedup.txt.
Run before any rebuild; rebuilds should also split train/test at DOC level.
"""
import re, sys

THRESH = 0.30
docs = [d.strip() for d in open("ministral_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 50]

def sig(doc):
    toks = re.findall(r"[a-z]+", doc.lower())
    return set(hash(tuple(toks[i:i+8])) for i in range(0, max(1, len(toks)-8), 3))

kept, sigs, dropped = [], [], 0
for d in docs:
    s = sig(d)
    dup = any(len(s & t)/max(1, len(s | t)) > THRESH for t in sigs)
    if dup: dropped += 1
    else: kept.append(d); sigs.append(s)

open("ministral_dedup.txt", "w", encoding="utf-8").write("\n<|doc|>\n".join(kept))
w = sum(len(d.split()) for d in kept)
print(f"docs: {len(docs)} -> kept {len(kept)} (dropped {dropped} near-dups)  ~{w:,} words")
