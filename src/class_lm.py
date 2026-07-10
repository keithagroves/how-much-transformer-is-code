"""Semantic-class backoff tier: qwen-embedding k-means classes as a middle
layer between lexical rules and default.

Backoff (specific -> general, the family that works):
  lexical ctx (order 4->2)  ->  class-signature ctx (order 4->2)  ->  default

k tuned on validation; one-shot test. Same 80/10/10 hygiene as cache_lm.
"""
import json, re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN = 10

def load(p):
    t = open(p, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    b = t.rfind("*** END");  t = t[:b] if b != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

austen = load("austen_corpus.txt")
c80, c90 = int(len(austen)*0.8), int(len(austen)*0.9)
train = austen[:c80]
for pid in (98, 1400, 766, 1260, 768, 145, 4276, 599, 2701):
    train += load(f"pg{pid}.txt")
val, test = austen[c80:c90], austen[c90:]

vocab = json.load(open("vocab_words.json"))
E = np.load("vocab_emb.npy")
print(f"vocab={len(vocab)}  emb={E.shape}", file=sys.stderr)

# lexical tier (as before)
lex_ctx = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: lex_ctx[tuple(train[i-o+1:i])][train[i]] += 1
lex = {k: c.most_common(1)[0][0] for k, c in lex_ctx.items() if sum(c.values()) >= MIN}
default = Counter(train).most_common(1)[0][0]
print(f"lexical rules: {len(lex):,}", file=sys.stderr)

def build_classes(k):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048).fit(E)
    w2c = {w: int(c) for w, c in zip(vocab, km.labels_)}
    def cls(tok):                      # punctuation/rare words: own singleton-ish ids
        if tok in w2c: return w2c[tok]
        if re.fullmatch(r"[^\w\s]", tok): return "P" + tok
        return "RARE"
    cls_train = [cls(t) for t in train]
    cg = defaultdict(Counter)
    for i in range(len(train)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: cg[tuple(cls_train[i-o+1:i])][train[i]] += 1
    cgram = {kk: c.most_common(1)[0][0] for kk, c in cg.items() if sum(c.values()) >= MIN}
    return cls, cgram

def top1(seg, cls, cgram):
    cseg = [cls(t) for t in seg] if cls else None
    hits = n = 0
    for i in range(3, len(seg)):
        pred = None
        for o in range(MAXN, 1, -1):
            kk = tuple(seg[max(0, i-o+1):i])
            if len(kk) == o-1 and kk in lex: pred = lex[kk]; break
        if pred is None and cgram is not None:
            for o in range(MAXN, 1, -1):
                kk = tuple(cseg[max(0, i-o+1):i])
                if len(kk) == o-1 and kk in cgram: pred = cgram[kk]; break
        hits += (pred or default) == seg[i]; n += 1
    return hits/n

base = top1(val, None, None)
print(f"\nval baseline (lexical only): {base:.4f}")
best = (base, None, None, "baseline")
for k in (128, 512, 2048):
    cls, cgram = build_classes(k)
    a = top1(val, cls, cgram)
    print(f"val k={k:<5} class-grams={len(cgram):>7,}  top1={a:.4f}  ({a-base:+.4f})")
    if a > best[0]: best = (a, cls, cgram, f"k={k}")

print(f"\nchosen: {best[3]}")
tb = top1(test, None, None)
tt = top1(test, best[1], best[2]) if best[1] else tb
print(f"TEST: lexical-only={tb:.4f}   +class-tier={tt:.4f}   ({tt-tb:+.4f})")
