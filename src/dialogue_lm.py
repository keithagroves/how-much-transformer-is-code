"""Dialogue-state conditioning: one linguistic bit -- inside vs outside
quotation -- added to the rule CONTEXT (the family that works), with fallback.

Lookup per order: (state, ctx) first, then plain ctx, then lower order.
Same 80/10/10 hygiene as cache_lm; variants chosen on val, one-shot test.
"""
import re, sys
from collections import defaultdict, Counter

MAXN = 4; MIN = 10

def load(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    b = t.rfind("*** END");  t = t[:b] if b != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

def quote_states(seg):
    """state[i] = True if token i is spoken dialogue (inside quotes)."""
    st, out = False, []
    for tok in seg:
        if tok == "“": st = True
        out.append(st)                     # the quote mark itself: opening counts in
        if tok == "”": st = False
        if tok == '"': st = not st
    return out

austen = load("austen_corpus.txt")
c80, c90 = int(len(austen)*0.8), int(len(austen)*0.9)
train = austen[:c80]
for pid in (98, 1400, 766, 1260, 768, 145, 4276, 599, 2701):
    train += load(f"pg{pid}.txt")
val, test = austen[c80:c90], austen[c90:]
tstate = quote_states(train)
print(f"train={len(train):,} (dialogue share {sum(tstate)/len(tstate):.0%})  "
      f"val={len(val):,} test={len(test):,}", file=sys.stderr)

plain, cond = defaultdict(Counter), defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 < 0: continue
        k = tuple(train[i-o+1:i])
        plain[k][train[i]] += 1
        cond[(tstate[i], k)][train[i]] += 1
plain = {k: c.most_common(1)[0][0] for k, c in plain.items() if sum(c.values()) >= MIN}
cond  = {k: c.most_common(1)[0][0] for k, c in cond.items()  if sum(c.values()) >= MIN}
default = Counter(train).most_common(1)[0][0]
print(f"plain rules={len(plain):,}  conditioned rules={len(cond):,}", file=sys.stderr)

def top1(seg, use_cond):
    states = quote_states(seg)
    hits = n = 0
    for i in range(3, len(seg)):
        pred = default
        for o in range(MAXN, 1, -1):
            k = tuple(seg[max(0, i-o+1):i])
            if len(k) != o-1: continue
            if use_cond and (states[i-1], k) in cond:
                pred = cond[(states[i-1], k)]; break
            if k in plain:
                pred = plain[k]; break
        hits += pred == seg[i]; n += 1
    return hits/n

print("\n=== validation ===")
a_p, a_c = top1(val, False), top1(val, True)
print(f"  plain       : {a_p:.4f}")
print(f"  +quote-state: {a_c:.4f}  ({a_c-a_p:+.4f})")

print("\n=== held-out TEST (one shot) ===")
b_p, b_c = top1(test, False), top1(test, True)
print(f"  plain       : {b_p:.4f}")
print(f"  +quote-state: {b_c:.4f}  ({b_c-b_p:+.4f})")

# where do they differ? show a few state-driven wins
states = quote_states(test)
shown = 0
print("\n=== examples where the state bit changed the prediction (test) ===")
for i in range(3, len(test)):
    if shown >= 6: break
    for o in range(MAXN, 1, -1):
        k = tuple(test[max(0, i-o+1):i])
        if len(k) != o-1: continue
        ck, has_c = (states[i-1], k), (states[i-1], k) in cond
        if has_c and k in plain and cond[ck] != plain[k]:
            tag = "DLG" if states[i-1] else "NAR"
            mark = "WIN " if cond[ck] == test[i] else ("LOSS" if plain[k] == test[i] else "    ")
            print(f"  [{tag}] {mark} after {' '.join(k)!r}: cond={cond[ck]!r} plain={plain[k]!r} truth={test[i]!r}")
            shown += 1
        break
