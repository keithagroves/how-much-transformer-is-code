"""Interpolated cache LM on a counts-bearing rulebook (format v2).

p(w) = (1-lambda) * p_ngram(w)  +  lambda * p_cache(w)

- p_ngram: candidate counts from the fired (highest-order) rule.
- p_cache: distribution over content words in the recent window (burstiness);
  the cache may propose words the rule didn't list.
- Hygiene: build on austen[0:80%] + 9 novels; tune (lambda, window) on
  austen[80:90%] validation; report once on austen[90:100%] test.

Writes rules2.txt: `TOTAL | ORDER | context => w1:c1 :: w2:c2 :: ...` (top-5).
"""
import re, sys
from collections import defaultdict, Counter

MAXN = 4; MIN = 10
FUNC = set(("the a an this that these those his her its their my your our some any no every "
    "each all such another of in on at to from by with for about into over under after "
    "before between through upon without within against toward towards among during than as "
    "i he she it we they you me him them us who which what whom whose and but or nor yet so "
    "because though although while if when whereas since unless was were is are am be been "
    "being had have has do does did would could should will shall may might must can not").split())
is_content = lambda w: re.fullmatch(r"[a-z]{3,}", w) and w not in FUNC

def load(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    b = t.rfind("*** END");  t = t[:b] if b != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

austen = load("austen_corpus.txt")
c80, c90 = int(len(austen)*0.8), int(len(austen)*0.9)
train = austen[:c80]
for pid in (98, 1400, 766, 1260, 768, 145, 4276, 599, 2701):
    train += load(f"pg{pid}.txt")
val, test = austen[c80:c90], austen[c90:]
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}", file=sys.stderr)

ctx = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: ctx[tuple(train[i-o+1:i])][train[i]] += 1
rules = {}                                     # context -> (total, [(w,c) top5])
for k, c in ctx.items():
    tot = sum(c.values())
    if tot >= MIN: rules[k] = (tot, c.most_common(5))
default = Counter(train).most_common(1)[0][0]
print(f"rules kept: {len(rules):,}", file=sys.stderr)

with open("rules2.txt", "w", encoding="utf-8") as f:
    f.write("# rulebook v2: per-candidate counts  |  TOTAL | ORDER | ctx => w:c :: ...\n")
    f.write(f"@default => {default}\n")
    for k, (tot, cands) in sorted(rules.items(), key=lambda kv: -kv[1][0]):
        f.write(f"{tot:>7} | {len(k)+1} | {' '.join(k)}  =>  "
                + " :: ".join(f"{w}:{c}" for w, c in cands) + "\n")

def predict(seg, i, lam, window):
    dist = {}
    for o in range(MAXN, 1, -1):               # p_ngram from highest fired rule
        k = tuple(seg[max(0, i-o+1):i])
        if len(k) == o-1 and k in rules:
            tot, cands = rules[k]
            dist = {w: (1-lam)*c/tot for w, c in cands}
            break
    if lam > 0:                                # p_cache from recent content words
        recent = [t for t in seg[max(0, i-window):i] if is_content(t)]
        if recent:
            rc = Counter(recent); rtot = len(recent)
            for w, c in rc.items():
                dist[w] = dist.get(w, 0.0) + lam*c/rtot
    return max(dist, key=dist.get) if dist else default

def top1(seg, lam, window):
    hits = sum(predict(seg, i, lam, window) == seg[i] for i in range(3, len(seg)))
    return hits / (len(seg)-3)

print("\n=== tune on validation ===")
best = (-1, 0.0, 50)
for lam in (0.0, 0.05, 0.1, 0.2, 0.3):
    for w in ((50, 200) if lam > 0 else (50,)):
        a = top1(val, lam, w)
        print(f"  lambda={lam:<5} window={w:<4} val_top1={a:.4f}")
        if a > best[0]: best = (a, lam, w)
_, lam, w = best
print(f"  -> chosen: lambda={lam}, window={w}")

print("\n=== held-out TEST (one shot) ===")
base = top1(test, 0.0, 50)
inter = top1(test, lam, w)
print(f"  pure n-gram      : {base:.4f}")
print(f"  + interpolated cache: {inter:.4f}   ({inter-base:+.4f})")
