"""Mine the structured slotted rulebook for its elegant core.

Hypothesis: in a schematic domain the rulebook is not thousands of independent
facts but a small automaton = deterministic CHAINS (templates) + a few BRANCH
points + slot holes. We measure exactly that:
  1. determinism histogram of rules
  2. extracted template chains (follow argmax while p >= 0.75)
  3. behavioral equivalence classes (DFA-style minimization)
  4. share of real test text covered by deterministic links
"""
import re, sys
from collections import defaultdict, Counter
from slot_lm import slotify, classify

MAXN = 4; MIN_RULE = 5; P_DET = 0.75

docs = [d.strip() for d in open("structured_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 10]
schemas = [classify(d) for d in docs]
n = len(docs)
tr = [(slotify(d), s) for d, s in zip(docs[:int(n*.8)], schemas[:int(n*.8)])]
te = [(slotify(d), s) for d, s in zip(docs[int(n*.9):], schemas[int(n*.9):])]

cond = defaultdict(Counter)
for ts, s in tr:
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: cond[(s, tuple(ts[i-o+1:i]))][ts[i]] += 1
rules = {k: c for k, c in cond.items() if sum(c.values()) >= MIN_RULE}
print(f"rules (count>={MIN_RULE}): {len(rules):,}", file=sys.stderr)

# ---- 1. determinism histogram ----
probs = []
for k, c in rules.items():
    tot = sum(c.values())
    probs.append(c.most_common(1)[0][1] / tot)
bins = Counter("deterministic(>=0.9)" if p >= 0.9 else
               "strong(0.75-0.9)" if p >= 0.75 else
               "branch(0.4-0.75)" if p >= 0.4 else "open(<0.4)" for p in probs)
print("\n=== rule determinism ===")
for b in ("deterministic(>=0.9)", "strong(0.75-0.9)", "branch(0.4-0.75)", "open(<0.4)"):
    print(f"  {b:<22} {bins[b]:>6,}  ({bins[b]/len(probs):.0%})")

# ---- 2. template chains: follow argmax from sentence starts ----
def argmax(s, ctx):
    for o in range(MAXN, 1, -1):
        k = (s, tuple(ctx[-(o-1):]))
        if k in rules:
            c = rules[k]; tot = sum(c.values())
            w, cnt = c.most_common(1)[0]
            return w, cnt / tot
    return None, 0.0

print("\n=== induced templates (chains of p>=%.2f links, per schema) ===" % P_DET)
all_templates = {}
for s in ("weather", "product", "recap"):
    starts = Counter()
    for ts, s2 in tr:
        if s2 != s: continue
        for i in range(1, len(ts)):
            if ts[i-1] == "." and i < len(ts)-1: starts[ts[i]] += 1
        starts[ts[0]] += 1
    seen = set()
    for w0, _ in starts.most_common(6):
        ctx = [".", w0]
        chain = [w0]
        for _ in range(30):
            w, p = argmax(s, ctx)
            if w is None or p < P_DET: break
            chain.append(w); ctx.append(w)
            if w == ".": break
        t = " ".join(chain)
        if len(chain) > 3 and t not in seen:
            seen.add(t)
            print(f"  [{s:<7}] {t}")
    all_templates[s] = seen

# ---- 3. behavioral equivalence classes (minimization) ----
sig = defaultdict(list)
for k, c in rules.items():
    tot = sum(c.values())
    sg = tuple(sorted((w, round(cnt/tot, 1)) for w, cnt in c.most_common(3)))
    sig[sg].append(k)
print(f"\n=== minimization: {len(rules):,} rules -> {len(sig):,} behavioral classes "
      f"({len(rules)/len(sig):.1f}x compression) ===")
big = sorted(sig.items(), key=lambda kv: -len(kv[1]))[:5]
for sg, ks in big:
    print(f"  class of {len(ks):>3} contexts, behavior {dict((w,p) for w,p in sg)}"
          f"  e.g. {[' '.join(k[1]) for k in ks[:3]]}")

# ---- 4. how much of real text is deterministic template-following? ----
det_n = det_hit = br_n = br_hit = 0
for ts, s in te:
    for i in range(3, len(ts)):
        w, p = argmax(s, ts[:i])
        if w is None: continue
        if p >= P_DET:
            det_n += 1; det_hit += w == ts[i]
        else:
            br_n += 1; br_hit += w == ts[i]
tot = det_n + br_n
print(f"\n=== test text anatomy ===")
print(f"  deterministic links: {det_n/tot:.0%} of positions, accuracy {det_hit/det_n:.3f}")
print(f"  branch points      : {br_n/tot:.0%} of positions, accuracy {br_hit/br_n:.3f}")
