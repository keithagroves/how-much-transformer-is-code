"""Speculative-decoding acceptance test: can the rulebook draft for ministral?

Rulebook (built from the existing corpus) drafts tokens; ministral generates a
FRESH stream (unseen prompts + sampling). Acceptance = rulebook top-1 matches
the token ministral actually emitted. Speedup potential depends on runs of
consecutive accepts: with draft length g, each target-model call yields
~E[min(run, g)] + 1 tokens.

Caveat: word-level tokens (regex), a proxy for BPE-level acceptance.
"""
import re, sys, numpy as np, requests
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 3

# ---- rulebook from current corpus snapshot ----
text = open("ministral_corpus.txt", encoding="utf-8").read().replace("<|doc|>", " ")
toks = re.findall(r"[a-z]+|[^\w\s]", text.lower())
cc = defaultdict(Counter)
for i in range(len(toks)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: cc[tuple(toks[i-o+1:i])][toks[i]] += 1
rules = {k: c.most_common(1)[0][0] for k, c in cc.items() if sum(c.values()) >= MIN_RULE}
default = Counter(toks).most_common(1)[0][0]
print(f"rulebook: {len(rules):,} rules from {len(toks):,} tokens", file=sys.stderr)

def draft(ctx):
    for o in range(MAXN, 1, -1):
        k = tuple(ctx[-(o-1):])
        if len(k) == o-1 and k in rules: return rules[k]
    return default

# ---- fresh ministral stream (prompts not in the factory mix) ----
PROMPTS = [
 "Write a short story about a violin teacher who loses her hearing. Keep it natural and concrete.",
 "Describe in plain prose a night market closing down in the rain. Keep it natural and concrete.",
 "Write a scene with dialogue about two brothers fixing a roof before winter. Keep it natural and concrete.",
 "Tell a simple story about a ferry crossing delayed by fog. Keep it natural and concrete.",
 "Write a first-person account of getting locked inside a museum overnight. Keep it natural and concrete.",
 "Narrate an afternoon involving a beekeeper teaching an apprentice. Keep it natural and concrete.",
]
def gen(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0.8, "num_predict": 600}, "prompt": prompt}, timeout=600)
    return r.json().get("response", "")

FUNC = set(("the a an this that these those his her its their my your our some any no every each "
    "all such another of in on at to from by with for about into over under after before between "
    "through upon without within against toward towards among during than as i he she it we they "
    "you me him them us who which what whom whose and but or nor yet so because though although "
    "while if when whereas since unless was were is are am be been being had have has do does did "
    "would could should will shall may might must can not").split())
kind = lambda w: "punct" if not re.fullmatch(r"[a-z]+", w) else ("func" if w in FUNC else "content")

acc = Counter(); tot = Counter(); runs = []
for p in PROMPTS:
    s = re.sub(r"[*#_>`]+", " ", gen(p))
    st = re.findall(r"[a-z]+|[^\w\s]", s.lower())
    print(f"  stream: {len(st)} tokens | {s[:60]!r}", file=sys.stderr)
    run = 0
    for i in range(3, len(st)):
        d = draft(st[:i])
        k = kind(st[i])
        hit = d == st[i]
        acc[k] += hit; tot[k] += 1
        if hit: run += 1
        elif run: runs.append(run); run = 0
    if run: runs.append(run)

n = sum(tot.values()); hits = sum(acc.values())
runs = np.array(runs) if runs else np.array([0])
print(f"\n=== acceptance on fresh ministral stream (n={n:,} tokens) ===")
print(f"  overall acceptance: {hits/n:.3f}")
for k in ("punct", "func", "content"):
    if tot[k]: print(f"    {k:<8} {acc[k]/tot[k]:.3f}  (n={tot[k]:,})")
print(f"  accept-run lengths: mean={runs.mean():.2f}  p90={np.percentile(runs,90):.0f}  max={runs.max()}")
print(f"\n=== implied tokens per target-model call (draft length g) ===")
alpha = hits/n
for g in (2, 4, 8):
    exp = (1 - alpha**(g+1)) / (1 - alpha)     # standard iid approximation
    print(f"  g={g}:  {exp:.2f}x  (vs 1.0 without drafting)")
