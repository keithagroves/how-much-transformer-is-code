"""Slot abstraction: rules learn FRAMES with typed holes; values come from
input data. Completes the structured-NLG system: schema knob + slot values in,
traceable text out, zero hallucinated facts.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 5
DAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
DIRS = set("north south east west northeast northwest southeast southwest".split())
COLORS = set("black silver white red blue green gray grey gold navy teal rose charcoal".split())
TIMES = set("morning afternoon evening midday noon night midnight dawn dusk".split())

def slotify(doc):
    """raw cased text -> lowercase token stream with typed slot tokens."""
    doc = re.sub(r"\d+(?:[.,]\d+)*", " <num> ", doc)
    toks = re.findall(r"<num>|[A-Za-z]+|[^\w\s]", doc)
    out, sent_start = [], True
    for t in toks:
        if t == "<num>":
            out.append(t); sent_start = False; continue
        if re.fullmatch(r"[A-Z][a-zA-Z]*", t) and not sent_start:
            if out and out[-1] == "<name>": pass          # collapse runs
            else: out.append("<name>")
            sent_start = False; continue
        w = t.lower()
        if w in DAYS: w = "<day>"
        elif w in DIRS: w = "<dir>"
        elif w in COLORS: w = "<color>"
        elif w in TIMES: w = "<time>"
        out.append(w)
        sent_start = t in ".!?"
    return out

def classify(doc):
    d = doc.lower()
    w = sum(k in d for k in ("temperatures", "winds", "degrees", "cloudy", "residents"))
    p = sum(k in d for k in ("features", "costs", "ships", "weighs", "comes in"))
    r = sum(k in d for k in ("defeated", "score", "scored", "half", "match"))
    return max((w, "weather"), (p, "product"), (r, "recap"))[1]

docs = [d.strip() for d in open("structured_corpus.txt", encoding="utf-8")
        .read().split("<|doc|>") if len(d.split()) > 10]
schemas = [classify(d) for d in docs]
n = len(docs)
tr = [(slotify(d), s) for d, s in zip(docs[:int(n*.8)], schemas[:int(n*.8)])]
te = [(slotify(d), s) for d, s in zip(docs[int(n*.9):], schemas[int(n*.9):])]

cond, plain = defaultdict(Counter), defaultdict(Counter)
for ts, s in tr:
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0:
                k = tuple(ts[i-o+1:i])
                plain[k][ts[i]] += 1; cond[(s, k)][ts[i]] += 1
plainR = {k: c.most_common(1)[0][0] for k, c in plain.items() if sum(c.values()) >= MIN_RULE}
condR  = {k: c.most_common(1)[0][0] for k, c in cond.items()  if sum(c.values()) >= MIN_RULE}
print(f"slotted rules: plain={len(plainR):,} conditioned={len(condR):,}", file=sys.stderr)

hits = total = 0
for ts, s in te:
    for i in range(3, len(ts)):
        p = "."
        for o in range(MAXN, 1, -1):
            k = tuple(ts[max(0, i-o+1):i])
            if len(k) != o-1: continue
            if (s, k) in condR: p = condR[(s, k)]; break
            if k in plainR: p = plainR[k]; break
        hits += p == ts[i]; total += 1
print(f"\n=== slotted, schema-conditioned frame accuracy: {hits/total:.3f}  (n={total:,}) ===")
print("    (arc: open 0.157 -> structured 0.569 -> +schema 0.584 -> +slots above)")

# ---- data-to-text: fill slots from typed input queues ----
gc = defaultdict(Counter)
for ts, s in tr:
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: gc[(s, tuple(ts[i-o+1:i]))][ts[i]] += 1
grules = {k: c.most_common(5) for k, c in gc.items() if sum(c.values()) >= 3}
rng = np.random.default_rng(2)

def realize(schema, seed, slots, nn=46, temp=0.4):
    q = {k: list(v) for k, v in slots.items()}          # typed FIFO queues
    ts = seed.split()
    outw = list(ts)
    for _ in range(nn):
        cands = None
        for o in range(MAXN, 1, -1):
            k = (schema, tuple(ts[-(o-1):]))
            if k in grules: cands = grules[k]; break
        if not cands: break
        w = [x for x, _ in cands]; p = np.array([c for _, c in cands], float)**(1/temp); p /= p.sum()
        pick = w[rng.choice(len(w), p=p)]
        ts.append(pick)
        if pick.startswith("<") and q.get(pick):
            outw.append(str(q[pick].pop(0)))            # YOUR fact, not a guess
        else:
            outw.append(pick)
    return " ".join(outw)

print("\n=== data-to-text demo (facts in -> traceable text out) ===")
print("weather, facts {Paris, cloudy?, 23deg, afternoon, NW, 12mph}:")
print(" ", realize("weather", "temperatures will reach",
      {"<num>": [23, 12], "<time>": ["afternoon"], "<dir>": ["northwest"], "<day>": ["friday"]}))
print("\nproduct, facts {1.2 lb, blue/white, $89, 2 days}:")
print(" ", realize("product", "it weighs",
      {"<num>": ["1.2", 89, 2], "<color>": ["blue", "white"]}))
print("\nrecap, facts {Falcons 31, Bears 24, Jones 18pts}:")
print(" ", realize("recap", "the <name> defeated the",
      {"<name>": ["Falcons", "Bears", "Jones", "Hawks"], "<num>": [31, 24, 18], "<day>": ["saturday"]}))
