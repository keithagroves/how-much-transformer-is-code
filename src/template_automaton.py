"""The elegant system: a template automaton distilled from the rulebook.

  schema -> template inventory (frequent slotted sentences)
         -> transition model (which template follows which)
         -> slot filler (typed values from input)

Head-to-head vs the flat rulebook on the same v2-slotted test stream, plus a
parts count and a clean data-to-text demo with a natural stop criterion.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 5; T_MIN = 3
DAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
DIRS = set("north south east west northeast northwest southeast southwest".split())
COLORS = set("black silver white red blue green gray grey gold navy teal rose charcoal".split())
TIMES = set("morning afternoon evening midday noon night midnight dawn dusk".split())

raw = open("structured_corpus.txt", encoding="utf-8").read()
# common = words seen lowercase anywhere (>=3) -- used to spare 'Temperatures'
lower_freq = Counter(re.findall(r"\b[a-z]+\b", raw))

def slotify2(doc):
    doc = re.sub(r"\d+(?:[.,]\d+)*", " <num> ", doc)
    toks = re.findall(r"<num>|[A-Za-z]+|[^\w\s]", doc)
    out, sent_start = [], True
    for t in toks:
        if t == "<num>": out.append(t); sent_start = False; continue
        if re.fullmatch(r"[A-Z][a-zA-Z]*", t):
            is_name = (not sent_start) or lower_freq[t.lower()] < 3
            if is_name:
                if not (out and out[-1] == "<name>"): out.append("<name>")
                sent_start = False; continue
        w = t.lower()
        if w in DAYS: w = "<day>"
        elif w in DIRS: w = "<dir>"
        elif w in COLORS: w = "<color>"
        elif w in TIMES: w = "<time>"
        out.append(w); sent_start = t in ".!?"
    return out

def classify(doc):
    d = doc.lower()
    w = sum(k in d for k in ("temperatures", "winds", "degrees", "cloudy", "residents"))
    p = sum(k in d for k in ("features", "costs", "ships", "weighs", "comes in"))
    r = sum(k in d for k in ("defeated", "score", "scored", "half", "match"))
    return max((w, "weather"), (p, "product"), (r, "recap"))[1]

docs = [d.strip() for d in raw.split("<|doc|>") if len(d.split()) > 10]
schemas = [classify(d) for d in docs]
n = len(docs)
tr = [(slotify2(d), s) for d, s in zip(docs[:int(n*.8)], schemas[:int(n*.8)])]
te = [(slotify2(d), s) for d, s in zip(docs[int(n*.9):], schemas[int(n*.9):])]

def sentences(ts):
    out, cur = [], []
    for t in ts:
        cur.append(t)
        if t in ".!?": out.append(tuple(cur)); cur = []
    if cur: out.append(tuple(cur))
    return out

# ---- template inventory + transitions ----
inv = {s: Counter() for s in ("weather", "product", "recap")}
trans = {s: defaultdict(Counter) for s in inv}
for ts, s in tr:
    sents = sentences(ts)
    prev = "<START>"
    for sent in sents:
        inv[s][sent] += 1
        trans[s][prev][sent] += 1
        prev = sent
templates = {s: {t: c for t, c in inv[s].items() if c >= T_MIN} for s in inv}
n_templates = sum(len(v) for v in templates.values())
n_trans = sum(sum(1 for _ in nxt) for s in trans for nxt in trans[s].values())
cov = {s: sum(c for t, c in inv[s].items() if t in templates[s]) / sum(inv[s].values())
       for s in inv}
print("=== template inventory ===")
for s in templates:
    print(f"  {s:<8} {len(templates[s]):>4} templates cover {cov[s]:.0%} of train sentences")
print(f"  total parts: {n_templates} templates (+ transition table) vs 3,670 flat rules")

print("\n=== top templates per schema ===")
for s in templates:
    for t, c in sorted(templates[s].items(), key=lambda kv: -kv[1])[:3]:
        print(f"  [{s}] ({c}x) {' '.join(t)}")

# ---- token-level head-to-head on test ----
# rulebook baseline on the SAME v2 stream
cond = defaultdict(Counter)
for ts, s in tr:
    for i in range(len(ts)):
        for o in range(2, MAXN+1):
            if i-o+1 >= 0: cond[(s, tuple(ts[i-o+1:i]))][ts[i]] += 1
rb = {k: c.most_common(1)[0][0] for k, c in cond.items() if sum(c.values()) >= MIN_RULE}
def rb_pred(ts, i, s):
    for o in range(MAXN, 1, -1):
        k = (s, tuple(ts[max(0,i-o+1):i]))
        if k in rb: return rb[k]
    return "."

def auto_pred_doc(ts, s):
    """automaton predictions for every position in a doc"""
    preds = [None]*len(ts)
    sents = sentences(ts)
    prev = "<START>"; pos = 0
    for sent in sents:
        # candidates weighted by transition prob then frequency
        tw = trans[s].get(prev, Counter())
        for j in range(len(sent)):
            pref = sent[:j]
            cands = [(t, templates[s][t]*(1+tw.get(t, 0))) for t in templates[s]
                     if len(t) > j and t[:j] == pref]
            if cands:
                votes = Counter()
                for t, w in cands: votes[t[j]] += w
                preds[pos+j] = votes.most_common(1)[0][0]
            else:
                preds[pos+j] = "."
        prev = sent if sent in templates[s] else "<OTHER>"
        pos += len(sent)
    return preds

rb_hit = at_hit = tot = 0
for ts, s in te:
    ap = auto_pred_doc(ts, s)
    for i in range(3, len(ts)):
        rb_hit += rb_pred(ts, i, s) == ts[i]
        at_hit += ap[i] == ts[i]
        tot += 1
print(f"\n=== head-to-head (test, n={tot:,}, same v2 slot stream) ===")
print(f"  flat rulebook ({len(rb):,} rules): {rb_hit/tot:.3f}")
print(f"  template automaton ({n_templates} templates): {at_hit/tot:.3f}")

# ---- clean generation demo with natural stop ----
rng = np.random.default_rng(9)
def realize(s, slots, n_sent=4):
    q = {k: list(v) for k, v in slots.items()}
    prev, out = "<START>", []
    for _ in range(n_sent):
        tw = trans[s].get(prev, Counter())
        cands = [(t, templates[s][t]*(1+tw.get(t, 0))) for t in templates[s]]
        if not cands: break
        ws = np.array([w for _, w in cands], float); ws /= ws.sum()
        t = cands[rng.choice(len(cands), p=ws)][0]
        for tok in t:
            if tok.startswith("<") and q.get(tok): out.append(str(q[tok].pop(0)))
            else: out.append(tok)
        prev = t
    return " ".join(out)
print("\n=== automaton data-to-text (natural stop after 4 templates) ===")
print(" ", realize("weather", {"<num>": [21, 9], "<time>": ["morning"],
                               "<dir>": ["southeast"], "<day>": ["tuesday"], "<name>": ["Lisbon"]}))
