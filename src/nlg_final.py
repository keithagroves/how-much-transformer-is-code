"""Cleanup-and-close pass: the finished structured-NLG system.

Fixes: (1) name detection -- a capitalized sentence-starter is <name> only if
the word also appears capitalized MID-sentence somewhere (Paris does; Winds
never does); (2) strip ministral's list numbering before tokenizing; (3) typed
slots -- <num> retyped by unit context (<temp>/<speed>/<price>/<points>/
<score>/<days>), <name> retyped per schema (<city>/<team>/<player>).
Then: rebuild rulebook, re-extract templates, hybrid eval, typed data-to-text.
"""
import re, sys, numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 5; T_MIN = 3
DAYS = set("monday tuesday wednesday thursday friday saturday sunday".split())
DIRS = set("north south east west northeast northwest southeast southwest".split())
COLORS = set("black silver white red blue green gray grey gold navy teal rose charcoal".split())
TIMES = set("morning afternoon evening midday noon night midnight dawn dusk".split())

raw = open("structured_corpus.txt", encoding="utf-8").read()
raw = re.sub(r"(?m)^\s*\d+\s*[\.\)]\s*", "", raw)              # strip list numbering
# name-words: capitalized in a NON-sentence-initial position somewhere
mid_caps = set(w.lower() for w in re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]+)\b", raw, re.M))

def slotify3(doc, schema):
    doc = re.sub(r"(?m)^\s*\d+\s*[\.\)]\s*", "", doc)
    doc = re.sub(r"\d+(?:[.,]\d+)*", " <num> ", doc)
    toks = re.findall(r"<num>|[A-Za-z]+|[^\w\s]", doc)
    CLOSED = {**{w: "<day>" for w in DAYS}, **{w: "<dir>" for w in DIRS},
              **{w: "<color>" for w in COLORS}, **{w: "<time>" for w in TIMES}}
    out, sent_start = [], True
    for t in toks:
        if t == "<num>":
            out.append(t); sent_start = False; continue
        w = t.lower()
        if w in CLOSED:                      # closed-class beats capitalization ('on Tuesday')
            out.append(CLOSED[w]); sent_start = False; continue
        if re.fullmatch(r"[A-Z][a-zA-Z]*", t) and (not sent_start or w in mid_caps):
            if not (out and out[-1] == "<name>"): out.append("<name>")
            sent_start = False; continue
        out.append(w); sent_start = t in ".!?"
    return retype(out, schema)

def retype(ts, schema):
    out = list(ts)
    for i, t in enumerate(out):
        nxt = out[i+1] if i+1 < len(out) else ""; nx2 = out[i+2] if i+2 < len(out) else ""
        prv = out[i-1] if i > 0 else ""
        if t == "<num>":
            if nxt == "degrees": out[i] = "<temp>"
            elif nxt == "miles": out[i] = "<speed>"
            elif prv == "$": out[i] = "<price>"
            elif nxt == "points": out[i] = "<points>"
            elif nxt == "to" and nx2 == "<num>": out[i] = "<score>"
            elif prv == "to" and out[i-2:i-1] == ["<score>"]: out[i] = "<score>"
            elif nxt in ("business", "days"): out[i] = "<days>"
        elif t == "<name>":
            if schema == "weather" and (i == 0 or prv in ".!?"): out[i] = "<city>"
            elif schema == "recap":
                if nxt in ("defeated",) or prv == "against" or (prv == "the" and nxt == "by"):
                    out[i] = "<team>"
                elif nxt == "scored": out[i] = "<player>"
    # second pass: 'the <team> defeated the <name>' -> object is a team too
    for i, t in enumerate(out):
        if t == "<name>" and i >= 2 and out[i-1] == "the" and out[i-2] == "defeated":
            out[i] = "<team>"
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
tr = [(slotify3(d, s), s) for d, s in zip(docs[:int(n*.8)], schemas[:int(n*.8)])]
te = [(slotify3(d, s), s) for d, s in zip(docs[int(n*.9):], schemas[int(n*.9):])]

# ---- rulebook ----
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

# ---- templates + transitions ----
def sentences(ts):
    out, cur = [], []
    for t in ts:
        cur.append(t)
        if t in ".!?": out.append(tuple(cur)); cur = []
    if cur: out.append(tuple(cur))
    return out
inv = {s: Counter() for s in ("weather","product","recap")}
trans = {s: defaultdict(Counter) for s in inv}
for ts, s in tr:
    prev = "<START>"
    for sent in sentences(ts):
        inv[s][sent] += 1; trans[s][prev][sent] += 1; prev = sent
templates = {s: {t: c for t, c in inv[s].items() if c >= T_MIN} for s in inv}
ntpl = sum(len(v) for v in templates.values())
cov = {s: sum(c for t,c in inv[s].items() if t in templates[s])/sum(inv[s].values()) for s in inv}
print("=== clean template inventory ===")
for s in templates:
    print(f"  {s:<8} {len(templates[s]):>3} templates, {cov[s]:.0%} sentence coverage")
    for t, c in sorted(templates[s].items(), key=lambda kv:-kv[1])[:2]:
        print(f"     ({c}x) {' '.join(t)}")

# ---- eval: flat vs hybrid ----
def auto_preds(ts, s):
    preds=[None]*len(ts); has=[False]*len(ts)
    prev="<START>"; pos=0
    for sent in sentences(ts):
        tw = trans[s].get(prev, Counter())
        for j in range(len(sent)):
            pref=sent[:j]
            cands=[(t, templates[s][t]*(1+tw.get(t,0))) for t in templates[s]
                   if len(t)>j and t[:j]==pref]
            if cands:
                v=Counter()
                for t,w in cands: v[t[j]]+=w
                preds[pos+j]=v.most_common(1)[0][0]; has[pos+j]=True
        prev = sent if sent in templates[s] else "<OTHER>"; pos+=len(sent)
    return preds, has

fl=hy=au=tot=used=uhit=0
for ts, s in te:
    ap, has = auto_preds(ts, s)
    for i in range(3, len(ts)):
        r = rb_pred(ts, i, s)
        h = ap[i] if has[i] else r
        fl += r==ts[i]; hy += h==ts[i]; tot += 1
        if has[i]: used+=1; uhit += ap[i]==ts[i]
print(f"\n=== final numbers (test n={tot:,}) ===")
print(f"  flat rulebook ({len(rb):,} rules)        : {fl/tot:.3f}")
print(f"  hybrid ({ntpl} templates + switches)   : {hy/tot:.3f}")
print(f"  automaton share: {used/tot:.0%} of positions at {uhit/used:.3f}")

# ---- typed data-to-text ----
rng = np.random.default_rng(6)
def realize(s, slots, n_sent=4):
    q = {k: list(v) for k, v in slots.items()}
    prev, out, used = "<START>", [], set()
    for _ in range(n_sent):
        tw = trans[s].get(prev, Counter())
        def fillable(t):   # every slot demanded by t must be available in the queues
            need = Counter(tok for tok in t if tok.startswith("<"))
            return all(len(q.get(k, [])) >= c for k, c in need.items())
        cands = [(t, templates[s][t]*(1+tw.get(t,0))) for t in templates[s]
                 if len(t) > 4 and t not in used and fillable(t)]
        if not cands: break     # out of facts -> natural stop
        ws = np.array([w for _,w in cands], float); ws/=ws.sum()
        t = cands[rng.choice(len(cands), p=ws)][0]
        used.add(t)
        for tok in t:
            out.append(str(q[tok].pop(0)) if tok.startswith("<") and q.get(tok) else tok)
        prev = t
    return " ".join(out)
print("\n=== typed data-to-text ===")
print("weather:", realize("weather", {"<city>":["Lisbon"], "<temp>":[21], "<speed>":[9],
      "<dir>":["southeast"], "<day>":["tuesday"], "<time>":["morning"]}))
print("recap  :", realize("recap", {"<team>":["Falcons","Bears","Hawks"], "<score>":[31,24],
      "<player>":["Jones"], "<points>":[18], "<day>":["saturday"], "<num>":[2]}))
