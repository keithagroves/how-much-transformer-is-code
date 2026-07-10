"""Vector rulebook: rules point into EMBEDDING SPACE, not at tokens.

Each rule (context, count>=MIN) stores a TARGET VECTOR = count-weighted mean of
the qwen embeddings of its observed next tokens. Prediction = nearest vocab
embeddings to the fired rule's target. This is what a neural LM's output layer
does; here the map context->vector is a legible rule table.

Metrics vs the token rulebook (same fired rules, argmax token):
  exact top-1 | top-10 recall | mean cosine(predicted emb, truth emb)
The cosine metric is the real objective under this framing: how close in space
did the rule point?
"""
import json, re, sys, numpy as np, requests
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

# ---- token embedding table: 16.4k words + punctuation marks ----
words = json.load(open("vocab_words.json"))
E = np.load("vocab_emb.npy")
punct = sorted(set(t for t in train + test if re.fullmatch(r"[^\w\s]", t)))
r = requests.post("http://localhost:11434/api/embed", json={
    "model": "qwen3-embedding:0.6b", "input": punct}, timeout=300)
PE = np.asarray(r.json()["embeddings"], dtype=np.float32)
PE /= np.linalg.norm(PE, axis=1, keepdims=True)
toks_all = words + punct
EMB = np.vstack([E, PE]).astype(np.float32)
tid = {t: i for i, t in enumerate(toks_all)}
print(f"embedding table: {len(toks_all):,} tokens", file=sys.stderr)

# ---- build rules: token argmax + target vector per context ----
ctx_counts = defaultdict(Counter)
for i in range(len(train)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: ctx_counts[tuple(train[i-o+1:i])][train[i]] += 1

contexts, tok_rule, targets = [], [], []
for k, c in ctx_counts.items():
    tot = sum(c.values())
    if tot < MIN: continue
    contexts.append(k)
    tok_rule.append(c.most_common(1)[0][0])
    v = np.zeros(EMB.shape[1], np.float32); m = 0
    for w, n in c.items():
        if w in tid: v += n * EMB[tid[w]]; m += n
    targets.append(v/np.linalg.norm(v) if m and np.linalg.norm(v) > 0 else None)
rule_id = {k: j for j, k in enumerate(contexts)}
print(f"rules: {len(contexts):,}", file=sys.stderr)

def fire(seg, i):
    for o in range(MAXN, 1, -1):
        k = tuple(seg[max(0, i-o+1):i])
        if len(k) == o-1 and k in rule_id: return rule_id[k]
    return None

# ---- evaluate on test ----
pos = [(i, fire(test, i)) for i in range(3, len(test))]
pos = [(i, j) for i, j in pos if j is not None and targets[j] is not None and test[i] in tid]
print(f"eval positions (rule fired, truth in vocab): {len(pos):,}", file=sys.stderr)

T = np.stack([targets[j] for _, j in pos])                  # target vectors
truth_ids = np.array([tid[test[i]] for i, _ in pos])
tok_preds = np.array([tid.get(tok_rule[j], -1) for _, j in pos])

ex_v = ex_t = r10 = 0
cos_v = cos_t = 0.0
B = 4096
for s in range(0, len(pos), B):
    sims = T[s:s+B] @ EMB.T                                  # (b, |V|)
    top10 = np.argpartition(-sims, 10, axis=1)[:, :10]
    best = sims.argmax(1)
    tr = truth_ids[s:s+B]
    ex_v += (best == tr).sum()
    r10 += (top10 == tr[:, None]).any(1).sum()
    cos_v += (EMB[best] * EMB[tr]).sum(1).sum()
    tp = tok_preds[s:s+B]
    ex_t += (tp == tr).sum()
    cos_t += (EMB[np.clip(tp, 0, None)] * EMB[tr]).sum(1).sum()
n = len(pos)
print(f"\n=== token rulebook vs vector rulebook (test, n={n:,}) ===")
print(f"  {'':<22}{'exact top1':>11}{'top10 recall':>14}{'mean cos to truth':>19}")
print(f"  {'token rule (argmax)':<22}{ex_t/n:>11.3f}{'--':>14}{cos_t/n:>19.3f}")
print(f"  {'vector rule (nearest)':<22}{ex_v/n:>11.3f}{r10/n:>14.3f}{cos_v/n:>19.3f}")

# ---- make the vector field visible: what regions do rules point at? ----
print("\n=== rules as regions: nearest tokens to some target vectors ===")
for probe in [("she","could","not"), ("i","am"), (",","and"), ("of","the")]:
    if probe in rule_id and targets[rule_id[probe]] is not None:
        v = targets[rule_id[probe]]
        near = np.argsort(-(EMB @ v))[:6]
        print(f"  after {' '.join(probe)!r}: " + ", ".join(toks_all[j] for j in near))
