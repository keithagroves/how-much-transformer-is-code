"""Reviewer push #2: before conceding "smooth", try a QUERY-CONDITIONAL rule.

The soft-rule bundle used only key-side / discourse-side features (recency,
salience, clause-subject, coref). Recency at 0.09-0.14 top-1 is consistent with
the head selecting on the QUERY token's identity, which no key-side vocabulary
can see. So we add the obvious query-conditional feature the reviewer names:

  cooc(i, j) = log1p( # of prior occurrences p<i of the CURRENT query token
                       seq[i], near which the candidate's entity token seq[j]
                       also occurs, within a +/-W window )   -- "attend to the
  entity that co-occurred with the current token."

Same sink-controlled, distributional (mass-overlap) protocol as probe_softrule.
Conditions: recency [0]; bundle [0,1,2,3]; bundle+q [0,1,2,3,4]; q-solo (recency
+ cooc) [0,4]; and the learned rank-1 direction as the un-interpreted ceiling.
If bundle+q or q-solo materially beats bundle/recency, "smooth by measurement"
was premature and the residue is partly a QUERY-conditional nameable rule.
"""
import collections, gc, math, sys, torch, spacy
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastcoref import FCoref

MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 320
KCAND = int(sys.argv[1]) if len(sys.argv) > 1 else 100
W = 10                                  # co-occurrence window (tokens)
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
coref = FCoref(device="cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size, cfg.vocab_size

torch.manual_seed(0)
sq = torch.randint(1000, V - 1000, (8, 50)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): out = model(ids, output_attentions=True)
qp = torch.arange(50, 99); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - 49].mean(dim=(0, 2))
del out; gc.collect()
HEADS = [(int(x) // NH, int(x) % NH) for x in ind.flatten().topk(6).indices]
LAYERS = sorted({l for l, _ in HEADS})
print(f"{MODEL}  heads: {HEADS}")

LAYER_LIST = model.gpt_neox.layers if hasattr(model, "gpt_neox") else model.model.layers
resid = {}
hooks = [LAYER_LIST[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())), with_kwargs=True)
    for l in LAYERS]

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)

def prep(o):
    text = raw[o:o + 6000]
    enc = tokz(text, return_offsets_mapping=True, add_special_tokens=False)
    ids_, offs = enc["input_ids"][:T_C], enc["offset_mapping"][:T_C]
    lim = offs[-1][1] + 5
    doc = nlp(text[:lim])
    subj_spans = [(t.idx, t.idx + len(t)) for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    subj = [any(a < sb and sa < b for sa, sb in subj_spans) for (a, b) in offs]
    pr = coref.predict(texts=[text[:lim]])[0]
    clus = [-1] * len(ids_)
    for ci, cl in enumerate(pr.get_clusters(as_strings=False)):
        for (sa, sb) in cl:
            for k, (a, b) in enumerate(offs):
                if a < sb and sa < b: clus[k] = ci
    return ids_, subj, clus

TRAIN = [prep(o) for o in range(40000, 460000, 60000)]
TEST = [prep(o) for o in (540000, 600000, 660000, 720000)]

def cooc_score(seq, i, j, pos_of):
    """# prior occurrences of query token seq[i] with candidate's entity token
    seq[j] within +/-W, all strictly before i (causal)."""
    prior_q = [p for p in pos_of[seq[i]] if p < i][-50:]
    if not prior_q: return 0.0
    occ = [q for q in pos_of[seq[j]] if q < i][-50:]
    if not occ: return 0.0
    c = 0
    for p in prior_q:
        if any(abs(q - p) <= W for q in occ): c += 1
    return math.log1p(c)

def collect(chunks):
    data = {hd: [] for hd in HEADS}
    for seq, subj, clus in chunks:
        n = len(seq); entset = set(j for j in range(n) if is_entity(seq[j]))
        pos_of = collections.defaultdict(list)
        for k in range(n): pos_of[seq[k]].append(k)
        freq = []; seen = collections.Counter()
        for k in range(n):
            freq.append(seen.copy())
            if k in entset: seen[seq[k]] += 1
        focus = [-1] * n; last = -1
        for k in range(n):
            if clus[k] != -1: last = clus[k]
            focus[k] = last
        with torch.no_grad(): o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu(); H = resid[l][0].float().cpu()
            for i in range(10, n):
                cand = [j for j in range(1, i) if j in entset][-KCAND:]
                if len(cand) < 3: continue
                row = A[i].clone(); row[0] = 0.0; s = row[1:i + 1].sum().item()
                if s < 1e-6: continue
                row = row / s
                cm = sum(row[j].item() for j in cand)
                if cm < 0.15: continue
                w = torch.tensor([row[j].item() for j in cand]); d = w / w.sum()
                feats = torch.tensor([[
                    -math.log(i - j),
                    math.log1p(freq[i].get(seq[j], 0)),
                    1.0 if subj[j] else 0.0,
                    1.0 if (focus[i] != -1 and clus[j] == focus[i]) else 0.0,
                    cooc_score(seq, i, j, pos_of),
                ] for j in cand])
                data[(l, h)].append((feats, d, H[i], H[torch.tensor(cand)]))
        del o; gc.collect()
    return data

print("collecting (spaCy + fastcoref + query-cooccurrence features)...")
TR = collect(TRAIN); TE = collect(TEST)
for hd in HEADS: print(f"  {hd}: {len(TR[hd])} train / {len(TE[hd])} test")

# diagnostic: how often is the query-cooc feature even active?
nz = [1.0 if feats[:, 4].abs().sum() > 0 else 0.0
      for hd in HEADS for feats, *_ in TE[hd]]
print(f"query-cooc feature nonzero on {sum(nz)/max(len(nz),1):.0%} of held-out samples")

def overlap(d, p): return torch.min(d, p).sum().item()

def fit_bundle(samples, cols, steps=300):
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.1, weight_decay=1e-3)
    for _ in range(steps):
        loss = 0.0
        for feats, d, _, _ in samples:
            loss = loss - (d * F.log_softmax(feats[:, cols] @ w, -1)).sum()
        (loss / len(samples)).backward(); opt.step(); opt.zero_grad()
    return w.detach()

def ev_bundle(samples, w, cols):
    return sum(overlap(d, F.softmax(feats[:, cols] @ w, -1)) for feats, d, _, _ in samples) / len(samples)

def fit_rank1(samples, steps=250, wd=1e-3):
    a = torch.zeros(1, requires_grad=True)
    P = (torch.randn(D, 1) * 0.02).requires_grad_(); Q = (torch.randn(D, 1) * 0.02).requires_grad_()
    opt = torch.optim.Adam([a, P, Q], lr=0.05, weight_decay=wd)
    for _ in range(steps):
        loss = 0.0
        for feats, d, hi, Hc in samples:
            lg = a * feats[:, 0] + (hi @ P) @ (Hc @ Q).T
            loss = loss - (d * F.log_softmax(lg, -1)).sum()
        (loss / len(samples)).backward(); opt.step(); opt.zero_grad()
    return a.detach(), P.detach(), Q.detach()

def ev_rank1(samples, a, P, Q):
    return sum(overlap(d, F.softmax(a * feats[:, 0] + (hi @ P) @ (Hc @ Q).T, -1))
               for feats, d, hi, Hc in samples) / len(samples)

import statistics as st
res = collections.defaultdict(list)
COND = {"recency": [0], "bundle": [0, 1, 2, 3], "bundle+q": [0, 1, 2, 3, 4], "q-solo(rec+cooc)": [0, 4]}
for hd in HEADS:
    if len(TR[hd]) < 20 or len(TE[hd]) < 8: continue
    for name, cols in COND.items():
        res[name].append(ev_bundle(TE[hd], fit_bundle(TR[hd], cols), cols))
    a, P, Q = fit_rank1(TR[hd])
    res["rank1"].append(ev_rank1(TE[hd], a, P, Q))

print(f"\nheld-out MASS-OVERLAP with the head's true selection (higher=better; {len(res['recency'])} heads):")
for k in ["recency", "bundle", "bundle+q", "q-solo(rec+cooc)", "rank1"]:
    if res[k]: print(f"  {k:>18}: {st.mean(res[k]):.3f}")
print("\nread: bundle+q >> bundle => a query-conditional rule names real residue,")
print("      'smooth' was premature. bundle+q ~ bundle => the query-cooc rule fails too.")
for hk in hooks: hk.remove()
