"""Reviewer push: is "smooth by measurement" an artifact of scoring only
per-position SYMBOLIC rules by ARGMAX? Maybe the selection is a SOFT weighting of
nameable features that no argmax-over-rules vocabulary can win. Test it the way
the reviewer suggests — distributionally, as a feature-bundle regression, against
the learned rank-1 direction as the "soft but un-interpreted" ceiling.

Sink-controlled (BOS removed, context-renormalized); d = the head's true
candidate-renormalized distribution over entity candidates. We fit soft models
predicting d and score by MASS-OVERLAP sum(min(d,p)) on held-out (a distributional
metric, not top-1):
  recency     softmax(a * -log(dist))                      weak positional baseline
  bundle      softmax(w . [recency, salience, subject, coref])   nameable FEATURES
  rank1       softmax(a*rec + (h_i P)(h_cand Q))           learned residual direction
  rank1-rand  same with a random direction                 control
If bundle >> recency and ~ rank1: the selection is a SOFT combination of nameable
features -> "un-nameable" softens to "soft-rule". If bundle ~ recency << rank1:
even soft feature bundles fail, the direction encodes non-feature content -> the
un-nameable claim holds distributionally too, and hardens.
"""
import collections, gc, math, sys, torch, spacy
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastcoref import FCoref

MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 320
KCAND = int(sys.argv[1]) if len(sys.argv) > 1 else 100
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

def collect(chunks):
    data = {hd: [] for hd in HEADS}
    for seq, subj, clus in chunks:
        n = len(seq); entset = set(j for j in range(n) if is_entity(seq[j]))
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
                ] for j in cand])
                data[(l, h)].append((feats, d, H[i], H[torch.tensor(cand)]))
        del o; gc.collect()
    return data

print("collecting (spaCy + fastcoref features + residuals)...")
TR = collect(TRAIN); TE = collect(TEST)
for hd in HEADS: print(f"  {hd}: {len(TR[hd])} train / {len(TE[hd])} test")

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

def ev_rank1(samples, a, P, Q, rand=False):
    tot = 0.0
    for feats, d, hi, Hc in samples:
        PP = torch.randn_like(P) if rand else P; QQ = torch.randn_like(Q) if rand else Q
        lg = a * feats[:, 0] + (hi @ PP) @ (Hc @ QQ).T
        tot += overlap(d, F.softmax(lg, -1))
    return tot / len(samples)

import statistics as st
res = collections.defaultdict(list)
for hd in HEADS:
    if len(TR[hd]) < 20 or len(TE[hd]) < 8: continue
    res["recency"].append(ev_bundle(TE[hd], fit_bundle(TR[hd], [0]), [0]))
    res["bundle"].append(ev_bundle(TE[hd], fit_bundle(TR[hd], [0, 1, 2, 3]), [0, 1, 2, 3]))
    for wd, tag in [(1e-3, "rank1_wd1e-3"), (1e-2, "rank1_wd1e-2"), (5e-2, "rank1_wd5e-2")]:
        a, P, Q = fit_rank1(TR[hd], wd=wd)
        res[tag].append(ev_rank1(TE[hd], a, P, Q))
    res["rank1_rand"].append(ev_rank1(TE[hd], a, P, Q, rand=True))

print(f"\nheld-out MASS-OVERLAP with the head's true selection distribution (higher=better):")
for k in ["recency", "bundle", "rank1_wd1e-3", "rank1_wd1e-2", "rank1_wd5e-2", "rank1_rand"]:
    print(f"  {k:>12}: {st.mean(res[k]):.3f}")
print(f"\nread: bundle>>recency & ~rank1 => selection is a soft combo of NAMEABLE features"
      f"\n      (softens 'un-nameable'). bundle~recency << rank1 => even soft feature bundles"
      f"\n      fail; the direction encodes non-feature content (hardens 'un-nameable').")
for hk in hooks: hk.remove()
