"""Probe, stage 1 (properly powered): does the diffuse induction attention
select among ENTITIES via a few learned directions, beyond recency?

Fixes over probe_rank.py: candidates restricted to the <=20 most recent ENTITY
positions (small, meaningful set -> random top-1 ~5%); target is the head's
attention over those; a RECENCY feature is always included; the rank-r content
bilinear is added on top; L2 regularization; many train chunks. Metric =
attention-MASS overlap (sum min(D,Dhat)) on held-out, plus top-1.

Baselines to beat: recency-only (r=0) and random-direction control. If the
learned rank-1/2 content term adds real overlap over recency, entity selection
is low-dimensional and learnable -> probe alive.
"""
import collections, gc, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L, KCAND = 320, 50, 20
RANKS = [0, 1, 2, 4]

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
V = cfg.vocab_size

torch.manual_seed(0)
sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): out = model(ids, output_attentions=True)
qpos = torch.arange(L, 2*L-1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
del out; gc.collect()
HEADS = [(int(x)//NH, int(x)%NH) for x in ind.flatten().topk(6).indices]
LAYERS = sorted({l for l, _ in HEADS})
print(f"heads: {HEADS}")

resid = {}
hooks = [model.model.layers[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())), with_kwargs=True)
    for l in LAYERS]

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)

def chunk(o): return tokz.encode(raw[o:o+6000])[:T_C]
TRAIN = [chunk(o) for o in range(0, 520000, 40000)]        # 13 chunks
TEST  = [chunk(o) for o in (540000, 580000, 620000, 660000)]

def collect(seqs):
    data = {hd: [] for hd in HEADS}
    for seq in seqs:
        n = len(seq)
        ent = [j for j in range(n) if is_entity(seq[j])]
        with torch.no_grad():
            o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu()
            H = resid[l][0].float().cpu()
            for i in range(10, n):
                cand = [j for j in ent if j < i][-KCAND:]
                if len(cand) < 3: continue
                w = A[i, cand]
                if w.sum() < 0.15: continue                # head not doing entity attn here
                dist = w / w.sum()
                rec = torch.tensor([-torch.tensor(float(i - j)).log() for j in cand])
                data[(l, h)].append((H[i], H[cand], dist, rec))
        del o; gc.collect()
    return data

print("collecting...")
TR = collect(TRAIN); TE = collect(TEST)
for hd in HEADS: print(f"  {hd}: {len(TR[hd])} train / {len(TE[hd])} test")

def fit(samples, r, steps=250, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    a = torch.zeros(1, requires_grad=True)                 # recency weight
    params = [a]
    if r > 0:
        P = (torch.randn(D, r) * 0.02).requires_grad_()
        Q = (torch.randn(D, r) * 0.02).requires_grad_()
        params += [P, Q]
    opt = torch.optim.Adam(params, lr=0.05, weight_decay=wd)
    for _ in range(steps):
        loss = 0.0
        for hi, Hc, dist, rec in samples:
            lg = a * rec
            if r > 0: lg = lg + (hi @ P) @ (Hc @ Q).T
            loss = loss - (dist * F.log_softmax(lg, -1)).sum()
        (loss / len(samples)).backward()
        opt.step(); opt.zero_grad()
    return (a.detach(), P.detach(), Q.detach()) if r > 0 else (a.detach(), None, None)

def ev(samples, a, P, Q, rand=False):
    ov, hit = 0.0, 0
    for hi, Hc, dist, rec in samples:
        lg = a * rec
        if P is not None:
            PP, QQ = (torch.randn_like(P), torch.randn_like(Q)) if rand else (P, Q)
            lg = lg + (hi @ PP) @ (Hc @ QQ).T
        pred = F.softmax(lg, -1)
        ov += torch.min(dist, pred).sum().item()
        hit += int(pred.argmax() == dist.argmax())
    return ov/len(samples), hit/len(samples)

import statistics as st
print(f"\n{'rank':>5}{'overlap':>10}{'top1':>8}{'rand-ov':>10}{'rand-t1':>9}")
for r in RANKS:
    ov, h1, rov, rh = [], [], [], []
    for hd in HEADS:
        if len(TR[hd]) < 20 or len(TE[hd]) < 8: continue
        a, P, Q = fit(TR[hd], r)
        o1, t1 = ev(TE[hd], a, P, Q)
        ov.append(o1); h1.append(t1)
        if r > 0:
            o2, t2 = ev(TE[hd], a, P, Q, rand=True)
            rov.append(o2); rh.append(t2)
    line = f"{r:>5}{st.mean(ov):>10.2f}{st.mean(h1):>8.0%}"
    if r > 0: line += f"{st.mean(rov):>10.2f}{st.mean(rh):>9.0%}"
    else: line += f"{'(recency baseline)':>19}"
    print(line, flush=True)
for hk in hooks: hk.remove()
