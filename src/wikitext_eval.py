"""Session-1 rigor: evaluate the healed combined hybrid on WIKITEXT.

Everything stays as fitted/healed on ministral text — templates, MLP lookup
tables, healed norms (healed_combined.pt). Only the evaluation text changes:
WikiText-103 validation slices. Kills the "your distilled corpus is easy"
objection, and tests transfer of the healing.

Reports intact vs hybrid (healed) on N wikitext chunks of 600 tokens.
"""
import torch
import replace_rich as RR
import replace_all as RA
from datasets import load_dataset

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
K_HEADS, K_MLP, NCHUNK, T_C = 160, 6, 6, 600

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:K_HEADS]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
MLPS = sorted(torch.load("mlp_solo_costs.pt")["costs"].items(), key=lambda kv: kv[1])
MLPS = [l for l, _ in MLPS[:K_MLP]]

# ---- wikitext chunks ----
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
text = "\n".join(r["text"] for r in ds if r["text"].strip())
toks = tokz.encode(text[:250000])
chunks = [toks[i*T_C:(i+1)*T_C] for i in range(NCHUNK)]
print(f"wikitext: {len(chunks)} chunks of {T_C} tokens")

# ---- MLP lookup tables refit on ministral chunks (same as heal_combined) ----
raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
fit_chunks = [tokz.encode(raw[o:o + 8000])[:T_C] for o in starts]
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hooks = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq in fit_chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in MLPS:
            o = cap[l]
            tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for hk in hooks: hk.remove()
MEAN = {l: tot[l] / cnt for l in MLPS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in MLPS}

def head_A(seq):
    base = RR.code_attn(seq)
    n = len(seq)
    out = {}
    for l, hs in BY_LAYER.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in RR.W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats).to(torch.float16)
    return out

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLDER = {"A": None, "L": None}
vcache, hooks = {}, []
def install():
    for l, hs in BY_LAYER.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            A = HOLDER["A"][l].to(DEV).float()
            v = vcache[l]
            for mi, h in enumerate(hs):
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A[mi] @ v[:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    for l in MLPS:
        def mhook(mod, inp, outp, l=l):
            return HOLDER["L"][l].unsqueeze(0)
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

def nll(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

# intact reference
intact = [nll(sq) for sq in chunks]

# hybrid with healed norms
norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
healed = torch.load("healed_combined.pt")["norms"]
for p, hn in zip(norm_params, healed): p.data.copy_(hn.to(DEV))
install()
hyb = []
for sq in chunks:
    HOLDER["A"], HOLDER["L"] = head_A(sq), {l: lut_mat(sq, l) for l in MLPS}
    hyb.append(nll(sq))
for hk in hooks: hk.remove()
for p, o in zip(norm_params, orig): p.data.copy_(o)

ti = torch.tensor(intact); th = torch.tensor(hyb)
print(f"\nWIKITEXT-103 validation ({NCHUNK} chunks x {T_C} tokens):")
print(f"  intact  {ti.mean():.3f} ± {ti.std():.3f}")
print(f"  hybrid  {th.mean():.3f} ± {th.std():.3f}   (+{(th-ti).mean():.3f} ± {(th-ti).std():.3f} nats)")
print(f"  per-chunk deltas: {[round(float(d),3) for d in (th-ti)]}")
print(f"  ministral-text reference: +0.681")
