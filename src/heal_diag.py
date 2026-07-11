"""Diagnose the WikiText heal divergence: heal REAL code with per-epoch logging.

heal_shuffle's 20-epoch blind heal produced +5.83 on WikiText (worse than the
~+2.0 unhealed), while the identical recipe gave +0.705 on fiction and the Colab
bf16 early-stopped heal was stable on WikiText. This script re-runs the real-code
heal only, printing train loss + held-out damage EVERY epoch, and tracks the best
epoch — so we can see whether the failure is slow overfit or optimizer blowup,
and what a val-selected heal would have reported.
Run with SUB_CORPUS=wikitext_corpus.txt from wikitext_run/.
"""
import torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS, T_TR, LR = 20, 600, 3e-4

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nr[k] + rr[k])[:160]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

import os as _os
raw = open(_os.environ.get("SUB_CORPUS", "ministral_corpus.txt")).read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]
print(f"heads {len(HEADS)}, MLPs {MLPS}; eval offsets {[o//1000 for o in eval_starts]}")

# MLP LUT from train chunks
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda m, i, o, l=l: cap.__setitem__(l, o[0].detach().float().cpu()))) for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq in train_chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in MLPS:
            o = cap[l]; tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for h in hk: h.remove()
MEAN = {l: tot[l] / cnt for l in MLPS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in MLPS}

def head_A(seq):
    base = RR.code_attn(seq); n = len(seq); out = {}
    for l, hs in BY_LAYER.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in RR.W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats)
    return out

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLDER = {"A": None, "L": None}
vcache, hooks = {}, []
for l in sorted(BY_LAYER):
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(mod, inp, l=l, hs=BY_LAYER[l]):
        if HOLDER["A"] is None: return None
        x = inp[0].clone(); A = HOLDER["A"][l].to(DEV); v = vcache[l]
        for mi, h in enumerate(hs):
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = A[mi] @ v[:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(mod, inp, outp, l=l):
        if HOLDER["A"] is None: return None
        return HOLDER["L"][l].unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

TR = [(sq, head_A(sq), {l: lut_mat(sq, l) for l in MLPS}) for sq in train_chunks]
EV = [(sq, head_A(sq), {l: lut_mat(sq, l) for l in MLPS}) for sq in eval_chunks]

def loss_of(sq, A, L):
    HOLDER.update(A=A, L=L)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def intact(sq):
    HOLDER["A"] = None
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()

INTACT = [intact(sq) for sq, *_ in EV]
print(f"intact held-out {sum(INTACT)/len(INTACT):.4f}")
d0 = sum(loss_of(*e) for e in EV)/len(EV) - sum(INTACT)/len(INTACT)
print(f"unhealed code damage: {d0:+.4f}", flush=True)

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
best = (d0, 0)
for ep in range(1, EPOCHS + 1):
    model.train(); tl = 0.0
    for sq, A, L in TR:
        HOLDER.update(A=A, L=L)
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
        tl += out.loss.item()
    model.eval()
    d = sum(loss_of(*e) for e in EV)/len(EV) - sum(INTACT)/len(INTACT)
    if d < best[0]: best = (d, ep)
    gnorm = max(p.abs().max().item() for p in norm_params)
    print(f"epoch {ep:>2}: train {tl/len(TR):.4f}  held-out damage {d:+.4f}  max|gain| {gnorm:.2f}", flush=True)
for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk_ in hooks: hk_.remove()
print(f"\nbest held-out damage {best[0]:+.4f} at epoch {best[1]} (0 = unhealed)")
print("read: damage rising from epoch 1 = heal diverges on this corpus; falling then")
print("rising = overfit (val-selected heal is the fix); falling throughout = 20-epoch")
print("blind heal was fine and heal_shuffle's number needs a different explanation.")
