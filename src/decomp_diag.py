"""Pin the WikiText +7.0 unhealed anomaly: decompose by component set (eval-only).

replace_rich's frontier said nat-only 160 heads (attention-only) = +1.83 unhealed,
but the heal harness (combined-rank 160 heads + 6 solo-cheapest MLPs) measures
+6.99 unhealed. Conditions:
  A  heads, nat-only 160, no MLPs      (should ~reproduce +1.83)
  B  heads, combined 160, no MLPs      (selection effect on heads)
  C  MLPs only, the 6 solo-cheapest    (joint MLP effect; solo said ~0)
  D  MLPs only, middle-6 [9..15 band]  (the Colab-like set, for contrast)
  E  combined 160 + solo-cheapest 6    (the anomalous full set)
  F  combined 160 + middle-6           (fix candidate)
Run with SUB_CORPUS=wikitext_corpus.txt from wikitext_run/.
"""
import torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
H_COMB = sorted(nat_c, key=lambda k: nr[k] + rr[k])[:160]
H_NAT = sorted(nat_c, key=lambda k: nat_c[k])[:160]
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
M_SOLO = sorted(mcosts, key=lambda l: mcosts[l])[:6]
M_MID = [9, 10, 11, 12, 13, 14]
print(f"solo-cheapest MLPs: {sorted(M_SOLO)}  | middle set: {M_MID}")

import os as _os
raw = open(_os.environ.get("SUB_CORPUS", "ministral_corpus.txt")).read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in train_starts]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]

ALL_MLPS = sorted(set(M_SOLO) | set(M_MID))
SUM = {l: {} for l in ALL_MLPS}; tot = {l: None for l in ALL_MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda m, i, o, l=l: cap.__setitem__(l, o[0].detach().float().cpu()))) for l in ALL_MLPS]
cnt = 0
with torch.no_grad():
    for sq in train_chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in ALL_MLPS:
            o = cap[l]; tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for h in hk: h.remove()
MEAN = {l: tot[l] / cnt for l in ALL_MLPS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in ALL_MLPS}

ACTIVE = {"heads": None, "mlps": set(), "A": None, "seq": None}
vcache, hooks = {}, []
layers_used = sorted({l for l, _ in set(H_COMB) | set(H_NAT)})
for l in layers_used:
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(mod, inp, l=l):
        if ACTIVE["A"] is None or l not in ACTIVE["A"]: return None
        x = inp[0].clone()
        for h, Ah in ACTIVE["A"][l]:
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = Ah.to(DEV) @ vcache[l][:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in ALL_MLPS:
    def mhook(mod, inp, outp, l=l):
        if l not in ACTIVE["mlps"]: return None
        sq = ACTIVE["seq"]
        return torch.stack([LUT[l].get(t, MEAN[l]) for t in sq]).to(DEV).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

def head_A(seq, heads):
    if not heads: return None
    base = RR.code_attn(seq); n = len(seq); out = {}
    for l, h in heads:
        M = torch.zeros(n, n)
        for k, wk in RR.W[(l, h)].items():
            if wk > 1e-4: M += wk * base[k]
        out.setdefault(l, []).append((h, M / M.sum(-1, keepdim=True).clamp(min=1e-9)))
    return out

def dmg(heads, mlps):
    tot_l = 0.0
    for sq in eval_chunks:
        ACTIVE["A"] = head_A(sq, heads); ACTIVE["mlps"] = set(mlps); ACTIVE["seq"] = sq
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
        tot_l += -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
    ACTIVE["A"] = None; ACTIVE["mlps"] = set()
    return tot_l / len(eval_chunks)

intact = dmg([], [])
print(f"intact {intact:.4f}", flush=True)
for name, hs, ms in [
    ("A heads nat-160 only", H_NAT, []),
    ("B heads comb-160 only", H_COMB, []),
    ("C MLPs solo-cheapest-6", [], M_SOLO),
    ("D MLPs middle-6", [], M_MID),
    ("E comb-160 + solo-6 (anomaly)", H_COMB, M_SOLO),
    ("F comb-160 + middle-6 (fix?)", H_COMB, M_MID),
]:
    print(f"{name:>32}: {dmg(hs, ms) - intact:+.4f}", flush=True)
for hk_ in hooks: hk_.remove()
