"""Resolve the +1.61 / +0.68 unhealed-damage discrepancy. +1.61 was heads+MLPs on
a single chunk (test_seq); the selection experiment reported +0.68 for heads-only
on 4 other chunks. Measure the full decomposition (intact / heads-only / MLPs-only
/ heads+MLPs) on a COMMON eval set to confirm it's a config difference, not error.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP, NL = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL
tokz = RA.tokz
RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:160]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]
print(f"160 heads + MLPs {sorted(MLPS)}")

raw = open("ministral_corpus.txt").read()
# same training chunks heal_combined used, to fit identical MLP lookup tables
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in starts]

print("fitting MLP lookup tables (as in heal_combined)...")
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
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

def evl(seq, heads_on, mlps_on):
    vcache, hooks = {}, []
    if heads_on:
        A = head_A(seq)
        for l, hs in BY_LAYER.items():
            attn = model.model.layers[l].self_attn
            def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
            hooks.append(attn.v_proj.register_forward_hook(vhook))
            def ohook(mod, inp, l=l, hs=hs):
                x = inp[0].clone(); Al = A[l].to(DEV)
                for mi, h in enumerate(hs):
                    g = h // GROUP
                    x[0, :, h*DH:(h+1)*DH] = Al[mi] @ vcache[l][:, g*DH:(g+1)*DH]
                return (x,) + inp[1:]
            hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    if mlps_on:
        for l in MLPS:
            Ll = lut_mat(seq, l)
            def mhook(mod, inp, outp, l=l, Ll=Ll): return Ll.unsqueeze(0)
            hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for h in hooks: h.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

N = min(len(raw) - 9000, 1_000_000)
EVAL = {"test_seq": RA.test_seq}
for o in (700_000, 780_000, 860_000, 940_000):
    if o < N: EVAL[f"chunk@{o//1000}k"] = tokz.encode(raw[o:o + 8000])[:300]

print(f"\n{'eval set':>12}{'intact':>8}{'heads':>8}{'MLPs':>8}{'both':>8}"
      f"{'dHeads':>8}{'dMLPs':>8}{'dBoth':>8}")
agg = {k: [] for k in ("dH", "dM", "dB")}
for name, seq in EVAL.items():
    it = evl(seq, False, False)
    hd = evl(seq, True, False)
    ml = evl(seq, False, True)
    bo = evl(seq, True, True)
    agg["dH"].append(hd - it); agg["dM"].append(ml - it); agg["dB"].append(bo - it)
    print(f"{name:>12}{it:>8.3f}{hd:>8.3f}{ml:>8.3f}{bo:>8.3f}"
          f"{hd-it:>+8.3f}{ml-it:>+8.3f}{bo-it:>+8.3f}", flush=True)
    gc.collect()
m = lambda x: sum(x) / len(x)
print(f"{'POOLED':>12}{'':>8}{'':>8}{'':>8}{'':>8}"
      f"{m(agg['dH']):>+8.3f}{m(agg['dM']):>+8.3f}{m(agg['dB']):>+8.3f}")
print(f"\ntest_seq heads+MLPs damage should reproduce ~+1.61 (the paper's number);")
print(f"held-out chunks heads-only should reproduce ~+0.68 (the selection number).")
