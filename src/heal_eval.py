"""Pooled healed damage for the combined hybrid, so unhealed/healed headline
numbers share one multi-chunk basis (test_seq alone ran high). Loads the saved
healed norms, installs 160 heads + 6 MLPs, evaluates over the same 5 eval sets
as disambig.py.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
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

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in starts]

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

def evl(seq):
    A = head_A(seq); vcache, hooks = {}, []
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
    for l in MLPS:
        Ll = lut_mat(seq, l)
        def mhook(mod, inp, outp, l=l, Ll=Ll): return Ll.unsqueeze(0)
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for h in hooks: h.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def intact(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

N = min(len(raw) - 9000, 1_000_000)
EVAL = {"test_seq": RA.test_seq}
for o in (700_000, 780_000, 860_000, 940_000):
    if o < N: EVAL[f"chunk@{o//1000}k"] = tokz.encode(raw[o:o + 8000])[:300]

# apply healed norms
norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
healed = torch.load("healed_combined.pt")["norms"]
assert len(healed) == len(norm_params)
for p, b in zip(norm_params, healed): p.data.copy_(b.to(DEV))

print(f"{'eval set':>12}{'intact':>8}{'healed':>8}{'dHealed':>9}")
dh = []
for name, seq in EVAL.items():
    it = intact(seq); hl = evl(seq)
    dh.append(hl - it)
    print(f"{name:>12}{it:>8.3f}{hl:>8.3f}{hl-it:>+9.3f}", flush=True)
    gc.collect()
print(f"{'POOLED':>12}{'':>8}{'':>8}{sum(dh)/len(dh):>+9.3f}")
print(f"\ntest_seq healed damage should reproduce ~+0.68 (the paper's number).")
