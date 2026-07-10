"""Shuffled-code heal control: does norm-healing recover FUNCTION, or just
reroute around whatever the code emits?

Reviewer's load-bearing test. Take the combined hybrid (160 heads + 6 MLPs),
but permute each surrogate's OUTPUT across sequence positions with a fixed
per-sequence permutation before writing it into the residual stream. The code
then carries the right marginal statistics (same bag of output vectors) but the
wrong per-position function. Heal norms against it under the SAME protocol as
real code, and compare held-out healed damage:

  real code + heal      -> ~+0.64 nats (function preserved)
  shuffled code + heal  -> if healing manufactures function, ~+0.64 too (BAD);
                           if code carries function, collapses toward the
                           zero-ablation level (~+2 nats) because norms cannot
                           undo a position permutation (they are per-channel).

Both conditions healed fresh here (identical epochs/LR/no early-stopping) and
evaluated on offsets held out of LUT + heal (0 mod 40000; training is 20000 mod
40000).
"""
import gc, random, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS, T_TR, LR = 20, 600, 3e-4

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
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]
print(f"held-out eval offsets (k): {[o//1000 for o in eval_starts]}")

# ---- MLP LUT from training chunks (real) ----
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

def perm_for(seq, idx):
    g = torch.Generator().manual_seed(1000 + idx)
    return torch.randperm(len(seq), generator=g).to(DEV)

# ---- hooks: apply surrogate, optionally permuting output rows across positions ----
HOLDER = {"A": None, "L": None, "perm": None, "shuf": False}
hooks = []
vcache = {}
for l, hs in BY_LAYER.items():
    attn = model.model.layers[l].self_attn
    def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
    hooks.append(attn.v_proj.register_forward_hook(vhook))
    def ohook(mod, inp, l=l, hs=hs):
        if HOLDER["A"] is None: return None          # intact passthrough
        x = inp[0].clone(); A = HOLDER["A"][l].to(DEV); v = vcache[l]
        p = HOLDER["perm"]
        for mi, h in enumerate(hs):
            g = h // GROUP
            o = A[mi] @ v[:, g*DH:(g+1)*DH]
            if HOLDER["shuf"]: o = o[p]
            x[0, :, h*DH:(h+1)*DH] = o
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(mod, inp, outp, l=l):
        if HOLDER["A"] is None: return None          # intact passthrough
        L = HOLDER["L"][l]
        if HOLDER["shuf"]: L = L[HOLDER["perm"]]
        return L.unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

# precompute surrogates + perms
TR = [(sq, head_A(sq), {l: lut_mat(sq, l) for l in MLPS}, perm_for(sq, i))
      for i, sq in enumerate(train_chunks)]
EV = [(sq, head_A(sq), {l: lut_mat(sq, l) for l in MLPS}, perm_for(sq, 9000 + i))
      for i, sq in enumerate(eval_chunks)]

def loss_of(seq, A, L, perm, shuf):
    HOLDER.update(A=A, L=L, perm=perm, shuf=shuf)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def intact(seq):
    HOLDER["A"] = None
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]

def heal_and_eval(shuf):
    for p, o in zip(norm_params, orig): p.data.copy_(o)   # reset norms
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR)
    model.train()
    for ep in range(EPOCHS):
        for sq, A, L, perm in TR:
            HOLDER.update(A=A, L=L, perm=perm, shuf=shuf)
            ids = torch.tensor([sq]).to(DEV)
            out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
    model.eval()
    hl = [loss_of(sq, A, L, perm, shuf) for sq, A, L, perm in EV]
    for p, o in zip(norm_params, orig): p.data.copy_(o)   # restore
    return [h - it for h, it in zip(hl, INTACT)]

def ci(dmg):
    random.seed(0); B = 5000; n = len(dmg)
    bs = sorted(sum(dmg[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(dmg)/n, bs[int(.025*B)], bs[int(.975*B)]

# fixed intact reference: ORIGINAL model, no code, no heal (used for both conditions)
INTACT = [intact(sq) for sq, *_ in EV]
print(f"intact (original) mean loss: {sum(INTACT)/len(INTACT):.3f}", flush=True)
print("healing REAL code...", flush=True)
d_real = heal_and_eval(False)
print("healing SHUFFLED code...", flush=True)
d_shuf = heal_and_eval(True)
for hk in hooks: hk.remove()

mr, lor, hir = ci(d_real)
ms, los, his = ci(d_shuf)
print(f"\n{'condition':>16}{'healed damage':>16}{'   95% CI':>18}")
print(f"{'real code':>16}{mr:>+16.3f}   [{lor:+.3f}, {hir:+.3f}]")
print(f"{'shuffled code':>16}{ms:>+16.3f}   [{los:+.3f}, {his:+.3f}]")
print(f"\ndelta (shuffled - real): {ms-mr:+.3f} nats")
print("if shuffled >> real, healing recovers FUNCTION not just distribution: the")
print("norm heal cannot repair position-scrambled code, so the code was doing real work.")
