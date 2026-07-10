"""Post-heal greedy frontier (backward elimination). The pre-heal greedy rejects
heads that healing would rescue (heads-only damage heals +0.77 -> +0.14), so it
understates the frontier. Correct criterion = does a replacement hurt the HEALED
model. Tractable version:
  1. heal a superset (cheapest 200 heads + 6 MLPs) once
  2. leave-one-out on the healed model: revert each coded head to its weights; if
     held-out loss DROPS, that head's code is net-detrimental post-heal -> drop it
  3. re-heal the survivors, report size + damage vs the fixed cheapest-160 (+0.70)
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
LR, EPOCHS, SUPER = 3e-4, 20, 200

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
ORDER = sorted(nat_c, key=lambda k: nr[k] + rr[k])
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]

SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu()))) for l in MLPS]
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

def head_A(seq, heads):
    base = RR.code_attn(seq); n = len(seq); byl = {}
    for l, h in heads: byl.setdefault(l, []).append(h)
    out = {}
    for l, hs in byl.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in RR.W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append((h, M / M.sum(-1, keepdim=True).clamp(min=1e-9)))
        out[l] = mats
    return out
def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLD = {"A": None, "mlp": False, "skip": set()}       # skip = heads to leave un-substituted
vcache, hooks = {}, []
for l in sorted({l for l, _ in ORDER}):
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(m, inp, l=l):
        if HOLD["A"] is None or l not in HOLD["A"]: return None
        x = inp[0].clone()
        for h, A in HOLD["A"][l]:
            if (l, h) in HOLD["skip"]: continue
            g = h // GROUP; x[0, :, h*DH:(h+1)*DH] = A.to(DEV) @ vcache[l][:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(m, i, o, l=l):
        if not HOLD["mlp"]: return None
        return lut_mat(SEQ["s"], l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

SEQ = {"s": None}
def loss_set(heads_A, skip):
    HOLD["skip"] = skip; tot = 0.0
    for i, sq in enumerate(eval_chunks):
        SEQ["s"] = sq; HOLD["A"] = heads_A[i]; HOLD["mlp"] = True
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
        HOLD["A"] = None; HOLD["mlp"] = False
        tot += -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
    HOLD["skip"] = set(); return tot / len(eval_chunks)

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
HOLD["A"] = None
intact = sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0,:-1].float(),-1)
    .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item() for sq in eval_chunks)/len(eval_chunks)
print(f"intact held-out: {intact:.4f}")

def heal(heads):
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR)
    A_tr = [head_A(sq, heads) for sq in train_chunks]; model.train()
    for ep in range(EPOCHS):
        for sq, A in zip(train_chunks, A_tr):
            SEQ["s"] = sq; HOLD["A"] = A; HOLD["mlp"] = True; HOLD["skip"] = set()
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            HOLD["A"] = None; HOLD["mlp"] = False
    model.eval()

# 1. heal the superset
SUP = ORDER[:SUPER]
heal(SUP)
A_ev = [head_A(sq, SUP) for sq in eval_chunks]
full = loss_set(A_ev, set()) - intact
print(f"healed superset ({SUPER} heads): {full:+.3f} nats")

# 2. post-heal leave-one-out: harm_h = full - revert_h (>0 harmful, <0 helpful)
marg = []
for j, hd in enumerate(SUP):
    d = loss_set(A_ev, {hd}) - intact
    marg.append((hd, full - d))               # harm: how much its code adds to damage
    if (j + 1) % 50 == 0: print(f"  scanned {j+1}/{SUPER}", flush=True)
n_harm = sum(1 for _, h in marg if h > 1e-4)
print(f"post-heal net-harmful heads: {n_harm} of {SUPER}")

# 3. THRESHOLD FRONTIER: keep the k least-harmful heads (rank by marginal), re-heal.
# tune k (equivalently a harm threshold tau) to trade set size for damage.
ranked = [hd for hd, _ in sorted(marg, key=lambda x: x[1])]   # helpful-first, harmful-last
print(f"\nPOST-HEAL FRONTIER (keep k least-harmful of {SUPER}, re-healed, full held-out):")
print(f"{'k kept':>8}{'healed dmg':>12}")
front = {}
for k in [120, 160, SUPER - n_harm, SUPER]:
    k = max(1, min(SUPER, k))
    sub = ranked[:k]; heal(sub)
    A_sub = [head_A(sq, sub) for sq in eval_chunks]
    front[k] = loss_set(A_sub, set()) - intact
    print(f"{k:>8}{front[k]:>+12.3f}", flush=True); gc.collect()
# reference: fixed cheapest-160 (the paper's floor)
heal(ORDER[:160])
A160 = [head_A(sq, ORDER[:160]) for sq in eval_chunks]
fixed = loss_set(A160, set()) - intact
for hk_ in hooks: hk_.remove()
print(f"\n  fixed cheapest-160 (floor):   {fixed:+.3f}")
print(f"  superset-200 (no pruning):    {full:+.3f}")
print("read: at k=160 the post-heal-ranked set should beat the fixed floor; drop the")
print("      harmful tail (k = 200 - n_harm) for the lowest damage at the largest safe budget.")
