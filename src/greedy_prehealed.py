"""Efficient PRE-heal frontier (companion to greedy_posthealed.py). Same
leave-one-out-from-superset structure, but rank heads by their marginal harm on
the UNHEALED model, then re-heal the k-least-harmful sets. Fast because each
head's code-attention is cached once and leave-one-out just skips a head (no
rebuild) — the slow O(set x candidates) sequential greedy is avoided.

The pre-vs-post comparison is the point: if the unhealed ranking picks the same
heads to drop as the healed one, pre-heal is a cheap proxy; if not, healing
rescues specific heads and only the post-heal criterion is trustworthy.
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

HOLD = {"A": None, "mlp": False, "skip": set()}
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

HOLD["A"] = None
intact = sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0,:-1].float(),-1)
    .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item() for sq in eval_chunks)/len(eval_chunks)
print(f"intact held-out: {intact:.4f}")

# 1. UNHEALED superset (no heal), cache A once
SUP = ORDER[:SUPER]
A_ev = [head_A(sq, SUP) for sq in eval_chunks]
full = loss_set(A_ev, set()) - intact
print(f"UNHEALED superset ({SUPER} heads): {full:+.3f} nats")

# 2. leave-one-out on the UNHEALED model: harm_h = full - revert_h
marg = []
for j, hd in enumerate(SUP):
    d = loss_set(A_ev, {hd}) - intact
    marg.append((hd, full - d))
    if (j + 1) % 50 == 0: print(f"  scanned {j+1}/{SUPER}", flush=True)
n_harm = sum(1 for _, h in marg if h > 1e-4)
harmful_pre = set(hd for hd, h in marg if h > 1e-4)
print(f"pre-heal net-harmful heads (unhealed): {n_harm} of {SUPER}")
# save ranking for the pre-vs-post comparison
torch.save({"marg": marg, "harmful": harmful_pre}, "prefrontier_rank.pt")

# 3. threshold frontier: re-heal the k least-harmful
ranked = [hd for hd, _ in sorted(marg, key=lambda x: x[1])]
print(f"\nPRE-HEAL-RANKED FRONTIER (keep k least-harmful of {SUPER}, re-healed):")
print(f"{'k kept':>8}{'healed dmg':>12}")
for k in [120, 160, SUPER - n_harm, SUPER]:
    k = max(1, min(SUPER, k)); sub = ranked[:k]; heal(sub)
    A_sub = [head_A(sq, sub) for sq in eval_chunks]
    print(f"{k:>8}{loss_set(A_sub, set()) - intact:>+12.3f}", flush=True); gc.collect()
heal(ORDER[:160])
A160 = [head_A(sq, ORDER[:160]) for sq in eval_chunks]
fixed = loss_set(A160, set()) - intact
for hk_ in hooks: hk_.remove()
print(f"\n  fixed cheapest-160 (floor):   {fixed:+.3f}")
print("read: compare this pre-heal frontier + harmful set to greedy_posthealed's;")
print("      divergence = healing rescues specific heads (only post-heal is trustworthy).")
