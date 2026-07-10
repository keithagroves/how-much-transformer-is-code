"""Damage vs. substitution budget: is +0.64 one point on a rising curve?

Substitute the cheapest-first k attention heads (code templates) plus the 6 coded
MLPs, and measure held-out damage as k grows. Unhealed across a fine grid (cheap,
shows the shape) and healed at three points (the real operating cost). Confirms the
'one operating point, not a constant' framing directly: if the curve is nearly flat
at small k and steepens, small budgets are near-free and 36% is a chosen tolerance.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS, T_TR, LR = 20, 600, 3e-4

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS_ALL = sorted(nat_c, key=lambda k: nr[k] + rr[k])          # cheapest-first order, all heads
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

# LUT for the 6 MLPs
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
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = (hs, torch.stack(mats))
    return out

def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLD = {"A": None, "mlp": False}
vcache, hooks = {}, []
LAY = sorted({l for l, _ in HEADS_ALL})
for l in LAY:
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(m, inp, l=l):
        if HOLD["A"] is None or l not in HOLD["A"]: return None
        hs, A = HOLD["A"][l]; x = inp[0].clone(); A = A.to(DEV)
        for mi, h in enumerate(hs):
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = A[mi] @ vcache[l][:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(m, i, o, l=l):
        if not HOLD["mlp"]: return None
        return lut_mat(SEQ["s"], l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

SEQ = {"s": None}
def loss(seq, heads_A, mlp):
    SEQ["s"] = seq; HOLD["A"] = heads_A; HOLD["mlp"] = mlp
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    HOLD["A"] = None; HOLD["mlp"] = False
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
intact = sum(loss(sq, None, False) for sq in eval_chunks) / len(eval_chunks)
print(f"intact held-out loss: {intact:.4f}\n")

GRID = [10, 20, 40, 80, 120, 160, 200, 256, 320]
print("UNHEALED damage vs #heads (with 6 coded MLPs) and heads-only:")
print(f"{'k heads':>8}{'+MLPs':>10}{'heads-only':>12}")
Acache = {}
for k in GRID:
    heads = HEADS_ALL[:k]
    A_ev = [head_A(sq, heads) for sq in eval_chunks]
    d_both = sum(loss(sq, A, True) for sq, A in zip(eval_chunks, A_ev)) / len(eval_chunks) - intact
    d_head = sum(loss(sq, A, False) for sq, A in zip(eval_chunks, A_ev)) / len(eval_chunks) - intact
    print(f"{k:>8}{d_both:>+10.3f}{d_head:>+12.3f}", flush=True)
    gc.collect()

print("\nHEALED damage (heads+MLPs) at three budgets:")
A_tr_cache = {}
for k in [80, 160, 240]:
    heads = HEADS_ALL[:k]
    A_tr = [head_A(sq, heads) for sq in train_chunks]
    A_ev = [head_A(sq, heads) for sq in eval_chunks]
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR); model.train()
    for ep in range(EPOCHS):
        for sq, A in zip(train_chunks, A_tr):
            SEQ["s"] = sq; HOLD["A"] = A; HOLD["mlp"] = True
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            HOLD["A"] = None; HOLD["mlp"] = False
    model.eval()
    dh = sum(loss(sq, A, True) for sq, A in zip(eval_chunks, A_ev)) / len(eval_chunks) - intact
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    print(f"  k={k:>3} heads + 6 MLPs, healed: {dh:+.3f} nats", flush=True)
    gc.collect()
for hk_ in hooks: hk_.remove()
print("\nread: convex, near-flat then steep => small budgets nearly free, 36% is a tolerance.")
