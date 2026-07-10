"""Healed decomposition: does the norm-heal repair head damage and MLP damage
equally, or is the super-additive interaction the part it mainly fixes?

Unhealed (headline chunk, from disambig.py): heads-only +0.77, MLPs-only +0.48,
both +1.61, interaction +0.36. Here we heal each condition under the matched
protocol and read the held-out healed damage, so we can compare
heal(heads)+heal(MLPs) against heal(both).
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
HEADS = sorted(nat_c, key=lambda k: nr[k] + rr[k])[:160]
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
def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLD = {"A": None, "heads": False, "mlp": False}
vcache, hooks = {}, []
for l in BY_LAYER:
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(m, inp, l=l):
        if not HOLD["heads"]: return None
        hs = BY_LAYER[l]; A = HOLD["A"][l].to(DEV); x = inp[0].clone()
        for mi, h in enumerate(hs):
            g = h // GROUP; x[0, :, h*DH:(h+1)*DH] = A[mi] @ vcache[l][:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(m, i, o, l=l):
        if not HOLD["mlp"]: return None
        return lut_mat(SEQ["s"], l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

SEQ = {"s": None}
def loss(seq, A, heads, mlp):
    SEQ["s"] = seq; HOLD["A"] = A; HOLD["heads"] = heads; HOLD["mlp"] = mlp
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    HOLD["heads"] = HOLD["mlp"] = False
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
A_tr = [head_A(sq) for sq in train_chunks]
A_ev = [head_A(sq) for sq in eval_chunks]
intact = sum(loss(sq, None, False, False) for sq in eval_chunks) / len(eval_chunks)
print(f"intact held-out loss: {intact:.4f}")

def heal_eval(heads, mlp, label):
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR); model.train()
    for ep in range(EPOCHS):
        for sq, A in zip(train_chunks, A_tr):
            SEQ["s"] = sq; HOLD["A"] = A; HOLD["heads"] = heads; HOLD["mlp"] = mlp
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            HOLD["heads"] = HOLD["mlp"] = False
    model.eval()
    d = sum(loss(sq, A, heads, mlp) for sq, A in zip(eval_chunks, A_ev)) / len(eval_chunks) - intact
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    print(f"  {label:<18} healed damage {d:+.3f} nats", flush=True)
    gc.collect(); return d

print("\nHEALED decomposition (held-out):")
dh = heal_eval(True, False, "heads only (160)")
dm = heal_eval(False, True, "MLPs only (6)")
db = heal_eval(True, True, "both")
for hk_ in hooks: hk_.remove()
print(f"\nunhealed (headline chunk): heads 0.77, MLPs 0.48, both 1.61, interaction 0.36")
print(f"healed: heads {dh:+.3f}, MLPs {dm:+.3f}, both {db:+.3f}, "
      f"residual interaction {db-dh-dm:+.3f}")
print("read: if both << heads+MLPs, the heal mainly repairs the interaction.")
