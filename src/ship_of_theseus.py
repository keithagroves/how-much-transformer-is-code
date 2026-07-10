"""Ship of Theseus: does GRADUAL replace-heal-replace-heal beat one-shot?

One-shot installs all 160 heads + 6 MLPs, then heals the 65k norm gains once.
Gradual installs them in stages, carrying the healed norms forward, so each heal
only has to absorb a small new perturbation. Same total heal budget (4 stages x
5 epochs = 20, vs one-shot 20) so any gap is the curriculum effect, not compute.
If gradual reaches a lower final damage at k=160, progressive substitution is the
better protocol; if it ties, the norm-heal is expressive enough that order does
not matter.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
LR = 3e-4

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS_ALL = sorted(nat_c, key=lambda k: nr[k] + rr[k])
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
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = (hs, torch.stack(mats))
    return out
def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLD = {"A": None, "mlp": False}
vcache, hooks = {}, []
for l in sorted({l for l, _ in HEADS_ALL}):
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(m, inp, l=l):
        if HOLD["A"] is None or l not in HOLD["A"]: return None
        hs, A = HOLD["A"][l]; x = inp[0].clone(); A = A.to(DEV)
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
def loss(seq, A):
    SEQ["s"] = seq; HOLD["A"] = A; HOLD["mlp"] = True
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    HOLD["A"] = None; HOLD["mlp"] = False
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
HOLD["A"] = None
with torch.no_grad():
    intact = sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
                 .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
                 for sq in eval_chunks) / len(eval_chunks)
print(f"intact held-out loss: {intact:.4f}")

def heal(heads, epochs, opt):
    A_tr = [head_A(sq, heads) for sq in train_chunks]
    model.train()
    for ep in range(epochs):
        for sq, A in zip(train_chunks, A_tr):
            SEQ["s"] = sq; HOLD["A"] = A; HOLD["mlp"] = True
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            HOLD["A"] = None; HOLD["mlp"] = False
    model.eval()
    A_ev = [head_A(sq, heads) for sq in eval_chunks]
    return sum(loss(sq, A) for sq, A in zip(eval_chunks, A_ev)) / len(eval_chunks) - intact

# GRADUAL: carry norms forward across stages, 5 epochs each
print("\nGRADUAL (carry healed norms forward, 5 epochs/stage, MLPs present throughout):")
for p, o in zip(norm_params, orig): p.data.copy_(o)
for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
for k in [40, 80, 120, 160]:
    d = heal(HEADS_ALL[:k], 5, opt)
    print(f"  after stage k={k:>3}: healed damage {d:+.3f} nats", flush=True); gc.collect()
grad_final = d

# ONE-SHOT: fresh norms, 160 heads, 20 epochs (matched total budget)
print("\nONE-SHOT (fresh norms, 160 heads + MLPs, 20 epochs):")
for p, o in zip(norm_params, orig): p.data.copy_(o)
opt = torch.optim.Adam(norm_params, lr=LR)
one_shot = heal(HEADS_ALL[:160], 20, opt)
print(f"  one-shot healed damage: {one_shot:+.3f} nats", flush=True)
for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk_ in hooks: hk_.remove()
print(f"\ngradual {grad_final:+.3f} vs one-shot {one_shot:+.3f}  (delta {grad_final-one_shot:+.3f})")
print("read: gradual < one-shot => progressive replace-heal is the better protocol.")
