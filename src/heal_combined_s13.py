"""Chapter 2, finale: the COMBINED hybrid.

160 attention heads (36%, rank-sum selection) -> code attention templates,
6 MLP layers (21%) -> token-lookup tables, installed simultaneously, then one
norm-only healing pass over both.

  usage: python3 heal_combined.py [EPOCHS]
"""
import sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP, NL = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL
tokz = RA.tokz
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
K_HEADS, K_MLP, T_TR, LR = 160, 6, 600, 3e-4

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:K_HEADS]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:K_MLP]
print(f"combined hybrid: {K_HEADS} heads + MLPs {sorted(MLPS)}")

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in starts]
torch.manual_seed(13)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
chunks += [mkrnd(), mkrnd()]
test_seq = RA.test_seq
torch.manual_seed(23)
rnd_eval = (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist())

# ---------- MLP lookup tables (fit with HEAD substitution NOT active: intact) ----------
print("fitting MLP lookup tables...")
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hooks = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq in chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in MLPS:
            o = cap[l]
            tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for hk in hooks: hk.remove()
MEAN = {l: tot[l] / cnt for l in MLPS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in MLPS}

def head_A(seq):
    base = RR.code_attn(seq)
    n = len(seq)
    out = {}
    for l, hs in BY_LAYER.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in RR.W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats).to(torch.float16)
    return out

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

print("precomputing code attention for all chunks...")
A_chunks = [head_A(sq) for sq in chunks]
L_chunks = [{l: lut_mat(sq, l) for l in MLPS} for sq in chunks]
A_nat, L_nat = head_A(test_seq), {l: lut_mat(test_seq, l) for l in MLPS}
A_rnd, L_rnd = head_A(rnd_eval), {l: lut_mat(rnd_eval, l) for l in MLPS}

HOLDER = {"A": None, "L": None}
vcache, hooks = {}, []
for l, hs in BY_LAYER.items():
    attn = model.model.layers[l].self_attn
    def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
    hooks.append(attn.v_proj.register_forward_hook(vhook))
    def ohook(mod, inp, l=l, hs=hs):
        x = inp[0].clone()
        A = HOLDER["A"][l].to(DEV).float()
        v = vcache[l]
        for mi, h in enumerate(hs):
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = A[mi] @ v[:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(mod, inp, outp, l=l):
        return HOLDER["L"][l].unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

def nll(seq, A, L):
    HOLDER["A"], HOLDER["L"] = A, L
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
b_nat, b_rnd = nll(test_seq, A_nat, L_nat), nll(rnd_eval, A_rnd, L_rnd)
print(f"combined hybrid before healing: natural {b_nat:.3f}  rnd {b_rnd:.3f}")

for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
best = (1e9, None)
for ep in range(EPOCHS):
    for sq, A, L in zip(chunks, A_chunks, L_chunks):
        HOLDER["A"], HOLDER["L"] = A, L
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        model.eval()
        nn_ = nll(test_seq, A_nat, L_nat)
        nr_ = nll(rnd_eval, A_rnd, L_rnd)
        if nn_ < best[0]: best = (nn_, [p.detach().cpu().clone() for p in norm_params])
        print(f"  epoch {ep+1}: natural {nn_:.3f}  rnd {nr_:.3f}", flush=True)
        model.train()
model.eval()
if best[1] is not None:
    for p, b in zip(norm_params, best[1]): p.data.copy_(b.to(DEV))
a_nat, a_rnd = nll(test_seq, A_nat, L_nat), nll(rnd_eval, A_rnd, L_rnd)
torch.save({"norms": [p.detach().cpu() for p in norm_params]}, "healed_combined.pt")

for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([test_seq]).to(DEV)).logits[0, :-1].float(), -1)
i_nat = -lp.gather(-1, torch.tensor(test_seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([rnd_eval]).to(DEV)).logits[0, :-1].float(), -1)
i_rnd = -lp.gather(-1, torch.tensor(rnd_eval[1:]).to(DEV).unsqueeze(-1)).mean().item()

print(f"\nCOMBINED ({K_HEADS} heads + {K_MLP} MLPs coded):")
print(f"  natural: intact {i_nat:.3f} | before {b_nat:.3f} (+{b_nat-i_nat:.3f}) | "
      f"healed {a_nat:.3f} (+{a_nat-i_nat:.3f}, recovered {(b_nat-a_nat)/(b_nat-i_nat):.0%})")
print(f"  rnd:     intact {i_rnd:.3f} | before {b_rnd:.3f} | healed {a_rnd:.3f}")
