"""Session-1 rigor: the POSITIONAL-ONLY baseline (PAPA-style control).

Same 160 heads + 6 MLPs as heal_combined, same healing recipe — but the head
templates are refit using ONLY input-independent columns {BOS, self,
offsets 1..16}. No match rule, no duplicate-token lookup, no sentence
structure: nothing that reads content.

  zero+heal:        natural +2.114   rnd 15.17   (function destroyed)
  posonly+heal:     THIS RUN
  full code+heal:   natural +0.681   rnd  7.47   (better than intact on rnd)

If full templates beat posonly, the content in the templates is load-bearing
beyond mere smooth-signal restoration.
"""
import sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K_HEADS, K_MLP, T_TR, LR = 160, 6, 600, 3e-4
POS_KEYS = ["bos", "self"] + [f"off{d}" for d in range(1, 17)]

nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:K_HEADS]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:K_MLP]
print(f"POSITIONAL-ONLY control: {K_HEADS} heads (cols {POS_KEYS[:3]}...off16) + MLPs {sorted(MLPS)}")

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
chunks += [mkrnd(), mkrnd()]
test_seq = RA.test_seq
torch.manual_seed(23)
rnd_eval = (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist())

# ---------- refit head weights on POSITIONAL columns only ----------
print("refitting heads on positional basis...")
train_fit = tokz.encode(raw[:11000])[:900]
base_fit = RR.code_attn(train_fit)
rows = slice(50, 900)
X = torch.stack([base_fit[k][rows].flatten() for k in POS_KEYS])
XXt = X @ X.T + 1e-4 * torch.eye(len(POS_KEYS))
with torch.no_grad():
    out = model(torch.tensor([train_fit]).to(DEV), output_attentions=True)
atts = [a[0].float().cpu() for a in out.attentions]
del out
W = {}
for (l, h) in HEADS:
    y = atts[l][h][rows].flatten()
    w = torch.linalg.solve(XXt, X @ y).clamp(min=0)
    W[(l, h)] = {k: float(wk) for k, wk in zip(POS_KEYS, w)}

# ---------- MLP lookup tables (same as heal_combined) ----------
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
    base = RR.code_attn(seq)          # rich base, but weights only touch POS_KEYS
    n = len(seq)
    out = {}
    for l, hs in BY_LAYER.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats).to(torch.float16)
    return out

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

print("precomputing code attention...")
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
print(f"posonly hybrid before healing: natural {b_nat:.3f}  rnd {b_rnd:.3f}")

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

for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()

print(f"\nPOSONLY+HEAL: natural healed {a_nat:.3f} (+{a_nat-3.232:.3f})  rnd healed {a_rnd:.3f}")
print(f"references: full code+heal +0.681 / rnd 7.474;  zero+heal +2.114 / rnd 15.17")
