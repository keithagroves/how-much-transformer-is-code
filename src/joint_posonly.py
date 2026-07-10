"""Reviewer #2: does "the cheap majority is positional" survive when the head set
is chosen JOINTLY (the swapped, lower-damage set) instead of by solo cost?

On the joint-improved 160-set (cheapest-solo with m worst-keepers swapped for m
best-adders, from select_strategy.pt), heal twice with the SAME recipe: full code
templates vs POSITIONAL-ONLY templates ({BOS, self, offsets 1..16}, no content).
If positional-only still ~ties full code after healing, the "mostly positional"
reading holds for the better set too; if full code now wins clearly, the joint
search pulled in content-bearing (induction) heads and the reading weakens.
Healed damage reported on test_seq (held out from the healing chunks), matching
the paper's cheapest-solo numbers: full +0.68, positional +0.66.
    usage: python3 joint_posonly.py [M_SWAP] [EPOCHS]
"""
import sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP, NL = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL
tokz = RA.tokz
M_SWAP = int(sys.argv[1]) if len(sys.argv) > 1 else 80
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
LR = 3e-4
POS_KEYS = ["bos", "self"] + [f"off{d}" for d in range(1, 17)]

RR.W.update(torch.load("rich_templates.pt"))
FULL_W = dict(RR.W)
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
SOLO = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:160]
S = torch.load("select_strategy.pt"); rem, add = S["rem"], S["add"]
drop = set(sorted(SOLO, key=lambda h: rem[h], reverse=True)[:M_SWAP])
bring = sorted(add, key=lambda h: add[h])[:M_SWAP]
JOINT = [h for h in SOLO if h not in drop] + bring
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]
print(f"JOINT set: {len(SOLO)} cheapest-solo with {M_SWAP} swapped in/out -> {len(JOINT)} heads")

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
chunks += [mkrnd(), mkrnd()]
test_seq = RA.test_seq

# positional-only templates for the joint heads (refit on positional columns)
print("refitting joint heads on positional basis...")
train_fit = tokz.encode(raw[:11000])[:900]
base_fit = RR.code_attn(train_fit)
rows = slice(50, 900)
Xp = torch.stack([base_fit[k][rows].flatten() for k in POS_KEYS])
XXt = Xp @ Xp.T + 1e-4 * torch.eye(len(POS_KEYS))
with torch.no_grad():
    out = model(torch.tensor([train_fit]).to(DEV), output_attentions=True)
atts = [a[0].float().cpu() for a in out.attentions]; del out
POS_W = {}
for (l, h) in JOINT:
    y = atts[l][h][rows].flatten()
    w = torch.linalg.solve(XXt, Xp @ y).clamp(min=0)
    POS_W[(l, h)] = {k: float(wk) for k, wk in zip(POS_KEYS, w)}

# MLP lookup tables
print("fitting MLP lookup tables...")
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq in chunks:
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
BY = {}
for l, h in JOINT: BY.setdefault(l, []).append(h)

def head_A(seq, Wd):
    base = RR.code_attn(seq); n = len(seq); out = {}
    for l, hs in BY.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in Wd[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats)
    return out

def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]

def intact(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
i_nat = intact(test_seq)

def heal(Wd, tag):
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    A_tr = [head_A(sq, Wd) for sq in chunks]
    L_tr = [{l: lut_mat(sq, l) for l in MLPS} for sq in chunks]
    A_te = head_A(test_seq, Wd); L_te = {l: lut_mat(test_seq, l) for l in MLPS}
    HOLD = {"A": None, "L": None}
    vcache, hooks = {}, []
    for l, hs in BY.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone(); A = HOLD["A"][l].to(DEV).float(); v = vcache[l]
            for mi, h in enumerate(hs):
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A[mi] @ v[:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    for l in MLPS:
        def mhook(mod, inp, outp, l=l): return HOLD["L"][l].unsqueeze(0)
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

    def nll(seq, A, L):
        HOLD["A"], HOLD["L"] = A, L
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
        return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

    before = nll(test_seq, A_te, L_te)
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR); model.train()
    best = 1e9
    for ep in range(EPOCHS):
        for sq, A, L in zip(chunks, A_tr, L_tr):
            HOLD["A"], HOLD["L"] = A, L
            ids = torch.tensor([sq]).to(DEV)
            out = model(ids, labels=ids); opt.zero_grad(); out.loss.backward(); opt.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            model.eval(); best = min(best, nll(test_seq, A_te, L_te)); model.train()
    model.eval()
    for p in norm_params: p.requires_grad_(False)
    for h in hooks: h.remove()
    print(f"  {tag}: healed {best:.3f}  (+{best - i_nat:.3f} vs intact {i_nat:.3f}; "
          f"before-heal +{before - i_nat:.3f})", flush=True)
    return best - i_nat

print(f"\nhealing joint set (test_seq; cheapest-solo reference: full +0.68 / pos +0.66):")
df = heal(FULL_W, "full code   ")
dp = heal(POS_W, "positional-only")
for p, o in zip(norm_params, orig): p.data.copy_(o)
print(f"\nJOINT set healed damage: full +{df:.3f} | positional-only +{dp:.3f} | "
      f"gap {dp - df:+.3f}")
print("gap ~0 => 'mostly positional' survives joint selection; gap>>0 => joint set is"
      " more content-bearing.")
