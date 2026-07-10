"""Greedy frontier: avoid detrimental replacements instead of taking a fixed
cheapest-k set. Walk heads in cheapest-solo order; accept a head's code only if
adding it raises held-out damage (unhealed, in the current context) by less than
EPS, else reject it. This adapts to interaction (solo cost mispredicts joint,
corr +0.14), so it should either fit MORE heads under a damage budget or the same
count at lower damage — turning the 'floor' into a measured frontier. Then heal
the accepted set once and compare to the fixed cheapest-160 (+0.70).
"""
import gc, sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
LR, EPOCHS = 3e-4, 20

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
ORDER = sorted(nat_c, key=lambda k: nr[k] + rr[k])          # cheapest-first
CODE160 = set(ORDER[:160])
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
sel_chunks = eval_chunks[:3]                                 # cheaper subset for the greedy pass

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
for l in sorted({l for l, _ in ORDER}):
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
def damage(heads, chunks):
    A = {sq_i: head_A(sq, heads) for sq_i, sq in enumerate(chunks)}
    tot = 0.0
    for i, sq in enumerate(chunks):
        SEQ["s"] = sq; HOLD["A"] = A[i]; HOLD["mlp"] = True
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
        HOLD["A"] = None; HOLD["mlp"] = False
        tot += -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
    return tot / len(chunks)

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
HOLD["A"] = None
intact = sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
             .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
             for sq in sel_chunks) / len(sel_chunks)
print(f"intact (selection subset): {intact:.4f};  EPS={EPS}")

# --- greedy forward pass over ALL heads in cheapest-solo order ---
accepted = []; d_cur = damage([], sel_chunks) - intact       # MLPs-only baseline
print(f"MLPs-only unhealed damage: {d_cur:+.3f}")
rejected_from_160 = 0; accepted_beyond_160 = 0
for j, h in enumerate(ORDER):
    d_new = damage(accepted + [h], sel_chunks) - intact
    if d_new - d_cur < EPS:
        accepted.append(h); d_cur = d_new
        if h not in CODE160: accepted_beyond_160 += 1
    else:
        if h in CODE160: rejected_from_160 += 1
    if (j + 1) % 80 == 0:
        print(f"  scanned {j+1}/{len(ORDER)}: accepted {len(accepted)}, unhealed dmg {d_cur:+.3f}", flush=True)
    gc.collect()
print(f"\ngreedy accepted {len(accepted)} heads (unhealed damage {d_cur:+.3f} on subset)")
print(f"  rejected {rejected_from_160} of the fixed cheapest-160; added {accepted_beyond_160} beyond it")

# --- heal the accepted set, full held-out eval ---
def heal_eval(heads):
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    for p in norm_params: p.requires_grad_(True)
    opt = torch.optim.Adam(norm_params, lr=LR)
    A_tr = [head_A(sq, heads) for sq in train_chunks]; model.train()
    for ep in range(EPOCHS):
        for sq, A in zip(train_chunks, A_tr):
            SEQ["s"] = sq; HOLD["A"] = A; HOLD["mlp"] = True
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
            HOLD["A"] = None; HOLD["mlp"] = False
    model.eval()
    d = damage(heads, eval_chunks) - (sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0,:-1].float(),-1)
            .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item() for sq in eval_chunks)/len(eval_chunks))
    for p, o in zip(norm_params, orig): p.data.copy_(o)
    return d

g = heal_eval(accepted)
c = heal_eval(ORDER[:160])
for hk_ in hooks: hk_.remove()
print(f"\nHEALED (full held-out):")
print(f"  greedy frontier ({len(accepted)} heads): {g:+.3f} nats")
print(f"  fixed cheapest-160:                 {c:+.3f} nats")
print("read: greedy fits more heads and/or lower damage => the 36% floor understates the frontier.")
