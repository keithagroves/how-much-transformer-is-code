"""Lean pre-vs-post frontier comparison: does healing change WHICH replacements
are detrimental? The expensive part before was re-healing at every threshold
point; that is unnecessary. The comparison needs only two leave-one-out rankings:
  pre  = LOO marginal harm on the UNHEALED superset (no heal)
  post = LOO marginal harm after ONE capped 10-epoch heal
Report: harmful-set overlap, Spearman rank correlation, and each ranking's
additive frontier prediction (full - sum of dropped heads' harm, free). One heal
total, ~5-8 min. If pre ~ post, the cheap unhealed ranking is a trustworthy proxy;
if not, healing rescues specific heads and only post-heal is valid.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
LR, EPOCHS, SUPER = 3e-4, 10, 200          # capped 10-epoch heal

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
        out[l] = [(h, (lambda M: M / M.sum(-1, keepdim=True).clamp(min=1e-9))(
            sum((wk * base[k] for k, wk in RR.W[(l, h)].items() if wk > 1e-4),
                torch.zeros(n, n)))) for h in hs]
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
def loss_set(A, skip):
    HOLD["skip"] = skip; t = 0.0
    for i, sq in enumerate(eval_chunks):
        SEQ["s"] = sq; HOLD["A"] = A[i]; HOLD["mlp"] = True
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
        HOLD["A"] = None; HOLD["mlp"] = False
        t += -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()
    HOLD["skip"] = set(); return t / len(eval_chunks)

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
SUP = ORDER[:SUPER]
A_ev = [head_A(sq, SUP) for sq in eval_chunks]

HOLD["A"] = None
intact = sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0,:-1].float(),-1)
    .gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item() for sq in eval_chunks)/len(eval_chunks)

def scan(tag):
    full = loss_set(A_ev, set()) - intact
    marg = {}
    for hd in SUP:
        marg[hd] = full - (loss_set(A_ev, {hd}) - intact)   # harm: >0 = code adds damage
    print(f"{tag}: full {full:+.3f}, net-harmful {sum(1 for v in marg.values() if v>1e-4)}/{SUPER}", flush=True)
    return full, marg

# PRE: unhealed scan
pre_full, pre = scan("PRE (unhealed)")
# one capped heal, then POST scan
for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
A_tr = [head_A(sq, SUP) for sq in train_chunks]; model.train()
for ep in range(EPOCHS):
    for sq, A in zip(train_chunks, A_tr):
        SEQ["s"] = sq; HOLD["A"] = A; HOLD["mlp"] = True; HOLD["skip"] = set()
        ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
        HOLD["A"] = None; HOLD["mlp"] = False
model.eval()
post_full, post = scan("POST (10-epoch heal)")
for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk_ in hooks: hk_.remove()

# --- comparison ---
def ranks(d):
    order = sorted(d, key=lambda k: d[k]); return {k: i for i, k in enumerate(order)}
rp, rq = ranks(pre), ranks(post)
xs = [rp[h] for h in SUP]; ys = [rq[h] for h in SUP]
mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
sp = cov / ((sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5)
harm_pre = set(h for h, v in pre.items() if v > 1e-4)
harm_post = set(h for h, v in post.items() if v > 1e-4)
jac = len(harm_pre & harm_post) / max(1, len(harm_pre | harm_post))
# top-40 most-harmful agreement
top_pre = set(sorted(SUP, key=lambda h: -pre[h])[:40])
top_post = set(sorted(SUP, key=lambda h: -post[h])[:40])
print(f"\nPRE vs POST ranking comparison ({SUPER} heads):")
print(f"  Spearman rank corr of marginals: {sp:+.3f}")
print(f"  net-harmful set Jaccard overlap: {jac:.2f}  (pre {len(harm_pre)}, post {len(harm_post)})")
print(f"  top-40 most-harmful agreement:   {len(top_pre & top_post)}/40")
# additive frontier prediction (free), each ranking
def additive(full, marg, k):
    drop = sorted(SUP, key=lambda h: -marg[h])[:SUPER-k]     # drop the most-harmful
    return full - sum(marg[h] for h in drop)
print(f"\nadditive frontier (no re-heal), keep k least-harmful:")
print(f"{'k':>6}{'pre-pred':>10}{'post-pred':>11}")
for k in [47, 120, 160]:
    print(f"{k:>6}{additive(pre_full, pre, k):>+10.3f}{additive(post_full, post, k):>+11.3f}")
print("\nread: high Spearman + high overlap => pre-heal ranking is a cheap proxy for post-heal;")
print("      low => healing rescues specific heads and only the post-heal criterion is valid.")
