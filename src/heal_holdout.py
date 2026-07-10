"""Pooled healed damage on GENUINELY held-out text.

The paper's +0.68 was measured on test_seq, which healing early-stopped on
(heal_combined.py selects norms by test_seq loss); heal_eval.py's pooled chunks
(700/780/860/940k) were all in the healing+LUT training set (starts 20000+40000k).
Both are in-sample.

Here: reuse the FIXED surrogate (healed_combined.pt norms + LUT built from the
same 24 training chunks) but evaluate on offsets that are clean multiples of
40000 -- disjoint from the training starts (which are all 20000 mod 40000) and
never used for early stopping. This turns the headline from indicative to
measured. Reports per-chunk and pooled intact/healed damage with a chunk bootstrap.
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:160]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

raw = open("ministral_corpus.txt").read()
# LUT training chunks: EXACTLY heal_combined's 24 (starts = 20000 + 40000k)
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_set = set(train_starts)
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in train_starts]

# held-out eval offsets: clean multiples of 40000 (0 mod 40000), disjoint from
# the training starts (all 20000 mod 40000); skip 185-215k; skip any collisions.
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in train_set and not 185000 <= o <= 215000][:8]
assert not (set(eval_starts) & train_set), "eval overlaps training!"
print(f"held-out eval offsets (k): {[o//1000 for o in eval_starts]}")

SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
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

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

def evl(seq):
    A = head_A(seq); vcache, hooks = {}, []
    for l, hs in BY_LAYER.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone(); Al = A[l].to(DEV)
            for mi, h in enumerate(hs):
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = Al[mi] @ vcache[l][:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    for l in MLPS:
        Ll = lut_mat(seq, l)
        def mhook(mod, inp, outp, l=l, Ll=Ll): return Ll.unsqueeze(0)
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for h in hooks: h.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def intact(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

# apply the FIXED healed norms
norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
healed = torch.load("healed_combined.pt")["norms"]
assert len(healed) == len(norm_params)
for p, b in zip(norm_params, healed): p.data.copy_(b.to(DEV))

print(f"{'eval offset':>12}{'intact':>8}{'healed':>8}{'dHealed':>9}")
dmg = []
for o in eval_starts:
    seq = tokz.encode(raw[o:o + 8000])[:300]
    it = intact(seq); hl = evl(seq); dmg.append(hl - it)
    print(f"{'chunk@'+str(o//1000)+'k':>12}{it:>8.3f}{hl:>8.3f}{hl-it:>+9.3f}", flush=True)
    gc.collect()

import random
random.seed(0); B = 5000; nchk = len(dmg)
bs = sorted(sum(dmg[random.randrange(nchk)] for _ in range(nchk)) / nchk for _ in range(B))
pooled = sum(dmg) / nchk
print(f"\nPOOLED held-out healed damage: {pooled:+.3f} nats  "
      f"95% CI [{bs[int(.025*B)]:+.3f}, {bs[int(.975*B)]:+.3f}]  (n={nchk} chunks)")
print("This is out-of-sample: none of these offsets trained the LUT, healed the "
      "norms, or early-stopped the heal.")
