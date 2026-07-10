"""Verify the swap gain by MEASUREMENT, not the additive upper bound. Build sets
that swap the m most-expensive-to-keep coded heads for the m cheapest-to-add
candidates (from select_strategy.pt) and measure true unhealed joint damage at
K=160. If damage drops materially, cheapest-solo-first genuinely leaves a better
same-budget set on the table (interactions included).
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
RR.W.update(torch.load("rich_templates.pt"))
S = torch.load("select_strategy.pt"); rem, add = S["rem"], S["add"]
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
ALL = list(nat_c.keys())
SET = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:160]

keep_worst = sorted(SET, key=lambda h: rem[h], reverse=True)     # drop these first
add_best = sorted(add, key=lambda h: add[h])                     # bring these in first

def swapped(m):
    drop = set(keep_worst[:m]); bring = add_best[:m]
    return [h for h in SET if h not in drop] + bring

raw = open("ministral_corpus.txt").read()
N = min(len(raw) - 9000, 1_000_000)
EVAL = [tokz.encode(raw[o:o + 8000])[:300]
        for o in (700_000, 780_000, 860_000, 940_000) if o < N]

def head_mat(base, l, h, n):
    M = torch.zeros(n, n)
    for k, wk in RR.W[(l, h)].items():
        if wk > 1e-4: M += wk * base[k]
    return M / M.sum(-1, keepdim=True).clamp(min=1e-9)

def eval_set(seq, Abh, subset):
    by_layer = {}
    for l, h in subset: by_layer.setdefault(l, []).append(h)
    vcache, hooks = {}, []
    for l, hs in by_layer.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                x[0, :, h * DH:(h + 1) * DH] = Abh[(l, h)] @ vcache[l][:, g * DH:(g + 1) * DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

MS = [0, 20, 47, 80]
dmg = {m: 0.0 for m in MS}
for seq in EVAL:
    n = len(seq); base = RR.code_attn(seq)
    Abh = {(l, h): head_mat(base, l, h, n).to(DEV) for (l, h) in ALL}
    ni = eval_set(seq, Abh, [])
    for m in MS:
        dmg[m] += eval_set(seq, Abh, swapped(m)) - ni
    del Abh; gc.collect()
for m in MS: dmg[m] /= len(EVAL)

print(f"\ntrue unhealed joint damage at K=160, swapping m worst-keepers for m best-adders:")
print(f"{'m swaps':>8}{'damage':>10}{'vs m=0':>10}")
for m in MS:
    print(f"{m:>8}{dmg[m]:>+10.3f}{dmg[m]-dmg[0]:>+10.3f}", flush=True)
print(f"\ncheapest-solo (m=0) damage {dmg[0]:+.3f}; best measured {min(dmg.values()):+.3f}"
      f" at m={min(dmg, key=dmg.get)} -> a same-budget set with"
      f" {dmg[0]-min(dmg.values()):.3f} nats less damage exists.")
