"""Selection-strategy sensitivity (reviewer follow-up): joint damage is not the
sum of solo costs, so is cheapest-solo-first the right way to pick the 160 coded
heads? True greedy-joint selection is ~59k forwards (infeasible); its decisive
core is tractable -- measure IN-CONTEXT marginal cost at the K=160 boundary:

  remove(h in SET)   loss(SET) - loss(SET\{h})     how much h adds given the rest
  add(h not in SET)  loss(SET+{h}) - loss(SET)     cost to bring h in

Then: (1) does solo cost predict in-context marginal cost? (2) are there swaps --
an un-coded head cheaper to add than a coded head is to keep -- i.e. would greedy
find a lower-damage set at the SAME budget? Many/large swaps => the 36% figure is
a heuristic artifact; few/tiny => cheapest-solo is near-optimal. Unhealed loss
(the selection question is separate from healing).
"""
import gc, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP, NL, NH = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL, RA.NH
tokz = RA.tokz
RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
ALL = list(nat_c.keys())
SET = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:160]   # the paper's 160
SETS = set(SET)
OUT = [h for h in ALL if h not in SETS]
print(f"{len(SET)} coded heads (rank-sum), {len(OUT)} candidates to add")

raw = open("ministral_corpus.txt").read()
N = min(len(raw) - 9000, 1_000_000)
EVAL = [tokz.encode(raw[o:o + 8000])[:300]
        for o in (700_000, 780_000, 860_000, 940_000) if o < N]
print(f"{len(EVAL)} held-out eval chunks")


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


rem = {h: 0.0 for h in SET}
add = {h: 0.0 for h in OUT}
D0 = 0.0
for ci, seq in enumerate(EVAL):
    n = len(seq)
    base = RR.code_attn(seq)
    Abh = {(l, h): head_mat(base, l, h, n).to(DEV) for (l, h) in ALL}
    lset = eval_set(seq, Abh, SET)
    ni = eval_set(seq, Abh, [])
    D0 += (lset - ni)
    for h in SET:
        rem[h] += eval_set(seq, Abh, [x for x in SET if x != h]) - lset  # >0: h adds damage
    for h in OUT:
        add[h] += eval_set(seq, Abh, SET + [h]) - lset                  # cost to add h
    del Abh; gc.collect()
    print(f"  chunk {ci}: joint damage {lset - ni:+.3f} nats", flush=True)

nc = len(EVAL)
D0 /= nc
for h in SET: rem[h] /= nc
for h in OUT: add[h] /= nc

# correlation: solo cost vs in-context marginal (remove) for coded heads
xs = torch.tensor([nat_c[h] for h in SET]); ys = torch.tensor([rem[h] for h in SET])
xc, yc = xs - xs.mean(), ys - ys.mean()
corr = float((xc @ yc) / (xc.norm() * yc.norm() + 1e-9))

rem_sorted = sorted(SET, key=lambda h: rem[h])              # cheapest-to-keep first
add_sorted = sorted(OUT, key=lambda h: add[h])             # cheapest-to-add first
# beneficial swaps: pair most-expensive coded head with cheapest un-coded candidate
gain, swaps = 0.0, 0
i, j = len(rem_sorted) - 1, 0
while i >= 0 and j < len(add_sorted):
    r, a = rem[rem_sorted[i]], add[add_sorted[j]]
    if a < r - 1e-4:                                       # add cheaper than keep -> swap wins
        gain += (r - a); swaps += 1; i -= 1; j += 1
    else:
        break

print(f"\njoint damage of the paper's 160 (unhealed): {D0:+.3f} nats")
print(f"corr(solo cost, in-context marginal-remove) over 160 heads: {corr:+.2f}")
print(f"coded heads with marginal-remove <= 0 (free/helpful in context): "
      f"{sum(r <= 0 for r in rem.values())}/160")
print(f"most-expensive coded head marginal-remove: {max(rem.values()):+.3f}")
print(f"cheapest candidate marginal-add: {min(add.values()):+.3f}")
print(f"\nbeneficial single-swaps available at K=160: {swaps}")
print(f"estimated damage reduction from swapping (greedy upper bound): {gain:.3f} nats")
print(f"  => cheapest-solo set is {'FAR FROM' if gain > 0.1 else 'NEAR'} optimal at this budget")
torch.save({"rem": rem, "add": add, "D0": D0, "corr": corr, "gain": gain, "swaps": swaps},
           "select_strategy.pt")
