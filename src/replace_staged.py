"""Chapter 2, step 8c: STAGED replacement with refitting.

One-shot replacement compounds interaction error: every template was fitted
against the intact model, but after 100 substitutions the survivors see
different inputs. Fix: replace in batches; after each batch, re-capture the
hybrid model's attentions on the train seq and fit the NEXT batch against
the model as it now is.
"""
import torch
import replace_all as RA
import replace_all2 as R2

model, DEV, DH, GROUP, NL, NH, T = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL, RA.NH, RA.T
train_seq, test_seq, rnd_test = RA.train_seq, RA.test_seq, RA.rnd_test
BATCH = 32

base_tr = RA.code_attn(train_seq)
base_te = RA.code_attn(test_seq)
base_rn = RA.code_attn(rnd_test)
names = list(base_tr.keys())
rows = slice(50, T)
X = torch.stack([base_tr[k][rows].flatten() for k in names])
XXt = X @ X.T + 1e-4 * torch.eye(len(names))

costs = torch.load("solo_costs.pt")
order = sorted(costs, key=lambda k: costs[k])

W = dict(torch.load("head_templates.pt"))     # start from intact-model fits
R2.W2 = W                                     # R2.run_sub reads this

def capture_hybrid(heads_done):
    """attentions of ALL heads while heads_done are substituted (train seq)."""
    n = len(train_seq)
    A = {}
    for (l, h) in heads_done:
        M = torch.zeros(n, n)
        for k, wk in W[(l, h)].items():
            if wk > 1e-4: M += wk * base_tr[k]
        A[(l, h)] = (M / M.sum(-1, keepdim=True).clamp(min=1e-9)).to(DEV)
    by_layer = {}
    for l, h in heads_done: by_layer.setdefault(l, []).append(h)
    vcache, hooks = {}, []
    for l in range(NL):
        attn = model.model.layers[l].self_attn
        if l in by_layer:
            def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
            hooks.append(attn.v_proj.register_forward_hook(vhook))
            hs = by_layer[l]
            def ohook(mod, inp, l=l, hs=hs):
                x = inp[0].clone()
                for h in hs:
                    g = h // GROUP
                    x[0, :, h*DH:(h+1)*DH] = A[(l, h)] @ vcache[l][:, g*DH:(g+1)*DH]
                return (x,) + inp[1:]
            hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        out = model(torch.tensor([train_seq]).to(DEV), output_attentions=True)
    for hk in hooks: hk.remove()
    return [a[0].float().cpu() for a in out.attentions]

ni_nat = R2.run_sub(test_seq, [], base_te)
ni_rnd = R2.run_sub(rnd_test, [], base_rn)
print(f"held-out intact: natural {ni_nat:.3f}  rnd {ni_rnd:.3f}")
print(f"{'k':>5}{'natural':>9}{'+nats':>8}{'rnd':>9}{'+nats':>8}")

done = []
for stage in range(12):
    nxt = [h for h in order if h not in done][:BATCH]
    if not nxt: break
    if stage > 0:                                  # refit next batch vs hybrid
        atts = capture_hybrid(done)
        for (l, h) in nxt:
            y = atts[l][h][rows].flatten()
            w = torch.linalg.solve(XXt, X @ y).clamp(min=0)
            W[(l, h)] = {k: float(wk) for k, wk in zip(names, w)}
    done += nxt
    nc = R2.run_sub(test_seq, done, base_te)
    nr = R2.run_sub(rnd_test, done, base_rn)
    print(f"{len(done):>5}{nc:>9.3f}{nc-ni_nat:>+8.3f}{nr:>9.3f}{nr-ni_rnd:>+8.3f}", flush=True)

torch.save({"W": W, "done": done}, "staged_templates.pt")
