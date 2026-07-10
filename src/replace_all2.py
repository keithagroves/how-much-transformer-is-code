"""Chapter 2, step 8b: better fits, better selection, honest frontier.

Upgrades over replace_all.py:
  1. per-head template weights by least squares against the head's true
     attention rows (train seq), instead of exclusive mass attribution
  2. head selection by MEASURED solo substitution cost on the TRAIN sequence
     (one forward per head, 448 total), not by explained mass
  3. greedy frontier: substitute the k cheapest heads cumulatively,
     evaluated on the HELD-OUT natural sequence + repeated random
"""
import torch
import replace_all as RA

model, DEV, DH, GROUP, NL, NH = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL, RA.NH
train_seq, test_seq, rnd_test = RA.train_seq, RA.test_seq, RA.rnd_test
T = RA.T

# ---------- least-squares template weights on train ----------
print("refitting per-head weights by least squares...")
with torch.no_grad():
    out = model(torch.tensor([train_seq]).to(DEV), output_attentions=True)
atts = [a[0].float().cpu() for a in out.attentions]
del out
base_tr = RA.code_attn(train_seq)
names = list(base_tr.keys())
rows = slice(50, T)
X = torch.stack([base_tr[k][rows].flatten() for k in names])       # [K, N]
XXt = X @ X.T + 1e-4 * torch.eye(len(names))
W2 = {}
for l in range(NL):
    for h in range(NH):
        y = atts[l][h][rows].flatten()
        w = torch.linalg.solve(XXt, X @ y).clamp(min=0)
        W2[(l, h)] = {k: float(wk) for k, wk in zip(names, w)}
torch.save(W2, "head_templates.pt")

def run_sub(seq, heads, base):
    n = len(seq)
    A = {}
    for (l, h) in heads:
        M = torch.zeros(n, n)
        for k, wk in W2[(l, h)].items():
            if wk > 1e-4: M += wk * base[k]
        A[(l, h)] = (M / M.sum(-1, keepdim=True).clamp(min=1e-9)).to(DEV)
    by_layer = {}
    for l, h in heads: by_layer.setdefault(l, []).append(h)
    vcache, hooks = {}, []
    for l, hs in by_layer.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A[(l, h)] @ vcache[l][:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    base_train = RA.code_attn(train_seq)
    ni_tr = run_sub(train_seq, [], base_train)
    print(f"train intact {ni_tr:.4f}; measuring solo cost for all {NL*NH} heads...")
    costs = {}
    for l in range(NL):
        for h in range(NH):
            costs[(l, h)] = run_sub(train_seq, [(l, h)], base_train) - ni_tr
        if l % 4 == 3:
            done = sorted(costs.values())
            print(f"  through L{l}: cheapest {done[0]:+.4f}, median {done[len(done)//2]:+.4f}")
    torch.save(costs, "solo_costs.pt")

    order = sorted(costs, key=lambda k: costs[k])
    base_test = RA.code_attn(test_seq)
    base_rnd = RA.code_attn(rnd_test)
    ni_nat = run_sub(test_seq, [], base_test)
    ni_rnd = run_sub(rnd_test, [], base_rnd)
    print(f"\nheld-out intact: natural {ni_nat:.3f}  repeated-random {ni_rnd:.3f}")
    print(f"{'k heads':>8}{'natural':>9}{'+nats':>8}{'rnd':>9}{'+nats':>8}")
    for k in (32, 64, 96, 128, 160, 224, 288, 352):
        heads = order[:k]
        nc = run_sub(test_seq, heads, base_test)
        nr = run_sub(rnd_test, heads, base_rnd)
        print(f"{k:>8}{nc:>9.3f}{nc-ni_nat:>+8.3f}{nr:>9.3f}{nr-ni_rnd:>+8.3f}")
