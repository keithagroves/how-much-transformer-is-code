"""Chapter 2, step 4: where does the veto live?

The 27-head circuit boosts the copy-candidate when copying is right (+0.19)
and suppresses it when copying is wrong (-0.41). Ablate each head ALONE and
measure its effect on the copy-candidate's log-prob in each bucket:

  boost score  = mean damage in A (copy right)   -- positive = head pushes copy
  veto score   = mean damage in B (copy wrong)   -- negative = head suppresses copy

Dual-role heads: +A and -B. Dedicated inhibitors: ~0 in A, strongly -B.
"""
import torch, natural_rule as NR

model, ids, seq, T, DH = NR.model, NR.ids, NR.seq, NR.T, NR.DH

# bucket positions and their copy-candidate token
last = {}
A, B = [], []          # (position, copy_candidate)
for i in range(1, T - 1):
    cur, nxt = seq[i], seq[i + 1]
    if cur in last:
        pred = seq[last[cur] + 1]
        (A if pred == nxt else B).append((i, pred))
    last[seq[i - 1]] = i - 1
pa = torch.tensor([p for p, _ in A]); ca = torch.tensor([c for _, c in A])
pb = torch.tensor([p for p, _ in B]); cb = torch.tensor([c for _, c in B])

def copy_lp(off):
    hooks = []
    for l, h in off:
        def hook(mod, inp, h=h):
            x = inp[0].clone(); x[..., h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return lp[pa, ca], lp[pb, cb]

base_a, base_b = copy_lp([])
print(f"{'head':<10}{'ind':>6}{'boost (dmg in A)':>18}{'veto (dmg in B)':>17}")
rows = []
for l, h in NR.INDUCTION:
    a, b = copy_lp([(l, h)])
    rows.append((l, h, float(NR.s['ind'][l, h]),
                 (base_a - a).mean().item(), (base_b - b).mean().item()))
for l, h, s, da, db in sorted(rows, key=lambda r: r[4]):
    tag = "  <- inhibitor?" if db < -0.02 and da < 0.02 else (
          "  <- dual-role" if db < -0.02 else "")
    print(f"L{l}.H{h:<6}{s:>6.2f}{da:>18.3f}{db:>17.3f}{tag}")
torch.save(rows, "veto_rows.pt")
