"""Measure per-head solo substitution cost (rich templates) on a repeated-
random sequence, so head selection can respect BOTH regimes."""
import torch
import replace_rich as RR
import replace_all as RA

RR.W.update(torch.load("rich_templates.pt"))
import os as _os
torch.manual_seed(int(_os.environ.get("SUB_SEED_RND", "7")))
seq = (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 4  # 400 toks
base = RR.code_attn(seq)
ni = RR.run_sub(seq, [], base)
print(f"rnd train intact {ni:.4f}")
costs = {}
for l in range(RA.NL):
    for h in range(RA.NH):
        costs[(l, h)] = RR.run_sub(seq, [(l, h)], base) - ni
    if l % 7 == 6: print(f"  through L{l}", flush=True)
torch.save(costs, "rich_solo_rnd.pt")
worst = sorted(costs.items(), key=lambda kv: -kv[1])[:8]
print("worst heads on rnd:", [(f"L{l}.H{h}", round(c, 3)) for (l, h), c in worst])
