"""The control that decides what the templates are worth.

Identical to heal_combined.py -- same 160 heads, same 6 MLPs, same healing
recipe (norm-only, 24 natural + 2 rnd chunks, best checkpoint) -- except the
replaced components emit ZEROS instead of code-template output.

  code+heal (measured): natural +0.681, rnd better than intact
  zero+heal (this run): if it lands near +0.68, healing does the work and
  the templates are decoration; if it stays far worse, the code content is
  load-bearing.
"""
import sys, torch
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
K_HEADS, K_MLP, T_TR, LR = 160, 6, 600, 3e-4

nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nat_rank = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rnd_rank = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS = sorted(nat_c, key=lambda k: nat_rank[k] + rnd_rank[k])[:K_HEADS]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:K_MLP]
print(f"ZERO control: {K_HEADS} heads + MLPs {sorted(MLPS)} -> zeros, then heal")

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
chunks += [mkrnd(), mkrnd()]
test_seq = RA.test_seq
torch.manual_seed(23)
rnd_eval = (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist())

hooks = []
for l, hs in BY_LAYER.items():
    def ohook(mod, inp, hs=hs):
        x = inp[0].clone()
        for h in hs: x[..., h*DH:(h+1)*DH] = 0
        return (x,) + inp[1:]
    hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(ohook))
for l in MLPS:
    def mhook(mod, inp, outp):
        return torch.zeros_like(outp)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

def nll(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
b_nat, b_rnd = nll(test_seq), nll(rnd_eval)
print(f"zero hybrid before healing: natural {b_nat:.3f}  rnd {b_rnd:.3f}")

for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
best = (1e9, None)
for ep in range(EPOCHS):
    for sq in chunks:
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        model.eval()
        nn_, nr_ = nll(test_seq), nll(rnd_eval)
        if nn_ < best[0]: best = (nn_, [p.detach().cpu().clone() for p in norm_params])
        print(f"  epoch {ep+1}: natural {nn_:.3f}  rnd {nr_:.3f}", flush=True)
        model.train()
model.eval()
if best[1] is not None:
    for p, b in zip(norm_params, best[1]): p.data.copy_(b.to(DEV))
a_nat, a_rnd = nll(test_seq), nll(rnd_eval)

for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([test_seq]).to(DEV)).logits[0, :-1].float(), -1)
i_nat = -lp.gather(-1, torch.tensor(test_seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([rnd_eval]).to(DEV)).logits[0, :-1].float(), -1)
i_rnd = -lp.gather(-1, torch.tensor(rnd_eval[1:]).to(DEV).unsqueeze(-1)).mean().item()

print(f"\nZERO+HEAL control ({K_HEADS} heads + {K_MLP} MLPs zeroed):")
print(f"  natural: intact {i_nat:.3f} | before {b_nat:.3f} (+{b_nat-i_nat:.3f}) | "
      f"healed {a_nat:.3f} (+{a_nat-i_nat:.3f})")
print(f"  rnd:     intact {i_rnd:.3f} | before {b_rnd:.3f} | healed {a_rnd:.3f}")
print(f"\n  code+heal reference: natural +0.681, rnd 7.474 (better than intact 7.979)")
