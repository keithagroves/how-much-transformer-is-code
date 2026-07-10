"""Reviewer control: is code better than same-budget OPTIMIZED PRUNING + heal?

The zero-ablation control deletes the code-selected set (chosen for codability).
The fairer alternative baseline deletes a same-budget set chosen for PRUNABILITY
(lowest ablation cost), then heals. If that heals to ~the code cost, the code's
advantage over 'just delete redundant components and heal' is small; if it heals
to ~the zero-ablation level, code genuinely beats pruning.

Budget = 160 heads + 6 MLPs (same as the hybrid). Prune set = the 160 heads with
lowest solo ablation cost and the 6 lowest-cost MLPs (standard magnitude pruning).
Zero those, heal only the 65k norm gains under the matched fresh-heal protocol,
eval held-out vs the intact model. Compare to code+heal (+0.70) and zero-of-code-
set+heal (+2.12). Caveat: solo cost is a proxy for marginal/joint prunability
(the paper's solo/joint corr is only +0.14), so this is a strong-but-not-optimal
prune baseline.
"""
import torch
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
EPOCHS, T_TR, LR = 20, 600, 3e-4

# --- select the PRUNE set: lowest solo ablation cost (not codability) ---
nat_c = torch.load("rich_solo_costs.pt")                       # head -> natural solo cost
PRUNE_HEADS = sorted(nat_c, key=lambda k: nat_c[k])[:160]      # cheapest-to-delete
BY_LAYER = {}
for l, h in PRUNE_HEADS: BY_LAYER.setdefault(l, []).append(h)
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
PRUNE_MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]
# overlap with the code set (rank-sum of nat+rnd solo), for context
rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
CODE_HEADS = set(sorted(nat_c, key=lambda k: nr[k] + rr[k])[:160])
overlap = len(set(PRUNE_HEADS) & CODE_HEADS)
print(f"prune set: 160 heads + MLPs {sorted(PRUNE_MLPS)}; overlap with code set: {overlap}/160")

raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]

# --- hooks that ZERO the prune-set heads and MLPs (active flag) ---
ACTIVE = {"on": False}
hooks = []
for l, hs in BY_LAYER.items():
    def ohook(mod, inp, l=l, hs=hs):
        if not ACTIVE["on"]: return None
        x = inp[0].clone()
        for h in hs: x[0, :, h*DH:(h+1)*DH] = 0
        return (x,) + inp[1:]
    hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(ohook))
for l in PRUNE_MLPS:
    def mhook(mod, inp, outp, l=l):
        if not ACTIVE["on"]: return None
        return torch.zeros_like(outp)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

def loss(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]

ACTIVE["on"] = False
intact = [loss(sq) for sq in eval_chunks]
print(f"intact held-out loss: {sum(intact)/len(intact):.4f}")

# heal norms with the prune-set zeroed
ACTIVE["on"] = True
for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
for ep in range(EPOCHS):
    for sq in train_chunks:
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
model.eval()
healed = [loss(sq) for sq in eval_chunks]
for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()

dmg = [h - it for h, it in zip(healed, intact)]
import random
random.seed(0); B = 5000; n = len(dmg)
bs = sorted(sum(dmg[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
print(f"PRUNE-set zero-ablation + heal: {sum(dmg)/n:+.3f} nats  95% CI [{bs[int(.025*B)]:+.3f}, {bs[int(.975*B)]:+.3f}]")
print("compare: code+heal +0.70, zero-of-code-set+heal +2.12")
