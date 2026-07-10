"""Chapter 2, step 10b: heal the MLP-lookup hybrid.

Replace the K cheapest MLP layers with token->vector lookup tables (fitted on
24 chunks), then heal RMSNorm gains only (the recipe that worked for heads;
o_proj-style big-matrix healing diverges).

  usage: python3 heal_mlp.py [K] [EPOCHS]
"""
import sys, torch
import replace_all as RA

model, DEV, NL, tokz = RA.model, RA.DEV, RA.NL, RA.tokz
K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
T_TR, LR = 600, 3e-4

costs = torch.load("mlp_solo_costs.pt")["costs"]
LAYERS = sorted(costs, key=lambda l: costs[l])[:K]
print(f"replacing MLPs {sorted(LAYERS)} with token-lookup tables")

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in starts]
test_seq = RA.test_seq
torch.manual_seed(23)
rnd_eval = (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist())

# ---------- fit lookup tables on all 24 chunks ----------
print("fitting lookup tables...")
SUM = {l: {} for l in LAYERS}
tot = {l: None for l in LAYERS}
cap = {}
hooks = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in LAYERS]
cnt = 0
with torch.no_grad():
    for sq in chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in LAYERS:
            o = cap[l]
            tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for hk in hooks: hk.remove()
MEAN = {l: tot[l] / cnt for l in LAYERS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in LAYERS}
print(f"tokens seen: {len(LUT[LAYERS[0]]):,}")

def lut_mat(seq, l):
    return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

HOLDER = {}
hooks = []
for l in LAYERS:
    def hook(mod, inp, outp, l=l):
        return HOLDER[l].unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(hook))

def set_seq(seq):
    for l in LAYERS: HOLDER[l] = lut_mat(seq, l)

def nll(seq):
    set_seq(seq)
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]
b_nat, b_rnd = nll(test_seq), nll(rnd_eval)
print(f"MLP-hybrid-{K} before healing: natural {b_nat:.3f}  rnd {b_rnd:.3f}")

for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
lut_cache = [None] * len(chunks)
for ep in range(EPOCHS):
    for ci, sq in enumerate(chunks):
        if lut_cache[ci] is None:
            lut_cache[ci] = {l: lut_mat(sq, l) for l in LAYERS}
        HOLDER.update(lut_cache[ci])
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if ep % 5 == 4:
        model.eval()
        print(f"  epoch {ep+1}: natural {nll(test_seq):.3f}  rnd {nll(rnd_eval):.3f}", flush=True)
        model.train()
model.eval()
a_nat, a_rnd = nll(test_seq), nll(rnd_eval)

for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([test_seq]).to(DEV)).logits[0, :-1].float(), -1)
i_nat = -lp.gather(-1, torch.tensor(test_seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([rnd_eval]).to(DEV)).logits[0, :-1].float(), -1)
i_rnd = -lp.gather(-1, torch.tensor(rnd_eval[1:]).to(DEV).unsqueeze(-1)).mean().item()

print(f"\nMLP K={K}: natural intact {i_nat:.3f} | before {b_nat:.3f} (+{b_nat-i_nat:.3f}) | "
      f"healed {a_nat:.3f} (+{a_nat-i_nat:.3f}, recovered {(b_nat-a_nat)/(b_nat-i_nat):.0%})")
print(f"          rnd    intact {i_rnd:.3f} | before {b_rnd:.3f} | healed {a_rnd:.3f}")
