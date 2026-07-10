"""Chapter 2, step 9b: heal at higher replacement fractions.

Same as heal.py but: K configurable (default 256 = 57% of heads), longer
healing, repeated-random chunks included in the healing data, and both eval
sets (natural held-out + repeated random) tracked.

  usage: python3 heal2.py [K] [EPOCHS]
"""
import sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
K = int(sys.argv[1]) if len(sys.argv) > 1 else 256
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
T_TR, LR = 600, 3e-4

RR.W.update(torch.load("rich_templates.pt"))
costs = torch.load("rich_solo_costs.pt")
if "combined" in sys.argv:
    rnd_costs = torch.load("rich_solo_rnd.pt")
    nat_rank = {k: i for i, k in enumerate(sorted(costs, key=lambda k: costs[k]))}
    rnd_rank = {k: i for i, k in enumerate(sorted(rnd_costs, key=lambda k: rnd_costs[k]))}
    HEADS = sorted(costs, key=lambda k: nat_rank[k] + rnd_rank[k])[:K]
else:
    HEADS = sorted(costs, key=lambda k: costs[k])[:K]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)
print(f"replacing {K}/448 heads ({K/4.48:.0f}%), healing {EPOCHS} epochs")

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

def head_A(seq):
    base = RR.code_attn(seq)
    n = len(seq)
    out = {}
    for l, hs in BY_LAYER.items():
        mats = []
        for h in hs:
            M = torch.zeros(n, n)
            for k, wk in RR.W[(l, h)].items():
                if wk > 1e-4: M += wk * base[k]
            mats.append(M / M.sum(-1, keepdim=True).clamp(min=1e-9))
        out[l] = torch.stack(mats).to(torch.float16)
    return out

print("precomputing code attention...")
A_chunks = [head_A(sq) for sq in chunks]
A_nat, A_rnd = head_A(test_seq), head_A(rnd_eval)

HOLDER = {"A": None}
vcache, hooks = {}, []
for l, hs in BY_LAYER.items():
    attn = model.model.layers[l].self_attn
    def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
    hooks.append(attn.v_proj.register_forward_hook(vhook))
    def ohook(mod, inp, l=l, hs=hs):
        x = inp[0].clone()
        A = HOLDER["A"][l].to(DEV).float()
        v = vcache[l]
        for mi, h in enumerate(hs):
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = A[mi] @ v[:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))

def nll(seq, A):
    HOLDER["A"] = A
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
oproj_params = [model.model.layers[l].self_attn.o_proj.weight for l in range(28)] \
    if "oproj" in sys.argv else []
train_params = norm_params + oproj_params
orig = [p.detach().clone() for p in train_params]
b_nat, b_rnd = nll(test_seq, A_nat), nll(rnd_eval, A_rnd)
print(f"hybrid-{K} before healing: natural {b_nat:.3f}  rnd {b_rnd:.3f}"
      + (f"  (+o_proj: {sum(p.numel() for p in oproj_params):,} params)" if oproj_params else ""))

for p in train_params: p.requires_grad_(True)
opt = torch.optim.Adam([{"params": norm_params, "lr": LR},
                        {"params": oproj_params, "lr": 1e-4}])
model.train()
for ep in range(EPOCHS):
    for sq, A in zip(chunks, A_chunks):
        HOLDER["A"] = A
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
    if ep % 5 == 4:
        model.eval()
        print(f"  epoch {ep+1}: natural {nll(test_seq, A_nat):.3f}  rnd {nll(rnd_eval, A_rnd):.3f}", flush=True)
        model.train()
model.eval()
a_nat, a_rnd = nll(test_seq, A_nat), nll(rnd_eval, A_rnd)
tag = "_oproj" if oproj_params else ""
torch.save({"norms": [p.detach().cpu() for p in norm_params], "K": K}, f"healed_norms_{K}{tag}.pt")

for p, o in zip(train_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([test_seq]).to(DEV)).logits[0, :-1].float(), -1)
i_nat = -lp.gather(-1, torch.tensor(test_seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([rnd_eval]).to(DEV)).logits[0, :-1].float(), -1)
i_rnd = -lp.gather(-1, torch.tensor(rnd_eval[1:]).to(DEV).unsqueeze(-1)).mean().item()

print(f"\nK={K}: natural intact {i_nat:.3f} | before {b_nat:.3f} (+{b_nat-i_nat:.3f}) | "
      f"healed {a_nat:.3f} (+{a_nat-i_nat:.3f}, recovered {(b_nat-a_nat)/(b_nat-i_nat):.0%})")
print(f"       rnd     intact {i_rnd:.3f} | before {b_rnd:.3f} (+{b_rnd-i_rnd:.3f}) | "
      f"healed {a_rnd:.3f} (+{a_rnd-i_rnd:.3f})")
