"""Chapter 2, step 9: HEAL the hybrid -- let the network adapt to its code organs.

Substitute the 160 cheapest heads (rich templates), then fine-tune ONLY the
RMSNorm gain vectors (~57k params of 596M) on fresh natural chunks with the
substitution active. If post-replacement damage is distribution shift, the
norms can re-center the residual stream around the prosthetics.

Eval: held-out natural NLL with substitution, before vs after healing
(plus intact reference; intact model itself is never touched -- we restore
original norms for the reference measurement).
"""
import copy, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz, T_EVAL = RA.tokz, RA.T
K, T_TR, EPOCHS, LR = 160, 600, 20, 3e-4

RR.W.update(torch.load("rich_templates.pt"))
costs = torch.load("rich_solo_costs.pt")
HEADS = sorted(costs, key=lambda k: costs[k])[:K]
BY_LAYER = {}
for l, h in HEADS: BY_LAYER.setdefault(l, []).append(h)

raw = open("ministral_corpus.txt").read()
starts = [20000 + 30000 * i for i in range(8)]           # disjoint from test @200k? 200k hits: avoid
starts = [20000, 50000, 80000, 110000, 140000, 260000, 290000, 320000]
chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in starts]
test_seq = RA.test_seq

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
        out[l] = torch.stack(mats).to(torch.float16)     # [m, n, n] cpu fp16
    return out

print(f"precomputing code attention for {len(chunks)} train chunks + eval...")
A_chunks = [head_A(sq) for sq in chunks]
A_eval = head_A(test_seq)

HOLDER = {"A": None}
vcache = {}
hooks = []
for l, hs in BY_LAYER.items():
    attn = model.model.layers[l].self_attn
    def vhook(mod, inp, outp, l=l): vcache[l] = outp[0]
    hooks.append(attn.v_proj.register_forward_hook(vhook))
    def ohook(mod, inp, l=l, hs=hs):
        x = inp[0].clone()
        A = HOLDER["A"][l].to(DEV).float()               # [m,n,n]
        v = vcache[l]                                    # [n, NKV*DH]
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
orig = [p.detach().clone() for p in norm_params]
print(f"trainable: {sum(p.numel() for p in norm_params):,} norm params "
      f"of {sum(p.numel() for p in model.parameters()):,}")

before = nll(test_seq, A_eval)
print(f"held-out hybrid BEFORE healing: {before:.3f}")

for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
for ep in range(EPOCHS):
    tot = 0
    for sq, A in zip(chunks, A_chunks):
        HOLDER["A"] = A
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
        tot += out.loss.item()
    if ep % 4 == 3 or ep == 0:
        model.eval()
        print(f"  epoch {ep+1}: train {tot/len(chunks):.3f}  held-out {nll(test_seq, A_eval):.3f}", flush=True)
        model.train()
model.eval()
after = nll(test_seq, A_eval)

# intact reference with ORIGINAL norms
for p, o in zip(norm_params, orig): p.data.copy_(o)
for hk in hooks: hk.remove()
with torch.no_grad():
    lp = torch.log_softmax(model(torch.tensor([test_seq]).to(DEV)).logits[0, :-1].float(), -1)
intact = -lp.gather(-1, torch.tensor(test_seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

print(f"\nintact {intact:.3f}   hybrid-160 before {before:.3f} (+{before-intact:.3f})"
      f"   after healing {after:.3f} (+{after-intact:.3f})")
print(f"healing recovered {(before-after)/(before-intact):.0%} of the substitution damage")
