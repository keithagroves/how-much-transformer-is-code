"""Stronger pruning baseline. The earlier prune baseline picked heads by SOLO
ablation cost, which mispredicts joint damage (r=0.14) and healed to a bad +3.81.
Here we select the deletion set by a joint-informed, post-heal criterion: heal the
intact model (domain adapt), rank every head by its marginal zero-ablation cost on
that healed model, delete the 160 cheapest-to-remove (+ the same 6 MLPs), re-heal,
and compare to code+heal (+0.70). If code still wins by ~3x against this stronger
baseline, that is the version a skeptic accepts.
"""
import gc, torch
import replace_all as RA
model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
LR, EPOCHS = 3e-4, 10
NL, NH = model.config.num_hidden_layers, model.config.num_attention_heads
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw)-10000,1000000),40000) if not 185000<=o<=215000][:24]
train_chunks = [tokz.encode(raw[o:o+8000])[:600] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r+r)(torch.randint(1000,RA.V-1000,(50,)).tolist())*3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw)-10000,1000000),80000) if o not in set(train_starts) and not 185000<=o<=215000][:8]
eval_chunks = [tokz.encode(raw[o:o+8000])[:300] for o in eval_starts]

# zero-ablation hooks: ZERO the heads in ACTIVE["heads"] and MLPs in ACTIVE["mlps"]
ACTIVE = {"heads": set(), "mlps": set()}
hooks = []
for l in range(NL):
    def oh(m, inp, l=l):
        hs = [h for (ll, h) in ACTIVE["heads"] if ll == l]
        if not hs: return None
        x = inp[0].clone()
        for h in hs: x[0, :, h*DH:(h+1)*DH] = 0
        return (x,) + inp[1:]
    hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(oh))
for l in range(NL):
    def mh(m, i, o, l=l):
        if l not in ACTIVE["mlps"]: return None
        return torch.zeros_like(o) if not isinstance(o, tuple) else (torch.zeros_like(o[0]),)+o[1:]
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mh))

def loss(seq):
    with torch.no_grad(): lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(),-1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
def dmg(): return sum(loss(sq) for sq in eval_chunks)/len(eval_chunks) - intact

norm = [p for n_,p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm]
def heal():
    for p, o in zip(norm, orig): p.data.copy_(o)
    for p in norm: p.requires_grad_(True)
    opt = torch.optim.Adam(norm, lr=LR); model.train()
    for ep in range(EPOCHS):
        for sq in train_chunks:
            ids = torch.tensor([sq]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
    model.eval()

ACTIVE["heads"]=set(); ACTIVE["mlps"]=set()
intact = sum(loss(sq) for sq in eval_chunks)/len(eval_chunks)
print(f"intact {intact:.4f}")

# 1. heal the intact model (domain adaptation), keep those norms for the marginal scan
heal()
# 2. marginal zero-ablation per head on the healed model
base = dmg()
marg = {}
allh = [(l,h) for l in range(NL) for h in range(NH)]
for j,(l,h) in enumerate(allh):
    ACTIVE["heads"]={(l,h)}; marg[(l,h)] = dmg() - base; ACTIVE["heads"]=set()
    if (j+1)%80==0: print(f"  marginal scan {j+1}/{len(allh)}", flush=True)
for p,o in zip(norm,orig): p.data.copy_(o)   # restore before re-heal
# 3. delete the 160 cheapest-to-remove + 6 MLPs, re-heal
prune = set(sorted(marg, key=lambda k: marg[k])[:160])
ACTIVE["heads"]=prune; ACTIVE["mlps"]=set(MLPS)
heal()
d = dmg()
ACTIVE["heads"]=set(); ACTIVE["mlps"]=set()
for p,o in zip(norm,orig): p.data.copy_(o)
for hk in hooks: hk.remove()
print(f"\nSTRONG pruning (post-heal-marginal-ranked 160 heads + 6 MLPs, re-healed): {d:+.3f} nats")
print(f"compare: code+heal +0.70 | solo-pruning +3.81 | zero-of-code-set +2.12")
print("read: if code (+0.70) still << this, code beats even joint-informed pruning.")
