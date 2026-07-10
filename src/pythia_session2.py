"""Session 2: replicate the three headline measurements on a second model
family — EleutherAI/pythia-410m (GPT-NeoX arch: fused QKV, parallel residual,
different lab, different training data).

  A. induction hunt: behavioral repeat-collapse + per-head induction scores
  B. substitution ceiling: true attention masked to rule-nameable columns
     (+BOS) vs masked to everything else, on natural text
  C. MLP U-map: per-layer solo cost of replacing the MLP with a token->vector
     lookup table

Qwen3-0.6B references: behavioral 14.44->0.69; top head 0.979; ceiling 32%
masked / 67% inverse (natural), 100% masked (repeated); MLP U-shape with
near-free middle (L10 negative) and expensive ends.
"""
import torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, L, BATCH, T = 8, 50, 8, 1000

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads
DH = cfg.hidden_size // NH
V = cfg.vocab_size
print(f"{MODEL}: {NL} layers x {NH} heads, head_dim {DH}")

layers = model.gpt_neox.layers

# ---------------- A. induction hunt ----------------
torch.manual_seed(0)
seq = torch.randint(1000, V - 1000, (BATCH, L))
ids = torch.cat([seq, seq], dim=1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
first, second = nll[:, :L-1].mean().item(), nll[:, L-1:].mean().item()
print(f"\nA. behavioral: first half {first:.2f}  repeated half {second:.2f}  "
      f"(drop {first-second:.2f} nats)   [qwen: 14.44 -> 0.69]")

qpos = torch.arange(L, 2*L - 1)
ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    a = att.float().cpu()
    ind[l] = a[:, :, qpos, qpos - L + 1].mean(dim=(0, 2))
del out
v_, i_ = ind.flatten().topk(8)
print("   top induction heads:", ", ".join(
    f"L{int(x)//NH}.H{int(x)%NH}={float(s):.2f}" for x, s in zip(i_, v_)))
HEADS = [(l, h) for l in range(NL) for h in range(NH) if ind[l, h] > 0.2]
print(f"   heads with score > 0.2: {len(HEADS)} of {NL*NH}")
torch.save(ind, "pythia_head_scores.pt")

# ---------------- shared machinery ----------------
raw = open("ministral_corpus.txt").read()
test_seq = tokz.encode(raw[200000:212000])[:T]
train_seq = tokz.encode(raw[:12000])[:T]

def match_cols(seq):
    """rule-nameable columns: follower positions of exact suffix matches."""
    occ = {}
    cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            cols[i].add(p + 1)
        occ.setdefault(seq[i], []).append(i)
    return cols

VCACHE = {}
def vhook_for(l):
    def hook(mod, inp, outp, l=l):
        # fused qkv: [B,T,NH*3*DH], per-head layout [q|k|v]
        q3 = outp.view(outp.shape[0], outp.shape[1], NH, 3 * DH)
        VCACHE[l] = q3[0, :, :, 2*DH:].detach()          # [T, NH, DH]
    return hook

def run_masked(seq, byl, Afn):
    """substitute heads in byl: head output = A(l,h) @ v_h; Afn(l,h)->[T,T] on DEV"""
    hooks = []
    for l, hs in byl.items():
        att = layers[l].attention
        hooks.append(att.query_key_value.register_forward_hook(vhook_for(l)))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                x[0, :, h*DH:(h+1)*DH] = Afn(l, h) @ VCACHE[l][:, h, :]
            return (x,) + inp[1:]
        hooks.append(att.dense.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lpp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for hk in hooks: hk.remove()
    return -lpp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def run_zero(seq, byl):
    hooks = []
    for l, hs in byl.items():
        def ohook(mod, inp, hs=hs):
            x = inp[0].clone()
            for h in hs: x[0, :, h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(layers[l].attention.dense.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lpp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for hk in hooks: hk.remove()
    return -lpp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def nll_plain(seq):
    with torch.no_grad():
        lpp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lpp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

# ---------------- B. ceiling ----------------
BYL = {}
for l, h in HEADS: BYL.setdefault(l, []).append(h)

def ceiling(seq, label):
    n = len(seq)
    with torch.no_grad():
        outp = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    ATT = {l: outp.attentions[l][0].float() for l in BYL}    # [NH,T,T] on DEV
    del outp
    cols = match_cols(seq)
    mask = torch.zeros(n, n)
    for i, cs in enumerate(cols):
        for j in cs:
            if j < n: mask[i, j] = 1
    mask[:, 0] = 1
    maskD = mask.to(DEV); invD = torch.tril(1 - mask).to(DEV)
    ni = nll_plain(seq)
    nz = run_zero(seq, BYL)
    nm = run_masked(seq, BYL, lambda l, h: ATT[l][h] * maskD)
    nv = run_masked(seq, BYL, lambda l, h: ATT[l][h] * invD)
    gap = nz - ni
    print(f"   {label}: intact {ni:.3f} zero {nz:.3f} | masked-to-rule {nm:.3f} "
          f"({(nz-nm)/gap:.0%}) | inverse {nv:.3f} ({(nz-nv)/gap:.0%})")

print(f"\nB. substitution ceiling ({len(HEADS)} heads)   [qwen: 32% masked / 67% inverse natural]")
ceiling(test_seq, "natural")
torch.manual_seed(3)
rnd = (lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist())
ceiling(rnd, "repeated")

# ---------------- C. MLP U-map ----------------
print("\nC. MLP token-lookup U-map   [qwen: middle ~free (L10 negative), ends expensive]")
starts = [0, 20000, 50000, 80000, 110000, 140000, 260000, 290000]
fit_chunks = [tokz.encode(raw[o:o+8000])[:900] for o in starts]
SUM = [dict() for _ in range(NL)]; tot = [None]*NL; cap = {}
hooks = [layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in range(NL)]
cnt = 0
with torch.no_grad():
    for sq in fit_chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in range(NL):
            o = cap[l]
            tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for hk in hooks: hk.remove()
MEAN = [tot[l]/cnt for l in range(NL)]
LUT = [{t: v/n for t, (v, n) in SUM[l].items()} for l in range(NL)]

def run_mlp(seq, ls):
    mats = {l: torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV) for l in ls}
    hooks = [layers[l].mlp.register_forward_hook(
        (lambda mod, inp, outp, l=l: mats[l].unsqueeze(0))) for l in ls]
    with torch.no_grad():
        lpp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for hk in hooks: hk.remove()
    return -lpp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

ni_tr = nll_plain(train_seq)
costs = {}
for l in range(NL):
    costs[l] = run_mlp(train_seq, [l]) - ni_tr
    print(f"   L{l:>2}: {costs[l]:+.4f}", flush=True)
torch.save(costs, "pythia_mlp_costs.pt")
mid = sorted(costs.values())[NL//2]
print(f"   median {mid:+.4f}; ends L0 {costs[0]:+.3f} L1 {costs[1]:+.3f} "
      f"L{NL-2} {costs[NL-2]:+.3f} L{NL-1} {costs[NL-1]:+.3f}")
