"""Probe, stage 1: is the diffuse (un-nameable) part of induction attention
LOW-DIMENSIONAL?

For each induction head on fiction text, take its true attention, remove the
rule-nameable columns (exact-match followers + BOS), and renormalize the
remainder -> D, the "diffuse" attention distribution the ~30% ceiling could
not name. Ask: can a rank-r bilinear readout of the residual stream reconstruct
D?  logits[i,j] = (h_i P)(h_j Q)^T  over causal non-rule j, softmax -> Dhat.

Sweep r; compare to a random-direction control (P,Q random, not fit). Metrics
on HELD-OUT positions: top-1 agreement (does argmax Dhat land on argmax D) and
mean KL(D||Dhat). If r=1..2 already captures most of D, the diffuse part is a
few learned directions, not irreducible — the probe is alive.
"""
import gc, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L = 320, 50
RANKS = [1, 2, 4, 8, 16]

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
V = cfg.vocab_size

# induction heads
torch.manual_seed(0)
sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): out = model(ids, output_attentions=True)
qpos = torch.arange(L, 2*L-1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
del out; gc.collect()
flat = ind.flatten()
top = flat.topk(6).indices
HEADS = [(int(x)//NH, int(x)%NH) for x in top]
print(f"probing top induction heads: {HEADS}")

# residual-stream capture (input to each induction layer)
LAYERS = sorted({l for l, _ in HEADS})
resid = {}
def prehook(l):
    def h(mod, args, kwargs):
        resid[l] = args[0].detach()
    return h
hooks = [model.model.layers[l].register_forward_pre_hook(prehook(l), with_kwargs=True)
         for l in LAYERS]

def match_cols(seq):
    occ = {}; cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p+1 < len(seq): cols[i].add(p+1)
        occ.setdefault(seq[i], []).append(i)
    return cols

raw = open("ministral_corpus.txt").read()
def chunk(o): return tokz.encode(raw[o:o+6000])[:T_C]
TRAIN = [chunk(o) for o in (0, 40000, 80000, 120000, 160000, 240000)]
TEST  = [chunk(o) for o in (300000, 340000, 380000)]

def collect(seqs):
    """per head: list of (h_i, diffuse target dist over non-rule j, valid mask, residuals)."""
    data = {hd: [] for hd in HEADS}
    for seq in seqs:
        n = len(seq)
        with torch.no_grad():
            o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        cols = match_cols(seq)
        rule = torch.zeros(n, n)
        for i, cs in enumerate(cols):
            for j in cs: rule[i, j] = 1
        rule[:, 0] = 1
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu()          # [n,n]
            H = resid[l][0].float().cpu()                    # [n,D]
            diff = A * (1 - rule)                             # remove nameable
            diff = torch.tril(diff)
            mass = diff.sum(-1)
            valid = mass > 0.15                              # rows with real diffuse mass
            for i in range(2, n):
                if not valid[i]: continue
                dist = diff[i, :i] / diff[i, :i].sum()
                allowed = (1 - rule[i, :i]).bool()
                if allowed.sum() < 2: continue
                data[(l, h)].append((H[i], H[:i], dist, allowed))
        del o; gc.collect()
    return data

print("collecting train/test residuals + attention...")
TR = collect(TRAIN); TE = collect(TEST)
for hd in HEADS:
    print(f"  {hd}: {len(TR[hd])} train / {len(TE[hd])} test positions")

def fit_head(samples, r, steps=300, seed=0):
    torch.manual_seed(seed)
    P = torch.randn(D, r) * 0.02; Q = torch.randn(D, r) * 0.02
    P.requires_grad_(); Q.requires_grad_()
    opt = torch.optim.Adam([P, Q], lr=0.05)
    for _ in range(steps):
        loss = 0.0
        for hi, Hj, dist, allowed in samples:
            lg = (hi @ P) @ (Hj @ Q).T                       # [i]
            lg = lg.masked_fill(~allowed, -1e9)
            loss = loss - (dist * F.log_softmax(lg, -1)).sum()
        loss = loss / len(samples)
        opt.zero_grad(); loss.backward(); opt.step()
    return P.detach(), Q.detach()

def evaluate(samples, P, Q):
    kl, hit, n = 0.0, 0, 0
    for hi, Hj, dist, allowed in samples:
        lg = (hi @ P) @ (Hj @ Q).T
        lg = lg.masked_fill(~allowed, -1e9)
        pred = F.softmax(lg, -1)
        kl += (dist * (dist.clamp(min=1e-9).log() - pred.clamp(min=1e-9).log())).sum().item()
        hit += int(pred.argmax() == dist.argmax())
        n += 1
    return kl/n, hit/n

print(f"\n{'rank':>5}{'fit KL':>9}{'fit top1':>10}{'rand KL':>9}{'rand top1':>11}")
for r in RANKS:
    kls, hits, rkls, rhits = [], [], [], []
    for hd in HEADS:
        if len(TR[hd]) < 10 or len(TE[hd]) < 5: continue
        P, Q = fit_head(TR[hd], r)
        k, h = evaluate(TE[hd], P, Q)
        Pr = torch.randn(D, r) * 0.02; Qr = torch.randn(D, r) * 0.02
        rk, rh = evaluate(TE[hd], Pr, Qr)
        kls.append(k); hits.append(h); rkls.append(rk); rhits.append(rh)
    import statistics as st
    print(f"{r:>5}{st.mean(kls):>9.2f}{st.mean(hits):>10.0%}"
          f"{st.mean(rkls):>9.2f}{st.mean(rhits):>11.0%}", flush=True)

# baseline: how often does the true diffuse argmax fall on an entity-ish token?
print("\n(reference) mean positions/head:", int(sum(len(TE[hd]) for hd in HEADS)/len(HEADS)))
for hk in hooks: hk.remove()
