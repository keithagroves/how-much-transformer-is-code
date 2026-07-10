"""Reviewer question: is the MLP lookup table compressible, or does 'code' trade
on a 3.5M-number table? Two truncations of the token->mean-vector LUT:
  top-k : keep the k most frequent tokens' vectors, rest fall back to the layer mean
  rank-r: SVD the [n_token x d] table, keep r singular directions
Substitute the 6 chosen MLP layers with the truncated LUT (heads intact, no heal)
and measure held-out damage vs intact. If a small k or r matches the full table,
the 'code' framing strengthens; if not, the table's size is load-bearing.
"""
import collections, torch
import replace_all as RA

model, DEV = RA.model, RA.DEV
tokz = RA.tokz

mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:600] for o in train_starts]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]

# --- build LUT + token frequency from training chunks ---
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}; freq = collections.Counter()
hk = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, outp[0].detach().float().cpu())))
    for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq in train_chunks:
        model(torch.tensor([sq]).to(DEV)); freq.update(sq)
        for l in MLPS:
            o = cap[l]; tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
            for i, t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [o[i].clone(), 1]
        cnt += len(sq)
for h in hk: h.remove()
MEAN = {l: tot[l] / cnt for l in MLPS}
LUT = {l: {t: v / n for t, (v, n) in SUM[l].items()} for l in MLPS}
ntok = len(set().union(*[set(LUT[l]) for l in MLPS]))
print(f"substituted MLPs: {sorted(MLPS)}; distinct token entries ~{ntok}")

# --- build truncated tables ---
def topk_lut(k):
    keep = set(t for t, _ in freq.most_common(k))
    return {l: {t: v for t, v in LUT[l].items() if t in keep} for l in MLPS}

def rank_r_lut(r):
    """low-rank SVD of each layer's [n_tok x d] table; rows re-expanded."""
    out = {}
    for l in MLPS:
        toks = list(LUT[l]); M = torch.stack([LUT[l][t] for t in toks])   # [n,d]
        mu = M.mean(0, keepdim=True); U, S, Vh = torch.linalg.svd(M - mu, full_matrices=False)
        Mr = (U[:, :r] * S[:r]) @ Vh[:r] + mu
        out[l] = {t: Mr[i] for i, t in enumerate(toks)}
    return out

HOLDER = {"lut": None}
def lut_mat(seq, l):
    tab = HOLDER["lut"][l]
    return torch.stack([tab.get(t, MEAN[l]) for t in seq]).to(DEV)
hooks = []
for l in MLPS:
    def mhook(mod, inp, outp, l=l):
        if HOLDER["lut"] is None: return None
        return lut_mat(SEQ["s"], l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mhook))

SEQ = {"s": None}
def loss(seq):
    SEQ["s"] = seq
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

HOLDER["lut"] = None
intact = sum(loss(sq) for sq in eval_chunks) / len(eval_chunks)
print(f"intact held-out loss: {intact:.4f}\n")

def dmg(tab, label):
    HOLDER["lut"] = tab
    d = sum(loss(sq) for sq in eval_chunks) / len(eval_chunks) - intact
    print(f"  {label:<20} MLP-substitution damage {d:+.4f} nats", flush=True)
    HOLDER["lut"] = None
    return d

print("full table (all ~%d tokens):" % ntok)
full = dmg(LUT, "full")
print("top-k frequent tokens (rest -> layer mean):")
for k in [100, 300, 1000]:
    dmg(topk_lut(k), f"top-{k}")
print("low-rank (SVD of the token table):")
for r in [8, 32, 128]:
    dmg(rank_r_lut(r), f"rank-{r}")
for h in hooks: h.remove()
print(f"\nread: if a small k or r matches full ({full:+.3f}), the table compresses and 'code' strengthens.")
