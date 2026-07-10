"""Chapter 2, step 5b: code prosthesis, v2.

v1 failed (0% gap recovery): train data had no high-order matches, and an
exact-single-candidate boost can't reproduce fuzzy induction. v2:

  P1  exact rule: boost the follower of the most recent longest match
  P2  rulebook:   boost ALL followers of the longest match, count-weighted
  P3  fuzzy:      P2 + smear each boost over the candidate's unembedding
                  neighbors (cosine in output-embedding space)

Boost of token c at a position with match order o:  b[o] * w_c, with per-order
scalars b[o] fitted on train data (natural chunk + repeated-random sequences so
every order is represented), then FROZEN for eval on held-out text.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, T, K = 8, 1000, 12          # K = max boosted tokens per position

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
DH = model.config.head_dim
V = model.config.vocab_size
s = torch.load("head_scores.pt")
ABLATE = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.1]

E = model.get_output_embeddings().weight.detach().float().cpu()
E = E / E.norm(dim=-1, keepdim=True)

raw = open("ministral_corpus.txt").read()
train_nat = tokz.encode(raw[:8000])[:T]
test_nat  = tokz.encode(raw[200000:212000])[:T]
torch.manual_seed(1)
rnd_train = [(lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist()) for _ in range(3)]
rnd_test  = [(lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist()) for _ in range(3)]

def run(seq, off):
    ids = torch.tensor([seq]).to(DEV)
    hooks = []
    for l, h in off:
        def hook(mod, inp, h=h):
            x = inp[0].clone(); x[..., h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return lp

# ---------- the rule, three variants ----------
def matches(seq, i):
    """longest-suffix matches: (order, Counter of followers)"""
    from collections import Counter
    for o in range(min(MAXO, i + 1), 0, -1):
        key = tuple(seq[i - o + 1 : i + 1])
        fol = Counter()
        for j in range(o - 1, i):
            if tuple(seq[j - o + 1 : j + 1]) == key: fol[seq[j + 1]] += 1
        if fol: return o, fol
    return 0, None

def candidates(seq, variant):
    """per position: (order, [(token, weight)...])  weights in (0,1]"""
    out = []
    nbr_cache = {}
    for i in range(len(seq) - 1):
        o, fol = matches(seq, i)
        if not fol: out.append((0, [])); continue
        tot = sum(fol.values())
        cand = {t: n / tot for t, n in fol.items()}
        if variant == "P3":
            smear = {}
            for t, w in list(cand.items()):
                if t not in nbr_cache:
                    sim = E @ E[t]
                    v, ix = sim.topk(6)
                    nbr_cache[t] = [(int(a), float(c)) for a, c in zip(ix[1:], v[1:]) if c > 0.4]
                for a, c in nbr_cache[t]:
                    smear[a] = max(smear.get(a, 0), w * c * 0.5)
            for a, w in smear.items(): cand[a] = max(cand.get(a, 0), w)
        top = sorted(cand.items(), key=lambda kv: -kv[1])[:K]
        out.append((o, top))
    return out

def build(seq, lp, cands):
    """tensors for analytic hybrid NLL: p of each candidate, weights, orders"""
    N = len(seq) - 1
    pc = torch.zeros(N, K); w = torch.zeros(N, K)
    tin = torch.full((N,), -1, dtype=torch.long)     # column of target if boosted
    o = torch.zeros(N, dtype=torch.long)
    for i, (oo, top) in enumerate(cands):
        o[i] = oo
        for k, (t, wt) in enumerate(top):
            pc[i, k] = lp[i, t].exp(); w[i, k] = wt
            if t == seq[i + 1]: tin[i] = k
    pt = lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).squeeze(-1).exp()
    return pt, pc, w, tin, o

def hybrid_nll(pt, pc, w, tin, o, b):
    beta = torch.tensor([b.get(int(x), 0.0) for x in o]).unsqueeze(-1) * w
    z = 1 - pc.sum(-1) + (pc * beta.exp()).sum(-1)
    boosted = tin >= 0
    bt = torch.where(boosted, beta.gather(-1, tin.clamp(min=0).unsqueeze(-1)).squeeze(-1),
                     torch.zeros_like(pt))
    lp_t = (pt * bt.exp() / z).log()
    lp_t[o == 0] = pt[o == 0].log()
    return -lp_t.mean().item()

# ---------- fit + eval per variant ----------
def stats_for(seqs, variant, off):
    parts = [build(sq, run(sq, off), candidates(sq, variant)) for sq in seqs]
    return [torch.cat(x) for x in zip(*parts)]

grid = torch.linspace(-6, 14, 101)
for variant in ("P1", "P2", "P3"):
    if variant == "P1":
        cand_fn = lambda sq: [(o, top[:1]) for o, top in candidates(sq, "P2")]
    else:
        cand_fn = lambda sq: candidates(sq, variant)
    def stats(seqs, off):
        parts = [build(sq, run(sq, off), cand_fn(sq)) for sq in seqs]
        return [torch.cat(x) for x in zip(*parts)]

    pt, pc, w, tin, o = stats([train_nat] + rnd_train, ABLATE)
    b = {}
    for order in range(1, MAXO + 1):
        m = o == order
        if m.sum() < 3: continue
        best = min(grid, key=lambda g: hybrid_nll(pt[m], pc[m], w[m], tin[m], o[m], {order: float(g)}))
        b[order] = float(best)
    print(f"\n{variant} fitted boosts:", {k: round(v, 1) for k, v in sorted(b.items())})

    for name, seqs in [("held-out natural", [test_nat]), ("repeated random", rnd_test)]:
        pt2, pc2, w2, tin2, o2 = stats(seqs, ABLATE)
        nll_hyb = hybrid_nll(pt2, pc2, w2, tin2, o2, b)
        t = torch.cat([torch.tensor(sq[1:]) for sq in seqs])
        lp_int = torch.cat([run(sq, []) for sq in seqs])
        lp_abl = torch.cat([run(sq, ABLATE) for sq in seqs])
        nll_int = -lp_int.gather(-1, t.unsqueeze(-1)).mean().item()
        nll_abl = -lp_abl.gather(-1, t.unsqueeze(-1)).mean().item()
        rec = (nll_abl - nll_hyb) / (nll_abl - nll_int)
        print(f"  {name:<18} intact {nll_int:.3f}  ablated {nll_abl:.3f}  "
              f"hybrid {nll_hyb:.3f}   gap recovered {rec:.0%}")
