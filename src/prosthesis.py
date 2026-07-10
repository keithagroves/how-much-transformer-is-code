"""Chapter 2, step 5: substitute the induction circuit with code.

Excise: zero-ablate all heads with induction score > 0.1 (60 of 448).
Prosthesis: at the logits, an explicit rule --
    find the longest suffix of the context that occurred earlier;
    add boost b[order] to the logit of the token that followed it.
b[order] is FITTED on a train chunk (per-order 1D exact optimization, since
each order's boost only touches its own positions), then FROZEN and evaluated
on a held-out chunk and on repeated random sequences.

If the circuit is a calibrated backoff rule, the hybrid should recover most
of the ablation gap -- and b[1] (weak match) should come out NEGATIVE,
rediscovering the veto without being told about it.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, T = 8, 1000

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
DH = model.config.head_dim
s = torch.load("head_scores.pt")
ABLATE = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.1]
print(f"excising {len(ABLATE)} heads (ind > 0.1)")

raw = open("ministral_corpus.txt").read()
train_ids = tokz.encode(raw[:8000])[:T]
test_ids  = tokz.encode(raw[200000:212000])[:T]

def run(ids_list, off):
    ids = torch.tensor([ids_list]).to(DEV)
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

def rule(seq):
    """per position i (predicting i+1): (match order, copy candidate) or (0, -1)"""
    out = []
    for i in range(len(seq) - 1):
        best_o, cand = 0, -1
        for o in range(min(MAXO, i + 1), 0, -1):
            key = tuple(seq[i - o + 1 : i + 1])
            hit = -1
            for j in range(i - 1, o - 2, -1):
                if tuple(seq[j - o + 1 : j + 1]) == key: hit = j; break
            if hit >= 0: best_o, cand = o, seq[hit + 1]; break
        out.append((best_o, cand))
    return out

def stats(seq, lp):
    """per position: p_target, p_cand, order, is_cand_target"""
    R = rule(seq)
    t = torch.tensor(seq[1:])
    pt = lp.gather(-1, t.unsqueeze(-1)).squeeze(-1).exp()
    o = torch.tensor([r[0] for r in R])
    c = torch.tensor([max(r[1], 0) for r in R])
    pc = lp.gather(-1, c.unsqueeze(-1)).squeeze(-1).exp()
    pc[o == 0] = 0
    return pt, pc, o, (c == t) & (o > 0)

def hybrid_nll(pt, pc, o, hit, b):
    """NLL after adding boost b[order] to the candidate's logit."""
    beta = torch.tensor([b.get(int(x), 0.0) for x in o])
    z = 1 - pc + pc * beta.exp()                     # new normalizer / old
    lp_t = torch.where(hit, (pc * beta.exp() / z).log(), (pt / z).log())
    lp_t[o == 0] = pt[o == 0].log()
    return -lp_t.mean().item(), lp_t

# ---- fit boosts on train ----
lp_tr = run(train_ids, ABLATE)
pt, pc, o, hit = stats(train_ids, lp_tr)
b = {}
grid = torch.linspace(-6, 12, 181)
for order in range(1, MAXO + 1):
    m = o == order
    if m.sum() < 3: continue
    best = min(grid, key=lambda g: hybrid_nll(pt[m], pc[m], o[m], hit[m], {order: float(g)})[0])
    b[order] = float(best)
print("fitted boosts b[order]:", {k: round(v, 1) for k, v in sorted(b.items())})

# ---- evaluate ----
def evaluate(name, ids_list):
    seq = ids_list
    lp_int = run(seq, [])
    lp_abl = run(seq, ABLATE)
    t = torch.tensor(seq[1:])
    nll_int = -lp_int.gather(-1, t.unsqueeze(-1)).mean().item()
    nll_abl = -lp_abl.gather(-1, t.unsqueeze(-1)).mean().item()
    pt2, pc2, o2, hit2 = stats(seq, lp_abl)
    nll_hyb, _ = hybrid_nll(pt2, pc2, o2, hit2, b)
    gap = nll_abl - nll_int
    rec = (nll_abl - nll_hyb) / gap if gap > 1e-6 else float("nan")
    print(f"{name:<18} intact {nll_int:.3f}  ablated {nll_abl:.3f}  "
          f"hybrid {nll_hyb:.3f}   gap recovered {rec:.0%}")

evaluate("train chunk", train_ids)
evaluate("held-out chunk", test_ids)

torch.manual_seed(1)
rnd = torch.randint(1000, model.config.vocab_size - 1000, (50,)).tolist()
evaluate("repeated random", rnd + rnd)
