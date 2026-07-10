"""Chapter 2, step 5e: prosthesis v5 -- SOFT multi-occurrence rule attention.

v4-gated recovered 96% verbatim / 26% natural with hard attention to the
single most-recent longest match. Hypothesis for the missing 74%: real heads
attend DIFFUSELY -- over all matching occurrences, graded by match strength.

v5 code rule, per query position i:
    for every earlier follower position j: m(j) = length of the common
    suffix between context ending at i and context ending at j-1 (<= MAXO)
    attention(i, j) = softmax_j( alpha * m(j) )  over j with m(j) >= 1
    head write = gate[max_m(i)] * sum_j attention(i,j) * V_head(hidden[j])

Params: alpha (sharpness) + 8 per-order gains. OV kept, as in v4.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, T = 8, 1000

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
DH, NH, NKV = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
GROUP = NH // NKV
V = cfg.vocab_size

s = torch.load("head_scores.pt")
HEADS = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.2]
LAYERS = sorted({l for l, _ in HEADS})

raw = open("ministral_corpus.txt").read()
train_seq = tokz.encode(raw[:12000])[:T]
test_seq = tokz.encode(raw[200000:212000])[:T]
torch.manual_seed(1)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist())
rnd_all = [mkrnd() for _ in range(4)]
rnd_fit, rnd_test = rnd_all[:1], rnd_all[1:]

def match_table(seq):
    """[(i, j, m)] for all followers j of suffix matches, m capped at MAXO."""
    occ = {}
    out = []
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):          # p: earlier position with same token
            m = 1
            while (m < MAXO and p - m >= 0 and i - m >= 0
                   and seq[p - m] == seq[i - m]):
                m += 1
            out.append((i, p + 1, m))          # attend to FOLLOWER of the match
        occ.setdefault(seq[i], []).append(i)
    return out

CACHE = {}
def attn_matrix(seq, alpha, gains):
    """dense [T,T] rule-attention, rows gated by g[max match order]."""
    key = id(seq)
    if key not in CACHE: CACHE[key] = match_table(seq)
    tab = CACHE[key]
    n = len(seq)
    A = torch.zeros(n, n)
    mrow = torch.zeros(n, dtype=torch.long)
    for i, j, m in tab:
        A[i, j] = torch.tensor(alpha * m).exp()
        mrow[i] = max(mrow[i], m)
    Z = A.sum(-1, keepdim=True).clamp(min=1e-9)
    g = torch.zeros(MAXO + 1); g[1:] = torch.tensor(gains)
    return (A / Z) * g[mrow].unsqueeze(-1)

def run(seq, mode, alpha=2.0, gains=None):
    """mode: 'intact' | 'zero' | 'soft'"""
    hooks = []
    if mode != "intact":
        A = (attn_matrix(seq, alpha, gains).to(DEV) if mode == "soft" else None)
        vcache = {}
        for l in LAYERS:
            attn = model.model.layers[l].self_attn
            def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
            hooks.append(attn.v_proj.register_forward_hook(vhook))
            hs = [h for ll, h in HEADS if ll == l]
            def ohook(mod, inp, l=l, hs=hs):
                x = inp[0].clone()
                for h in hs:
                    g = h // GROUP
                    if mode == "soft":
                        x[0, :, h*DH:(h+1)*DH] = A @ vcache[l][:, g*DH:(g+1)*DH]
                    else:
                        x[0, :, h*DH:(h+1)*DH] = 0
                return (x,) + inp[1:]
            hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    print(f"v5: soft rule-attention, {len(HEADS)} heads")
    fit_seqs = [train_seq] + rnd_fit
    best = None
    for alpha in (1.0, 2.0, 4.0):
        gains = [0.0] * MAXO
        for sweep in range(2):
            for oi in range(MAXO):
                cur = None
                for cand in (0.0, 0.25, 0.5, 0.75, 1.0):
                    gains[oi] = cand
                    nll = sum(run(sq, "soft", alpha, gains) for sq in fit_seqs)
                    if cur is None or nll < cur[0]: cur = (nll, cand)
                gains[oi] = cur[1]
        if best is None or cur[0] < best[0]: best = (cur[0], alpha, gains[:])
    _, alpha, gains = best
    print(f"fitted alpha {alpha}, gains {gains}")

    for name, seqs in [("natural B (held-out)", [test_seq]), ("repeated random", rnd_test)]:
        ni = sum(run(sq, "intact") for sq in seqs) / len(seqs)
        nz = sum(run(sq, "zero") for sq in seqs) / len(seqs)
        ns = sum(run(sq, "soft", alpha, gains) for sq in seqs) / len(seqs)
        print(f"{name:<22} intact {ni:.3f}  zero {nz:.3f}  soft-rule {ns:.3f}   "
              f"recovered {(nz-ns)/(nz-ni):.0%}")
