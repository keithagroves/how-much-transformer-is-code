"""Query-edit causal probe, GPT-NeoX (Pythia) port of probe_queryedit.py.

Cross-families the one positive residue claim that was Qwen-only: is the rank-1
entity-selection direction WIRED INTO the head, or only correlated with its
attention? Reconstruct each induction head's TRUE attention from residual + real
weights (validated in qk_recon_neox.py), edit the query at one position along
+/- P-hat, and measure whether real softmax attention shifts toward the entity
the readout selects. Pythia specifics: fused QKV, partial rotary (first 16 of 64
dims), LayerNorm, no per-head QK-norm, no GQA.
"""
import collections, gc, math, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L, KCAND = 320, 50, 20
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH = D // NH
RND = int(DH * cfg.rotary_pct)          # rotary_ndims = 16
SCALE, V = DH ** -0.5, cfg.vocab_size


def rot_half(x):
    a, b = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


def apply_rope(x, cos, sin):            # x (n,DH); RoPE first RND dims
    xr, xp = x[..., :RND], x[..., RND:]
    xr = xr * cos + rot_half(xr) * sin
    return torch.cat((xr, xp), -1)


def ln(H, layer):
    m = layer.input_layernorm
    mu = H.mean(-1, keepdim=True); var = H.var(-1, keepdim=True, unbiased=False)
    return (H - mu) / torch.sqrt(var + m.eps) * m.weight + m.bias


def keys_of(H, l, h, cos, sin):
    """RoPE'd keys (n,DH) for head (l,h) from residual H."""
    layer = model.gpt_neox.layers[l]
    w = layer.attention.query_key_value
    z = (ln(H, layer) @ w.weight.T + w.bias).view(-1, NH, 3 * DH)
    k = apply_rope(z[:, h, DH:2 * DH], cos, sin)
    return k, layer


def edited_row(Hi_edit, l, h, i, k, layer, cos, sin):
    """Attention row at i when this head's query is built from residual Hi_edit."""
    w = layer.attention.query_key_value
    hn = ln(Hi_edit.unsqueeze(0), layer)
    q = (hn @ w.weight.T + w.bias).view(NH, 3 * DH)[h, :DH]
    q = apply_rope(q.unsqueeze(0), cos[i:i + 1], sin[i:i + 1])[0]
    logits = (k[:i + 1] @ q) * SCALE
    return torch.softmax(logits, -1)


# ---- induction heads ----
torch.manual_seed(0)
sq = torch.randint(1000, V - 1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
qp = torch.arange(L, 2 * L - 1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - L + 1].mean(dim=(0, 2))
del out; gc.collect()
HEADS = [(int(x) // NH, int(x) % NH) for x in ind.flatten().topk(6).indices]
LAYERS = sorted({l for l, _ in HEADS})
print(f"heads: {HEADS}")

resid = {}
hooks = [model.gpt_neox.layers[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())),
    with_kwargs=True) for l in LAYERS]

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)
def chunk(o): return tokz.encode(raw[o:o + 6000])[:T_C]
TRAIN = [chunk(o) for o in range(0, 520000, 40000)]
TEST = [chunk(o) for o in (540000, 580000, 620000, 660000, 700000)]


# ---- fit rank-1 selector (recency + one direction) ----
def collect(seqs):
    data = {hd: [] for hd in HEADS}
    for seq in seqs:
        n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
        with torch.no_grad():
            o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu(); H = resid[l][0].float().cpu()
            for i in range(10, n):
                cand = [j for j in ent if j < i][-KCAND:]
                if len(cand) < 3: continue
                w = A[i, cand]
                if w.sum() < 0.15: continue
                rec = torch.tensor([-math.log(i - j) for j in cand])
                data[(l, h)].append((H[i], H[cand], w / w.sum(), rec))
        del o; gc.collect()
    return data


print("fitting rank-1 selectors...")
TR = collect(TRAIN)
SEL = {}
for hd in HEADS:
    s = TR[hd]
    if len(s) < 20:
        continue
    a = torch.zeros(1, requires_grad=True)
    P = (torch.randn(D, 1) * 0.02).requires_grad_()
    Q = (torch.randn(D, 1) * 0.02).requires_grad_()
    opt = torch.optim.Adam([a, P, Q], lr=0.05, weight_decay=1e-3)
    for _ in range(250):
        loss = 0.0
        for hi, Hc, dist, rec in s:
            lg = a * rec + (hi @ P) @ (Hc @ Q).T
            loss = loss - (dist * F.log_softmax(lg, -1)).sum()
        (loss / len(s)).backward(); opt.step(); opt.zero_grad()
    SEL[hd] = (a.detach(), P.detach()[:, 0], Q.detach()[:, 0])
print(f"  fit {len(SEL)}/{len(HEADS)} heads")


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = x.norm() * y.norm()
    return (x @ y / d).item() if d > 1e-8 else 0.0


# ---- causal edit on held-out chunks ----
torch.manual_seed(1)
STAT = {hd: {"r_true": [], "r_read": [], "r_rand": [], "mtop": [], "mtop_rand": []}
        for hd in SEL}
for seq in TEST:
    n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
    with torch.no_grad():
        model(torch.tensor([seq]).to(DEV))
    pos = torch.arange(n).unsqueeze(0).to(DEV)
    cos, sin = model.gpt_neox.rotary_emb(resid[LAYERS[0]].float(), pos)
    cos, sin = cos[0], sin[0]
    for (l, h) in SEL:
        a, P, Q = SEL[(l, h)]
        P, Q = P.to(DEV), Q.to(DEV)
        Ph = P / P.norm()
        H = resid[l][0].float()
        k, layer = keys_of(H, l, h, cos, sin)
        proj = H @ Ph
        delta = ALPHA * proj.std().item()
        rnd = torch.randn(D, device=DEV); rnd = rnd / rnd.norm()
        for i in range(10, n):
            cand = [j for j in ent if j < i][-KCAND:]
            if len(cand) < 3:
                continue
            base = edited_row(H[i], l, h, i, k, layer, cos, sin)
            if base[cand].sum().item() < 0.10:
                continue
            ci = torch.tensor(cand, device=DEV)
            s = (H[ci] @ Q)
            def cand_dist(vec):
                d = vec[ci]
                return d / d.sum().clamp(min=1e-8)
            dp = cand_dist(edited_row(H[i] + delta * Ph, l, h, i, k, layer, cos, sin))
            dm = cand_dist(edited_row(H[i] - delta * Ph, l, h, i, k, layer, cos, sin))
            dpr = cand_dist(edited_row(H[i] + delta * rnd, l, h, i, k, layer, cos, sin))
            dmr = cand_dist(edited_row(H[i] - delta * rnd, l, h, i, k, layer, cos, sin))
            rec = torch.tensor([-math.log(i - j) for j in cand], device=DEV)
            def read_dist(scale):
                lg = a.to(DEV) * rec + (proj[i] + scale) * P.norm() * s
                return F.softmax(lg, -1)
            drp, drm = read_dist(delta), read_dist(-delta)
            top = int(s.argmax())
            STAT[(l, h)]["r_true"].append(pearson(dp - dm, s))
            STAT[(l, h)]["r_read"].append(pearson(drp - drm, s))
            STAT[(l, h)]["r_rand"].append(pearson(dpr - dmr, s))
            STAT[(l, h)]["mtop"].append((dp - dm)[top].item())
            STAT[(l, h)]["mtop_rand"].append((dpr - dmr)[top].item())
    gc.collect()


def mean(v): return sum(v) / len(v) if v else float("nan")
print(f"\nedit = {ALPHA} std along P-hat; N samples per head in brackets")
print(f"{'head':>9}{'n':>6}{'r_true':>9}{'r_read':>9}{'r_rand':>9}"
      f"{'massTop':>9}{'mTopRnd':>9}")
agg = collections.defaultdict(list)
for hd in SEL:
    st = STAT[hd]
    if not st["r_true"]:
        continue
    row = {k: mean(v) for k, v in st.items()}
    for k, v in row.items():
        agg[k].append(v)
    print(f"{str(hd):>9}{len(st['r_true']):>6}{row['r_true']:>9.3f}"
          f"{row['r_read']:>9.3f}{row['r_rand']:>9.3f}"
          f"{row['mtop']:>9.3f}{row['mtop_rand']:>9.3f}", flush=True)
print(f"{'POOLED':>9}{'':>6}{mean(agg['r_true']):>9.3f}{mean(agg['r_read']):>9.3f}"
      f"{mean(agg['r_rand']):>9.3f}{mean(agg['mtop']):>9.3f}{mean(agg['mtop_rand']):>9.3f}")

# ---- bootstrap CIs ----
import random
random.seed(0)

def cluster_ci(head_means, n=5000):
    k = len(head_means)
    ms = sorted(sum(head_means[random.randrange(k)] for _ in range(k)) / k
                for _ in range(n))
    return mean(head_means), ms[int(.025 * n)], ms[int(.975 * n)]

def sample_ci(per_head_lists, n=5000):
    pool = [x for hd in per_head_lists for x in hd]
    m = len(pool)
    ms = sorted(sum(pool[random.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return mean(pool), ms[int(.025 * n)], ms[int(.975 * n)]

gap_heads = [t - r for t, r in zip(agg["r_true"], agg["r_rand"])]
print(f"\nbootstrap 95% CIs ({len(agg['r_true'])} heads):")
for name, hv in [("r_true", agg["r_true"]), ("r_rand", agg["r_rand"]),
                 ("gap(true-rand)", gap_heads)]:
    c, lo, hi = cluster_ci(hv)
    print(f"  {name:>16}  cluster {c:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for name, key in [("r_true", "r_true"), ("r_rand", "r_rand")]:
    c, lo, hi = sample_ci([STAT[hd][key] for hd in SEL if STAT[hd]["r_true"]])
    print(f"  {name:>16}  sample  {c:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("  gap CI excludes 0 => effect real above the random-direction control.")

# --- small-n tests (cluster bootstrap with n=6 is anti-conservative) ---
import itertools
k = len(gap_heads)
npos = sum(1 for d in gap_heads if d > 0)
# exact sign test (two-sided): prob of >= npos same-sign under fair coin
from math import comb
tail = sum(comb(k, j) for j in range(npos, k + 1)) / 2**k
sign_p = min(1.0, 2 * tail)
# exact sign-flip permutation test on per-head gaps (H0: symmetric about 0)
obs = abs(sum(gap_heads) / k)
perm = [abs(sum(s * d for s, d in zip(signs, gap_heads)) / k)
        for signs in itertools.product([1, -1], repeat=k)]
perm_p = sum(1 for m in perm if m >= obs - 1e-12) / len(perm)
print(f"\nsmall-n tests on {k} per-head gaps ({npos}/{k} positive):")
print(f"  sign test (two-sided) p = {sign_p:.4f}")
print(f"  exact sign-flip permutation p = {perm_p:.4f}  (2^{k}={2**k} assignments)")
for hk in hooks:
    hk.remove()
