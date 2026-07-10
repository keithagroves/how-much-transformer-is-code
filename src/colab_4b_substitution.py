"""Self-contained substitution + heal, model-parametric (Qwen3-0.6B locally to
validate, Qwen3-4B on Colab). Answers the thesis-critical question the paper is
missing at scale: is the +0.6-0.9 nat healed-substitution band a small-model fact?

Uses POSITIONAL-ONLY templates (content-free: offsets, bos, punctuation, sentence/
line structure) — justified because the paper's positional-vs-fitted sweep shows
content columns buy ~nothing in the substitutable set, so positional-only ties the
full fitted code. Pipeline: identify cheap heads by solo ablation cost, fit each a
positional template (non-negative least squares to its true attention), replace a
budget-matched fraction + the cheap MLP layers (token lookup tables), heal only the
RMSNorm gains, and report healed damage against intact — plus the zero-ablation and
intact-heal controls.

Colab: set MODEL='Qwen/Qwen3-4B'; Runtime = L4/A100 (24GB+). Uses WikiText-103 (public).
  pip install -q transformers datasets accelerate
"""
import sys, math, gc, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
FRAC_H  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.36   # head budget (paper's 36%)
FRAC_M  = 0.21                                                # MLP budget (paper's 21%)
EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 12
DEV     = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE   = torch.bfloat16 if DEV == "cuda" else torch.float32
NOFF    = 8                                                   # fixed-offset columns off1..off8
print(f"model={MODEL} device={DEV} dtype={DTYPE} head-budget={FRAC_H:.0%} epochs={EPOCHS}")

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=DTYPE).to(DEV).eval()
tokz  = AutoTokenizer.from_pretrained(MODEL)
cfg   = model.config
NL, NH, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH    = getattr(cfg, "head_dim", HD // NH)
NKV   = getattr(cfg, "num_key_value_heads", NH)
GROUP = NH // NKV
V     = cfg.vocab_size
layers = model.model.layers

# ---- data: WikiText-103 (public); natural text, matches the paper's setting ----
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
text = "\n".join(r["text"] for r in ds.select(range(20000)) if r["text"].strip())
def chunks(a, b, step, n): return [tokz.encode(text[o:o+6000])[:n] for o in range(a, b, step)]
TRAIN = [c for c in chunks(0, 700000, 28000, 512) if len(c) == 512][:24]
EVAL  = [c for c in chunks(760000, 1200000, 40000, 300) if len(c) == 300][:8]
print(f"train chunks {len(TRAIN)}, eval chunks {len(EVAL)}")

SENT = {t for t in range(min(400, V)) if any(c in tokz.decode([t]) for c in ".!?\n")}
LINE = {t for t in range(min(400, V)) if "\n" in tokz.decode([t])}

def pos_columns(seq):
    """content-free column basis: bos, self, off1..NOFF, punct, sstart, sent, lstart."""
    n = len(seq); cols = {}
    bos = torch.zeros(n, n); bos[:, 0] = 1; cols["bos"] = bos
    slf = torch.eye(n); cols["self"] = slf
    for k in range(1, NOFF + 1):
        c = torch.zeros(n, n)
        for i in range(k, n): c[i, i - k] = 1
        cols[f"off{k}"] = c
    punct = torch.zeros(n, n); last = 0
    for i in range(n):
        punct[i, last] = 1
        if seq[i] in SENT: last = i
    cols["punct"] = punct
    sid = [0]*n; s = 0
    for i in range(n):
        sid[i] = s
        if seq[i] in SENT: s += 1
    lid = [0]*n; l = 0
    for i in range(n):
        lid[i] = l
        if seq[i] in LINE: l += 1
    fs, fl = {}, {}
    for i in range(n): fs.setdefault(sid[i], i); fl.setdefault(lid[i], i)
    sst = torch.zeros(n, n); lst = torch.zeros(n, n); sent = torch.zeros(n, n)
    for i in range(n):
        sst[i, fs[sid[i]]] = 1; lst[i, fl[lid[i]]] = 1
        for j in range(i + 1):
            if sid[j] == sid[i]: sent[i, j] = 1
    cols["sstart"] = sst; cols["lstart"] = lst; cols["sent"] = sent
    return {k: v / v.sum(-1, keepdim=True).clamp(min=1e-9) for k, v in cols.items()}

COLNAMES = ["bos","self"] + [f"off{k}" for k in range(1, NOFF+1)] + ["punct","sstart","lstart","sent"]

# ---- true attention + values via hooks ----
def true_attn_and_v(seq):
    with torch.no_grad():
        out = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    A = [a[0].float().cpu() for a in out.attentions]     # [NH, n, n] per layer
    del out; gc.collect()
    return A

# ---- induction heads (repeated random) to know where copying lives ----
torch.manual_seed(0)
sq = torch.randint(1000, V-1000, (6, 40)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): o = model(ids, output_attentions=True)
qp = torch.arange(40, 79); ind = torch.zeros(NL, NH)
for l, a in enumerate(o.attentions): ind[l] = a.float().cpu()[:, :, qp, qp-39].mean(dim=(0,2))
del o; gc.collect()

# ---- fit a positional template per head on one train seq (non-neg least squares) ----
print("fitting positional templates per head ...")
fitseq = TRAIN[0]; n = len(fitseq)
cols = pos_columns(fitseq); C = torch.stack([cols[k] for k in COLNAMES])   # [K, n, n]
A_fit = true_attn_and_v(fitseq)
W = {}                                    # (l,h) -> weight vector over COLNAMES
for l in range(NL):
    for h in range(NH):
        tgt = A_fit[l][h]                 # [n,n]
        # regress rows (i>=1) : minimize || sum_k w_k C[k] - tgt ||, w>=0
        Xc = C[:, 1:, :].reshape(len(COLNAMES), -1).T     # [(n-1)*n, K]
        yc = tgt[1:, :].reshape(-1)
        w = torch.linalg.lstsq(Xc, yc).solution.clamp(min=0)
        if w.sum() < 1e-6: w = torch.ones(len(COLNAMES)); w[0] = 3.0   # fallback ~ bos/offsets
        W[(l, h)] = w
del A_fit; gc.collect()

def head_template(seq, l, h):
    cols = pos_columns(seq); w = W[(l, h)]
    M = sum(w[k] * cols[COLNAMES[k]] for k in range(len(COLNAMES)))
    return (M / M.sum(-1, keepdim=True).clamp(min=1e-9)).to(DEV)

# ---- solo ablation cost per head (replace with its template, measure delta) ----
def loss_of(seq, hook_fns):
    hs = [layers[l].self_attn.o_proj.register_forward_pre_hook(fn) for l, fn in hook_fns]
    vh = [layers[l].self_attn.v_proj.register_forward_hook(vfn) for l, vfn in _vhooks(hook_fns)]
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
    for x in hs+vh: x.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

VCACHE = {}
def _vhooks(hook_fns):
    ls = set(l for l,_ in hook_fns)
    return [(l, (lambda m,i,o,l=l: VCACHE.__setitem__(l, o[0].detach()))) for l in ls]

def sub_hook(l, hmap):
    """hmap: {head: template[n,n]}; replaces those heads' attention@V."""
    def fn(mod, inp, l=l, hmap=hmap):
        x = inp[0].clone()
        for h, T in hmap.items():
            g = h // GROUP
            x[0, :, h*DH:(h+1)*DH] = (T @ VCACHE[l][:, g*DH:(g+1)*DH].float()).to(x.dtype)
        return (x,) + inp[1:]
    return fn

def intact_loss(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

print("scoring per-head solo cost ...")
seq0 = EVAL[0]; base0 = intact_loss(seq0)
solo = {}
Tcache0 = {(l,h): head_template(seq0, l, h) for l in range(NL) for h in range(NH)}
for l in range(NL):
    for h in range(NH):
        c = loss_of(seq0, [(l, sub_hook(l, {h: Tcache0[(l,h)]}))]) - base0
        solo[(l, h)] = c
    if l % max(1, NL//6) == 0: print(f"  layer {l}/{NL}", flush=True)
K_HEADS = int(FRAC_H * NL * NH)
HEADS = sorted(solo, key=lambda k: solo[k])[:K_HEADS]
BY_L = {}
for l, h in HEADS: BY_L.setdefault(l, {})[h] = None
print(f"selected {K_HEADS} cheapest heads ({FRAC_H:.0%})")

# ---- MLP lookup tables on the cheapest layers ----
mlp_cost = {}
capM = {}
for l in range(NL):
    hh = layers[l].mlp.register_forward_hook(lambda m,i,o,l=l: capM.__setitem__(l, o))
    b = intact_loss(seq0)
    def zero(m,i,o,l=l): return torch.zeros_like(o)
    hz = layers[l].mlp.register_forward_hook(zero)
    mlp_cost[l] = intact_loss(seq0) - b
    hz.remove(); hh.remove()
K_MLP = max(1, int(FRAC_M * NL))
MLPS = sorted(mlp_cost, key=lambda l: mlp_cost[l])[:K_MLP]
print(f"selected {K_MLP} cheapest MLP layers ({FRAC_M:.0%}): {sorted(MLPS)}")

print("building MLP lookup tables ...")
SUM = {l: {} for l in MLPS}; tot = {l: None for l in MLPS}; cap = {}
hk = [layers[l].mlp.register_forward_hook((lambda m,i,o,l=l: cap.__setitem__(l, o[0].detach().float()))) for l in MLPS]
cnt = 0
with torch.no_grad():
    for sq_ in TRAIN:
        model(torch.tensor([sq_]).to(DEV))
        for l in MLPS:
            oo = cap[l]; tot[l] = oo.sum(0) if tot[l] is None else tot[l] + oo.sum(0)
            for i, t in enumerate(sq_):
                if t in SUM[l]: SUM[l][t][0].add_(oo[i]); SUM[l][t][1] += 1
                else: SUM[l][t] = [oo[i].clone(), 1]
        cnt += len(sq_)
for h in hk: h.remove()
MEAN = {l: tot[l]/cnt for l in MLPS}; LUT = {l: {t: v/nn for t,(v,nn) in SUM[l].items()} for l in MLPS}
def lut_mat(seq, l): return torch.stack([LUT[l].get(t, MEAN[l]) for t in seq]).to(DEV)

# ---- install hybrid (heads+MLPs) / zero / intact, then heal, and evaluate ----
SEQ = {"s": None}
def install(mode):
    hooks = []
    for l in set(BY_L) | set(MLPS):
        if l in BY_L:
            attn = layers[l].self_attn
            hooks.append(attn.v_proj.register_forward_hook(lambda m,i,o,l=l: VCACHE.__setitem__(l, o[0].detach())))
            def oh(mod, inp, l=l):
                x = inp[0].clone()
                for h in BY_L[l]:
                    g = h // GROUP
                    if mode == "zero": x[0,:,h*DH:(h+1)*DH] = 0
                    else:
                        T = head_template(SEQ["s"], l, h)
                        x[0,:,h*DH:(h+1)*DH] = (T @ VCACHE[l][:, g*DH:(g+1)*DH].float()).to(x.dtype)
                return (x,) + inp[1:]
            hooks.append(attn.o_proj.register_forward_pre_hook(oh))
        if l in MLPS and mode != "zero_heads_only":
            def mh(mod, i, o, l=l):
                if mode == "zero": return torch.zeros_like(o)
                return lut_mat(SEQ["s"], l).unsqueeze(0).to(o.dtype)
            hooks.append(layers[l].mlp.register_forward_hook(mh))
    return hooks

def evloss(mode):
    hks = install(mode); tot = 0.0
    for sq_ in EVAL:
        SEQ["s"] = sq_
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq_]).to(DEV)).logits[0,:-1].float(), -1)
        tot += -lp.gather(-1, torch.tensor(sq_[1:]).to(DEV).unsqueeze(-1)).mean().item()
    for h in hks: h.remove()
    return tot / len(EVAL)

norm = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm]
def heal(mode):
    for p, o in zip(norm, orig): p.data.copy_(o)
    for p in norm: p.requires_grad_(True)
    opt = torch.optim.Adam(norm, lr=3e-4); model.train()
    for ep in range(EPOCHS):
        hks = install(mode)
        for sq_ in TRAIN:
            SEQ["s"] = sq_; ids = torch.tensor([sq_]).to(DEV)
            out = model(ids, labels=ids); opt.zero_grad(); out.loss.backward(); opt.step()
        for h in hks: h.remove()
    model.eval()
    d = evloss(mode)
    for p, o in zip(norm, orig): p.data.copy_(o)
    return d

intact = sum(intact_loss(sq_) for sq_ in EVAL)/len(EVAL)
print("\n==== RESULTS (held-out, nats above intact) ====")
print(f"intact loss: {intact:.4f}  (perplexity {math.exp(intact):.1f})")
code_unhealed = evloss("code") - intact
print(f"code, unhealed:        {code_unhealed:+.3f}")
code_healed = heal("code") - intact
print(f"code + heal:           {code_healed:+.3f}  (perplexity {math.exp(intact+code_healed):.1f})")
zero_healed = heal("zero") - intact
print(f"zero-ablation + heal:  {zero_healed:+.3f}   <- control: code should be << this")
# intact-heal offset (domain adaptation) -> fair number
def heal_intact():
    for p, o in zip(norm, orig): p.data.copy_(o)
    for p in norm: p.requires_grad_(True)
    opt = torch.optim.Adam(norm, lr=3e-4); model.train()
    for ep in range(EPOCHS):
        for sq_ in TRAIN:
            ids = torch.tensor([sq_]).to(DEV); out = model(ids, labels=ids)
            opt.zero_grad(); out.loss.backward(); opt.step()
    model.eval(); d = sum(intact_loss(sq_) for sq_ in EVAL)/len(EVAL) - intact
    for p, o in zip(norm, orig): p.data.copy_(o);
    return d
offset = heal_intact()
print(f"intact-heal offset:    {offset:+.3f}   (domain adaptation the heal earns for free)")
print(f"FAIR cost (code - offset): {code_healed - offset:+.3f}  <- compare to 0.6B's ~+0.88")
print("\nread: if code+heal and the fair cost sit in the +0.6-0.9 band, the headline is NOT a")
print("      small-model artifact; if much lower/higher, scale changes the substitutable fraction.")
