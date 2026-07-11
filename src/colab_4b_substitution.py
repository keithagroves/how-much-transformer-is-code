"""Self-contained substitution + heal, model-parametric (Qwen3-0.6B to validate,
Qwen3-4B on Colab). Answers the thesis-critical question missing at scale: is the
+0.6-0.9 nat healed-substitution band a small-model fact?

POSITIONAL-ONLY templates (content-free: offsets, bos, punctuation, sentence/line
structure) — justified because the paper's positional-vs-fitted sweep shows content
columns buy ~nothing in the substitutable set, so positional-only ties fitted code.

Pipeline (fast, no per-head forward scan): fit each head a positional template by
clipped least squares to its true attention; SELECT the heads best reconstructed
by positional templates (that IS codability) up to the budget; replace them + the
cheap MLP layers (token lookup tables); heal only the RMSNorm gains; report healed
damage vs intact, with zero-ablation and intact-heal (domain-adaptation) controls.
The shared DSL columns are cached per chunk (not one matrix per head), so healing
is cheap enough for 4B.

Colab: MODEL='Qwen/Qwen3-4B', Runtime = L4/A100 (24GB+).
  pip install -q transformers datasets accelerate
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, math, gc, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
FRAC_H = float(sys.argv[2]) if len(sys.argv) > 2 else 0.36
FRAC_M = 0.21
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SMOKE  = os.environ.get("SMOKE", "0") == "1"
DEV    = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE  = torch.bfloat16 if DEV == "cuda" else torch.float32
NOFF   = 8
print(f"model={MODEL} device={DEV} dtype={DTYPE} head-budget={FRAC_H:.0%} epochs={EPOCHS}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", dtype=DTYPE).to(DEV).eval()
tokz  = AutoTokenizer.from_pretrained(MODEL)
cfg   = model.config
NL, NH, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH    = getattr(cfg, "head_dim", HD // NH)
NKV   = getattr(cfg, "num_key_value_heads", NH)
GROUP = NH // NKV
V     = cfg.vocab_size
layers = model.model.layers

def get_wikitext():
    from itertools import islice
    last = None
    for repo in ["Salesforce/wikitext", "wikitext", "iohadrubin/wikitext-103-raw-v1"]:
        try:
            ds = load_dataset(repo, "wikitext-103-raw-v1", split="train", streaming=True)
            buf, n = [], 0
            for r in islice(ds, 200000):
                t = r.get("text", "")
                if t and t.strip():
                    buf.append(t); n += len(t)
                    if n > 1_400_000: break
            if n > 500_000:
                print(f"loaded corpus from {repo} ({n} chars)", flush=True)
                return "\n".join(buf)
        except Exception as e:
            last = e; print(f"  dataset repo {repo} failed: {type(e).__name__}", flush=True)
    raise RuntimeError(f"could not load wikitext ({last})")
text = open("ministral_corpus.txt").read() if SMOKE and os.path.exists("ministral_corpus.txt") else get_wikitext()
def chunks(a, b, step, n): return [tokz.encode(text[o:o+6000])[:n] for o in range(a, b, step)]
T_TR = 64 if SMOKE else (384 if DEV == "cuda" else 512)  # shorter train seqs on GPU for activation memory
T_EV = 64 if SMOKE else 256
TRAIN = [c for c in chunks(0, 700000, 28000, T_TR) if len(c) == T_TR][:(2 if SMOKE else 16)]
VAL   = [c for c in chunks(705000, 755000, 25000, T_EV) if len(c) == T_EV][:(1 if SMOKE else 2)]
EVAL  = [c for c in chunks(760000, 1200000, 40000, T_EV) if len(c) == T_EV][:(1 if SMOKE else 8)]
print(f"train {len(TRAIN)}, validation {len(VAL)}, test {len(EVAL)} chunks", flush=True)

SENT = {t for t in range(min(400, V)) if any(c in tokz.decode([t]) for c in ".!?\n")}
LINE = {t for t in range(min(400, V)) if "\n" in tokz.decode([t])}

def pos_columns(seq):
    n = len(seq); cols = {}
    bos = torch.zeros(n, n); bos[:, 0] = 1; cols["bos"] = bos
    cols["self"] = torch.eye(n)
    for k in range(1, NOFF + 1):
        c = torch.zeros(n, n)
        for i in range(k, n): c[i, i - k] = 1
        cols[f"off{k}"] = c
    punct = torch.zeros(n, n); last = 0
    for i in range(n):
        punct[i, last] = 1
        if seq[i] in SENT: last = i
    cols["punct"] = punct
    sid = []; s = 0
    for i in range(n):
        sid.append(s)
        if seq[i] in SENT: s += 1
    lid = []; l = 0
    for i in range(n):
        lid.append(l)
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

# ---- fit a positional template per head + its reconstruction residual (codability) ----
print("fitting positional templates + selecting by codability ...", flush=True)
fitseq = TRAIN[0]
with torch.no_grad():
    out = model(torch.tensor([fitseq]).to(DEV), output_attentions=True)
A_fit = [a[0].float().cpu() for a in out.attentions]; del out; gc.collect()
cols = pos_columns(fitseq); C = torch.stack([cols[k] for k in COLNAMES])   # [K,n,n]
Xc = C[:, 1:, :].reshape(len(COLNAMES), -1).T                              # [(n-1)*n, K]
W = {}; resid = {}
for l in range(NL):
    for h in range(NH):
        tgt = A_fit[l][h]
        yc = tgt[1:, :].reshape(-1)
        w = torch.linalg.lstsq(Xc, yc).solution.clamp(min=0)
        if w.sum() < 1e-6: w = torch.zeros(len(COLNAMES)); w[0] = 1.0
        W[(l, h)] = w
        recon = sum(w[k] * cols[COLNAMES[k]] for k in range(len(COLNAMES)))
        recon = recon / recon.sum(-1, keepdim=True).clamp(min=1e-9)
        resid[(l, h)] = (recon - tgt).abs().mean().item()      # lower = more positional/codable
del A_fit; gc.collect()
K_HEADS = int(FRAC_H * NL * NH)
HEADS = sorted(resid, key=lambda k: resid[k])[:K_HEADS]        # best-reconstructed = substitutable
BY_L = {}
for l, h in HEADS: BY_L.setdefault(l, []).append(h)
print(f"selected {K_HEADS} best-reconstructed heads ({FRAC_H:.0%}); "
      f"median residual kept {sorted(resid.values())[K_HEADS]:.4f}", flush=True)

W_BY_L = {l: torch.stack([W[(l, h)] for h in hs]).to(torch.float32)
          for l, hs in BY_L.items()}
def columns_for(seq):
    cols = pos_columns(seq)
    return torch.stack([cols[k] for k in COLNAMES]).to(torch.float16)
print("caching shared DSL columns per chunk ...", flush=True)
CC_TRAIN = [columns_for(sq) for sq in TRAIN]
CC_VAL   = [columns_for(sq) for sq in VAL]
CC_EVAL  = [columns_for(sq) for sq in EVAL]

# ---- MLP lookup tables on the cheapest layers (cheap: NL zero-ablation probes) ----
def intact_loss(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
seq0 = VAL[0]; b0 = intact_loss(seq0); mlp_cost = {}
for l in range(NL):
    hz = layers[l].mlp.register_forward_hook(lambda m,i,o: torch.zeros_like(o))
    mlp_cost[l] = intact_loss(seq0) - b0; hz.remove()
K_MLP = max(1, int(FRAC_M * NL))
MLPS = sorted(mlp_cost, key=lambda l: mlp_cost[l])[:K_MLP]
print(f"selected {K_MLP} cheapest MLP layers ({FRAC_M:.0%}): {sorted(MLPS)}", flush=True)

print("building MLP lookup tables ...", flush=True)
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
LMAT_TRAIN = [{l: torch.stack([LUT[l].get(t, MEAN[l]) for t in sq_]).half() for l in MLPS} for sq_ in TRAIN]  # CPU
LMAT_VAL   = [{l: torch.stack([LUT[l].get(t, MEAN[l]) for t in sq_]).half() for l in MLPS} for sq_ in VAL]
LMAT_EVAL  = [{l: torch.stack([LUT[l].get(t, MEAN[l]) for t in sq_]).half() for l in MLPS} for sq_ in EVAL]

# ---- install hooks once; mode in {"code","zero","off"}; templates+LUTs cached ----
HOLD = {"C": None, "L": None, "mode": "off"}
VC = {}
hooks = []
for l in set(BY_L):
    attn = layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m,i,o,l=l: VC.__setitem__(l, o[0])))
    def oh(mod, inp, l=l):
        if HOLD["mode"] == "off": return None
        x = inp[0].clone()
        mats = None
        if HOLD["mode"] == "code":
            C = HOLD["C"].to(x.device, torch.float32)
            mats = torch.einsum("hk,knm->hnm", W_BY_L[l].to(x.device), C)
            mats = mats / mats.sum(-1, keepdim=True).clamp(min=1e-9)
        for mi, h in enumerate(BY_L[l]):
            g = h // GROUP
            if HOLD["mode"] == "zero": x[0,:,h*DH:(h+1)*DH] = 0
            else: x[0,:,h*DH:(h+1)*DH] = (mats[mi] @ VC[l][:, g*DH:(g+1)*DH].float()).to(x.dtype)
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(oh))
for l in MLPS:
    def mh(mod, i, o, l=l):
        if HOLD["mode"] == "off": return None
        if HOLD["mode"] == "zero": return torch.zeros_like(o)
        return HOLD["L"][l].unsqueeze(0).to(o.device, o.dtype)
    hooks.append(layers[l].mlp.register_forward_hook(mh))

def loss_seq(sq_, C, L, mode):
    HOLD["mode"], HOLD["C"], HOLD["L"] = mode, C, L
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([sq_]).to(DEV)).logits[0,:-1].float(), -1)
    return -lp.gather(-1, torch.tensor(sq_[1:]).to(DEV).unsqueeze(-1)).mean().item()
def run_split(seqs, columns, lmats, mode):
    return sum(loss_seq(sq_, columns[i], lmats[i], mode) for i, sq_ in enumerate(seqs)) / len(seqs)
def run(mode): return run_split(EVAL, CC_EVAL, LMAT_EVAL, mode)
def validate(mode): return run_split(VAL, CC_VAL, LMAT_VAL, mode)

norm = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm]
# Freeze every matrix and embedding. Gradients still pass through frozen modules
# to earlier norm gains, but weight-gradient buffers are never allocated.
for p in model.parameters(): p.requires_grad_(False)
for p in norm: p.requires_grad_(True)
def heal(mode):
    for p, o in zip(norm, orig): p.data.copy_(o)
    opt = torch.optim.Adam(norm, lr=3e-4); model.train()
    if DEV == "cuda":                      # gradient checkpointing: trade compute for memory
        model.config.use_cache = False
        try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception: model.gradient_checkpointing_enable()
    best = (float("inf"), None, -1); stale = 0; patience = 3
    for ep in range(EPOCHS):
        train_total = 0.0
        for i, sq_ in enumerate(TRAIN):
            HOLD["mode"], HOLD["C"], HOLD["L"] = mode, CC_TRAIN[i], LMAT_TRAIN[i]
            ids = torch.tensor([sq_]).to(DEV)
            out = model(ids, labels=ids); opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
            train_total += out.loss.item()
        model.eval(); val = validate(mode); model.train()
        print(f"  {mode} epoch {ep+1:>2}: train {train_total/len(TRAIN):.4f} val {val:.4f}", flush=True)
        if val < best[0] - 1e-4:
            best = (val, [p.detach().cpu().clone() for p in norm], ep); stale = 0
        else:
            stale += 1
            if ep >= 2 and stale >= patience:
                print(f"  {mode}: early stop; best epoch {best[2]+1}", flush=True); break
    if DEV == "cuda":
        model.gradient_checkpointing_disable()
    if best[1] is not None:
        for p, value in zip(norm, best[1]): p.data.copy_(value.to(DEV))
    model.eval()

print("\n==== RESULTS (held-out, nats above intact) ====", flush=True)
intact = run("off")
print(f"intact loss: {intact:.4f}  (perplexity {math.exp(intact):.1f})", flush=True)
code_unhealed = run("code") - intact
print(f"code, unhealed:        {code_unhealed:+.3f}", flush=True)
heal("code"); code_healed = run("code") - intact
print(f"code + heal:           {code_healed:+.3f}  (perplexity {math.exp(intact+code_healed):.1f})", flush=True)
heal("zero"); zero_healed = run("zero") - intact
print(f"zero-ablation + heal:  {zero_healed:+.3f}   <- control: code should be << this", flush=True)
heal("off"); offset = run("off") - intact
for p, o in zip(norm, orig): p.data.copy_(o)
print(f"intact-heal offset:    {offset:+.3f}   (domain adaptation the heal earns for free)", flush=True)
print(f"FAIR cost (code - offset): {code_healed - offset:+.3f}", flush=True)
result = {"model": MODEL, "head_fraction": FRAC_H, "mlp_fraction": FRAC_M,
          "epochs_cap": EPOCHS, "intact_loss": intact,
          "code_unhealed": code_unhealed, "code_healed": code_healed,
          "zero_healed": zero_healed, "intact_heal_offset": offset,
          "fair_code_cost": code_healed-offset, "n_train": len(TRAIN),
          "n_validation": len(VAL), "n_test": len(EVAL)}
safe_model = MODEL.replace("/", "_")
with open(f"scale_result_{safe_model}_{FRAC_H:.2f}.json", "w") as f: json.dump(result, f, indent=2)
print("\n==== HOW TO READ ====")
print("This protocol (WikiText, POSITIONAL-ONLY templates, %d epochs) is NOT the paper's" % EPOCHS)
print("fiction/fitted/20-epoch +0.88 — do not compare directly to +0.88. Instead run this SAME")
print("notebook on the 0.6B model (dropdown) and compare the two numbers head to head:")
print("  * zero+heal should be >> code+heal on both (control: the code carries function)")
print("  * if 4B's code+heal / fair-cost is <= 0.6B's under this identical protocol, the")
print("    substitutability is NOT a small-model artifact — it holds or improves with scale.")
print("  * a much larger 4B cost would mean scale shrinks the substitutable fraction.")
