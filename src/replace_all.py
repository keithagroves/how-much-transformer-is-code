"""Chapter 2, step 8: replace as MANY heads as possible with code.

Every attention head gets approximated by a code template -- a fixed mixture
of nameable columns:
    BOS sink | self | offsets 1..16 (positional profile) | last punctuation
    before i | fuzzy-match rule columns (induction)
Mixture weights per head = that head's measured mean attention mass on each
column type (train sequence, one forward pass). At eval the template is built
BY CODE on the unseen sequence (renormalized rows), and every head whose
explained mass >= threshold is substituted; the rest of the network runs
unchanged.

Deliverable: curve of #heads-replaced vs held-out NLL, with zero-ablation of
the same heads as control. QK of a replaced head = retired into code.
"""
import torch, collections
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
MAXO, T, NOFF = 8, 900, 16

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
DH, NH, NKV, NL = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.num_hidden_layers
GROUP = NH // NKV
V = cfg.vocab_size

import os as _os
raw = open(_os.environ.get("SUB_CORPUS", "ministral_corpus.txt")).read()
_CAL = int(_os.environ.get("SUB_CALIB_OFFSET", "0"))
train_seq = tokz.encode(raw[_CAL:_CAL + 11000])[:T]
test_seq = tokz.encode(raw[200000:211000])[:T]
torch.manual_seed(3)
rnd_test = (lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist())

PUNCT = {tokz.encode(p)[0] for p in (".", "!", "?", ",", ";", ":")} | \
        {t for t in range(200) if "\n" in tokz.decode([t])}

# ---------- fuzzy match rule (from fuzzy.py, self-contained here) ----------
EMB = model.model.embed_tokens.weight.detach().float().cpu()
def fuzzy_cols(seq, sim_th=0.55):
    uniq = sorted(set(seq)); idx = {t: k for k, t in enumerate(uniq)}
    E = EMB[torch.tensor(uniq)]; E = E / E.norm(dim=-1, keepdim=True)
    S = (E @ E.T).clamp(min=0)
    norm = [tokz.decode([t]).strip().lower().strip('.,"\'') for t in uniq]
    bystr = {}
    for k, s_ in enumerate(norm): bystr.setdefault(s_, []).append(k)
    for ks in bystr.values():
        for a in ks:
            for b in ks:
                if a != b: S[a, b] = max(S[a, b], torch.tensor(0.9))
    S.fill_diagonal_(1.0); S[S < sim_th] = 0
    ids = [idx[t] for t in seq]
    n = len(seq)
    A = torch.zeros(n, n)
    for i in range(1, n):
        for j in range(1, i + 1):
            s0 = S[ids[i], ids[j - 1]]
            if s0 <= 0: continue
            st, k = float(s0), 1
            while k < MAXO and i - k >= 0 and j - 1 - k >= 0:
                s = S[ids[i - k], ids[j - 1 - k]]
                if s <= 0: break
                st += float(s); k += 1
            A[i, j] = st
    # softmax-ish: exp(strength), rows normalized later inside template
    M = (A > 0).float() * A.exp()
    Z = M.sum(-1, keepdim=True).clamp(min=1e-9)
    return M / Z, (A > 0).float()

def template_columns(seq):
    """returns dict of named [T,T] column masks/profiles built BY CODE."""
    n = len(seq)
    idx = torch.arange(n)
    cols = {}
    e = torch.zeros(n, n); e[:, 0] = 1; cols["bos"] = e
    e = torch.zeros(n, n); e[idx, idx] = 1; cols["self"] = e
    for d in range(1, NOFF + 1):
        e = torch.zeros(n, n); e[idx[d:], idx[d:] - d] = 1; cols[f"off{d}"] = e
    lp = torch.zeros(n, n); last = -1
    for i in range(n):
        if last >= 0: lp[i, last] = 1
        if seq[i] in PUNCT: last = i
    cols["punct"] = lp
    rule_soft, rule_mask = fuzzy_cols(seq)
    cols["rule"] = rule_soft          # already row-normalized soft distribution
    return cols, rule_mask

# ---------- measure per-head weights on train ----------
print("measuring head profiles on train sequence...")
with torch.no_grad():
    out = model(torch.tensor([train_seq]).to(DEV), output_attentions=True)
atts = [a[0].float().cpu() for a in out.attentions]     # NL x [NH,T,T]
del out
cols_tr, rmask_tr = template_columns(train_seq)
names = list(cols_tr.keys())
mask_tr = {k: (v if k != "rule" else rmask_tr) for k, v in cols_tr.items()}

W = {}          # (l,h) -> {name: weight}, explained
rows = slice(50, T)
overlap = torch.zeros(len(names), T - 50, T)
for ki, k in enumerate(names): overlap[ki] = mask_tr[k][rows]
claimed = overlap.cumsum(0).clamp(max=1)                # avoid double-counting
excl = torch.cat([overlap[:1], (overlap[1:] * (1 - claimed[:-1]))])
for l in range(NL):
    for h in range(NH):
        a = atts[l][h][rows]
        w = {k: float((a * excl[ki]).sum(-1).mean()) for ki, k in enumerate(names)}
        W[(l, h)] = (w, sum(w.values()))
expl = torch.tensor([W[(l, h)][1] for l in range(NL) for h in range(NH)])
print(f"explained mass: median {expl.median():.2f}, >=0.9: {(expl>=0.9).sum()}, "
      f">=0.8: {(expl>=0.8).sum()}, >=0.7: {(expl>=0.7).sum()} of {NL*NH} heads")

# ---------- build code attention for a sequence ----------
def code_attn(seq):
    cols, _ = template_columns(seq)
    base = {k: v / v.sum(-1, keepdim=True).clamp(min=1e-9) for k, v in cols.items()}
    return base

def run_sub(seq, heads, mode, base=None):
    if base is None and mode == "code": base = code_attn(seq)
    n = len(seq)
    A = {}
    if mode == "code":
        for (l, h) in heads:
            w, tot = W[(l, h)]
            M = torch.zeros(n, n)
            for k, wk in w.items(): M += wk * base[k]
            A[(l, h)] = (M / M.sum(-1, keepdim=True).clamp(min=1e-9)).to(DEV)
    by_layer = {}
    for l, h in heads: by_layer.setdefault(l, []).append(h)
    vcache, hooks = {}, []
    for l, hs in by_layer.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                if mode == "code":
                    x[0, :, h*DH:(h+1)*DH] = A[(l, h)] @ vcache[l][:, g*DH:(g+1)*DH]
                else:
                    x[0, :, h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    base_test = code_attn(test_seq)
    base_rnd = code_attn(rnd_test)
    ni_nat = run_sub(test_seq, [], "code")
    ni_rnd = run_sub(rnd_test, [], "code")
    print(f"\nintact: natural {ni_nat:.3f}  repeated-random {ni_rnd:.3f}")
    print(f"{'thresh':>7}{'heads':>7}{'nat code':>10}{'nat zero':>10}{'rnd code':>10}")
    order = sorted(W, key=lambda k: -W[k][1])
    for th in (0.95, 0.9, 0.85, 0.8, 0.7):
        heads = [k for k in order if W[k][1] >= th]
        if not heads: continue
        nc = run_sub(test_seq, heads, "code", base_test)
        nz = run_sub(test_seq, heads, "zero")
        nr = run_sub(rnd_test, heads, "code", base_rnd)
        print(f"{th:>7}{len(heads):>7}{nc:>10.3f}{nz:>10.3f}{nr:>10.3f}")
