"""Validate attention reconstruction for GPT-NeoX (Pythia), the prerequisite for
the query-edit probe on Pythia. Differences from the Qwen reconstruction:
  - fused query_key_value projection, viewed as [n, NH, 3*DH]
  - PARTIAL rotary: RoPE applies to only the first rotary_ndims (16) of 64 dims,
    the rest pass through unrotated
  - input_layernorm is a LayerNorm (subtract mean, divide std, *weight + bias)
  - no per-head q/k RMSNorm
If max|recon - true| ~ 1e-6, the hand-built QK circuit matches the model and the
edit probe is trustworthy.
"""
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-410m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH = D // NH
RND = int(DH * cfg.rotary_pct)          # rotary_ndims = 16
SCALE = DH ** -0.5
V = cfg.vocab_size


def rot_half(x):
    a, b = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


def apply_rope(x, cos, sin):
    """x: (n, DH). RoPE the first RND dims, pass the rest through."""
    xr, xp = x[..., :RND], x[..., RND:]
    xr = xr * cos + rot_half(xr) * sin
    return torch.cat((xr, xp), -1)


def ln(H, layer):
    lnm = layer.input_layernorm
    mu = H.mean(-1, keepdim=True)
    var = H.var(-1, keepdim=True, unbiased=False)
    return (H - mu) / torch.sqrt(var + lnm.eps) * lnm.weight + lnm.bias


def qkv(H, l):
    """(q,k) RoPE'd per head from residual H (n,D). Returns lists over heads."""
    layer = model.gpt_neox.layers[l]
    Hn = ln(H, layer)
    w = layer.attention.query_key_value
    z = (Hn @ w.weight.T + w.bias).view(-1, NH, 3 * DH)    # [n, NH, 3DH]
    q = z[:, :, :DH]; k = z[:, :, DH:2 * DH]
    return q, k


# ---- forward with true attentions on a real chunk ----
raw = open("ministral_corpus.txt").read()
seq = tokz.encode(raw[40000:46000])[:200]
resid = {}
hooks = [model.gpt_neox.layers[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())),
    with_kwargs=True) for l in range(NL)]
with torch.no_grad():
    out = model(torch.tensor([seq]).to(DEV), output_attentions=True)
n = len(seq)
pos = torch.arange(n).unsqueeze(0).to(DEV)
cos, sin = model.gpt_neox.rotary_emb(resid[0].float(), pos)
cos, sin = cos[0], sin[0]                                  # [n, RND]

worst = 0.0
for l in range(0, NL, 4):
    H = resid[l][0].float()
    q, k = qkv(H, l)
    for h in range(0, NH, 4):
        qh = apply_rope(q[:, h], cos, sin)                # [n, DH]
        kh = apply_rope(k[:, h], cos, sin)
        logits = (qh @ kh.T) * SCALE
        mask = torch.triu(torch.ones(n, n, device=DEV), 1).bool()
        logits = logits.masked_fill(mask, float("-inf"))
        recon = torch.softmax(logits, -1)
        true = out.attentions[l][0, h].float()
        err = (recon - true).abs().max().item()
        worst = max(worst, err)
        if l % 8 == 0 and h == 0:
            print(f"  layer {l:2d} head {h:2d}  max|err| {err:.2e}", flush=True)
print(f"\nWORST max|recon-true| over sampled heads: {worst:.2e}")
print("expect ~1e-6 if reconstruction is correct")
for hk in hooks:
    hk.remove()
