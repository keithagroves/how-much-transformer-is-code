"""Sanity check: reconstruct an induction head's TRUE attention row from the
captured residual-stream input + the real head weights (q_proj/k_proj, q_norm/
k_norm, RoPE, GQA), and confirm it matches model(output_attentions=True).

If this matches, we can edit the query residual for a single head/position and
recompute its real softmax attention analytically -- the isolated intervention a
forward hook cannot do. This file exists only to validate the machinery used by
probe_queryedit.py.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH, NKV = cfg.head_dim, cfg.num_key_value_heads
GROUP = NH // NKV
EPS = cfg.rms_norm_eps
SCALE = DH ** -0.5


def rms(x, w):                                    # per-head RMSNorm over last dim
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    a, b = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


def head_attn(H, l, h, cos, sin, q_override=None, qpos=None):
    """True attention rows for head (l,h) given residual H (n,D), the layer INPUT
    (pre input_layernorm). If q_override (n,D) given, use it to form THIS head's
    query (keys unchanged). Returns softmax attention (n,n), causal."""
    layer = model.model.layers[l]
    att = layer.self_attn
    n = H.shape[0]
    g = h // GROUP
    Wq = att.q_proj.weight[h * DH:(h + 1) * DH]          # (DH, D)
    Wk = att.k_proj.weight[g * DH:(g + 1) * DH]
    ln = layer.input_layernorm.weight
    Hn = rms(H, ln)                                       # attention sees post-norm
    Hqn = Hn if q_override is None else rms(q_override, ln)
    q = rms(Hqn @ Wq.T, att.q_norm.weight)               # (n, DH)
    k = rms(Hn @ Wk.T, att.k_norm.weight)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    logits = (q @ k.T) * SCALE                           # (n, n)
    mask = torch.triu(torch.full((n, n), float("-inf"), device=H.device), 1)
    return torch.softmax(logits + mask, -1)


if __name__ == "__main__":
    resid = {}
    hooks = [model.model.layers[l].register_forward_pre_hook(
        (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())),
        with_kwargs=True) for l in range(NL)]

    raw = open("ministral_corpus.txt").read()
    seq = tokz.encode(raw[40000:46000])[:200]
    ids = torch.tensor([seq]).to(DEV)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    pos = torch.arange(len(seq)).unsqueeze(0).to(DEV)
    cos, sin = model.model.rotary_emb(resid[0].float(), pos)
    cos, sin = cos[0], sin[0]                              # (n, DH)

    worst = 0.0
    for (l, h) in [(0, 0), (3, 5), (10, 8), (20, 3), (27, 15)]:
        H = resid[l][0].float()
        rec = head_attn(H, l, h, cos, sin)
        true = out.attentions[l][0, h].float()
        err = (rec - true).abs().max().item()
        worst = max(worst, err)
        print(f"head ({l:2d},{h:2d})  max|recon-true| = {err:.2e}")
    print(f"\nworst reconstruction error: {worst:.2e}  "
          f"{'OK' if worst < 1e-3 else 'FAIL -- do not trust the edit'}")
    for hk in hooks:
        hk.remove()
