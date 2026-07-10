"""Chapter 2, step 2: causal test on Qwen3-0.6B. Zero-ablate the candidate
induction heads; second-half loss on repeated random sequences should jump.
Ablating an equal number of random heads should not.

Qwen3 attention output is a per-head concat [B,T,n_heads*head_dim] -> o_proj.
Zeroing a head's slice of that concat removes its contribution.
"""
import torch
from transformers import AutoModelForCausalLM

torch.manual_seed(0)
MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L, BATCH = 50, 8

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
V = model.config.vocab_size
DH = model.config.head_dim

seq = torch.randint(1000, V - 1000, (BATCH, L))
ids = torch.cat([seq, seq], dim=1).to(DEV)

INDUCTION = [(16, 14), (21, 8), (24, 6), (3, 10), (20, 14), (18, 5)]
RANDOM = [(2, 4), (8, 8), (12, 3), (22, 7), (26, 0), (10, 2)]

def second_half_loss(heads_off):
    hooks = []
    by_layer = {}
    for l, h in heads_off: by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def hook(mod, inp, hs=hs):
            x = inp[0].clone()                     # input to o_proj: [B,T,H*DH]
            for h in hs: x[..., h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        logits = model(ids).logits[:, :-1]
    for hk in hooks: hk.remove()
    lp = torch.log_softmax(logits.float(), -1)
    nll = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return nll[:, L-1:].mean().item(), nll[:, :L-1].mean().item()

for name, off in [("intact", []), ("ablate 6 induction heads", INDUCTION),
                  ("ablate 6 random heads", RANDOM)]:
    s, f = second_half_loss(off)
    print(f"{name:<28} second-half loss {s:6.2f}   (first half {f:.2f})")
