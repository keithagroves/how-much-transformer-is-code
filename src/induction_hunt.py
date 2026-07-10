"""Chapter 2, step 1: hunt induction heads (model-agnostic).

Setup: sequences of L random tokens repeated twice. Nothing in the second
half is predictable from grammar -- the ONLY way to predict token i is to
find where it appeared before and copy what followed. Any head/circuit that
does that is an induction mechanism.

Measures:
  1. behavioral: per-position loss; second half should collapse
  2. per-head induction score: attention from pos i to pos i-L+1
     (the token AFTER the previous occurrence of the current token)
  3. per-head previous-token score: attention to pos i-1
     (the expected partner: prev-token heads feed induction heads via K-composition)

  usage: python3 induction_hunt.py [model]   (default Qwen/Qwen3-0.6B)
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L, BATCH = 50, 8

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
V = model.config.vocab_size
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
print(f"{MODEL}: {n_layers} layers x {n_heads} heads")

# random tokens (avoid special ids at the top of the vocab), repeated twice
seq = torch.randint(1000, min(V, len(tokz)) - 1000, (BATCH, L))
ids = torch.cat([seq, seq], dim=1).to(DEV)          # [B, 2L]

with torch.no_grad():
    out = model(ids, output_attentions=True)
logits = out.logits[:, :-1]
targets = ids[:, 1:]
lp = torch.log_softmax(logits.float(), -1)
nll = -lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [B, 2L-1]

first = nll[:, : L - 1].mean().item()
second = nll[:, L - 1 :].mean().item()   # from pos L-1 on, target is a repeat
print(f"behavioral: loss first half {first:.2f}  second half {second:.2f}  "
      f"(drop {first - second:.2f} nats)")

# ---- per-head scores ----
ind = torch.zeros(n_layers, n_heads)
prev = torch.zeros(n_layers, n_heads)
qpos = torch.arange(L, 2 * L - 1)                    # query positions in 2nd half
for l, att in enumerate(out.attentions):             # att: [B, H, T, T]
    a = att.float().cpu()
    ind[l] = a[:, :, qpos, qpos - L + 1].mean(dim=(0, 2))   # attend to "after prev occurrence"
    prev[l] = a[:, :, 1:, :].diagonal(offset=-1, dim1=2, dim2=3).mean(dim=(0, 2))

def top(mat, k=10):
    v, i = mat.flatten().topk(k)
    return [(int(x) // n_heads, int(x) % n_heads, float(s)) for x, s in zip(i, v)]

print("\ntop induction heads (layer.head : score = mean attn to i-L+1):")
for l, h, s in top(ind):
    print(f"  L{l}.H{h}  {s:.3f}")
print("\ntop previous-token heads (mean attn to i-1):")
for l, h, s in top(prev):
    print(f"  L{l}.H{h}  {s:.3f}")

torch.save({"ind": ind, "prev": prev, "model": MODEL}, "head_scores.pt")
print("\nsaved head_scores.pt")
