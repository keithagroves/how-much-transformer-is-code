"""Scale check 3: is per-head importance concentrated in a THIN TAIL at 4B, and
is that tail the induction heads? Zero-ablation cost per head (the ablation
analog of the chapter-2 template-substitution cost curve), all heads, held-out.
    usage: python3 arch_heads.py [Qwen/Qwen3-4B] [bf16|fp32] [NCHUNK]
"""
import sys, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B"
DTYPE = torch.bfloat16 if (len(sys.argv) > 2 and sys.argv[2] == "bf16") else torch.float32
NCHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 2
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 256

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=DTYPE).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim * cfg.num_attention_heads
DH = cfg.head_dim
print(f"{MODEL}  {NL}L x {NH}H  head_dim={DH}  dtype={DTYPE}")

raw = open("ministral_corpus.txt").read()

# induction scores (to check tail overlap)
torch.manual_seed(0); L = 50
sq = torch.randint(1000, cfg.vocab_size - 1000, (4, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): out = model(ids, output_attentions=True)
qp = torch.arange(L, 2 * L - 1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - L + 1].mean(dim=(0, 2))
del out; gc.collect(); torch.mps.empty_cache()

EVAL = [tokz.encode(raw[o:o + 7000])[:T_C] for o in (540000, 620000, 700000)][:NCHUNK]

def nll(seq, abl=None):
    hk = []
    if abl is not None:
        l, h = abl
        def oh(mod, inp, l=l, h=h):
            x = inp[0].clone(); x[0, :, h * DH:(h + 1) * DH] = 0
            return (x,) + inp[1:]
        hk.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(oh))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for x in hk: x.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

ni = [nll(s) for s in EVAL]
print(f"intact held-out NLL: {sum(ni)/len(ni):.3f}  ({NCHUNK} chunks); ablating {NL*NH} heads...")
cost = {}
for l in range(NL):
    for h in range(NH):
        cost[(l, h)] = sum(nll(s, (l, h)) - ni[k] for k, s in enumerate(EVAL)) / NCHUNK
    if l % 6 == 5: print(f"  ...through layer {l}", flush=True)

vals = torch.tensor(sorted(cost.values()))
import numpy as np
q = np.percentile(vals.numpy(), [50, 75, 90, 95, 99, 100])
print(f"\nper-head zero-ablation cost (nats), {NL*NH} heads:")
print(f"  median {q[0]:.4f}  p75 {q[1]:.4f}  p90 {q[2]:.4f}  p95 {q[3]:.4f}  p99 {q[4]:.4f}  max {q[5]:.3f}")
print(f"  near-free (<=0.01): {int((vals<=0.01).sum())}/{NL*NH} "
      f"({100*float((vals<=0.01).float().mean()):.0f}%)")
top = sorted(cost, key=lambda k: cost[k], reverse=True)[:12]
print(f"\ntop-12 most costly heads (cost | induction score):")
for (l, h) in top:
    print(f"  L{l}H{h}: {cost[(l,h)]:+.3f}  ind={ind[l,h]:.2f}")
n_top = 12
ind_in_top = sum(ind[l, h] > 0.2 for (l, h) in top)
print(f"\nof the top-{n_top} costly heads, {ind_in_top} are induction heads (score>0.2) "
      f"-> tail {'IS' if ind_in_top >= n_top//2 else 'is NOT'} induction-dominated")
torch.save({"cost": cost, "ind": ind}, "arch_heads_4b.pt")
