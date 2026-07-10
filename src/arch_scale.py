"""Scale shape-replication: does the chapter-2 structure hold at a bigger model?
Runs the two cheapest headline checks, model-parametric, bf16-friendly:
  (1) induction heads exist (repeated-random behavioral collapse + per-head score)
  (2) MLP U-map: token-lookup solo cost per MLP layer -- cheap middle, expensive ends?

    usage: python3 arch_scale.py [Qwen/Qwen3-4B] [bf16|fp32]
"""
import sys, gc, collections, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
DTYPE = torch.bfloat16 if (len(sys.argv) > 2 and sys.argv[2] == "bf16") else torch.float32
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 256

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=DTYPE).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
print(f"{MODEL}  {NL}L x {NH}H  hidden={cfg.hidden_size}  dtype={DTYPE}")

raw = open("ministral_corpus.txt").read()

# ---------- (1) induction ----------
torch.manual_seed(0)
L = 50
sq = torch.randint(1000, V - 1000, (4, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    tgt = ids[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    first = nll[:, :L - 1].mean().item(); second = nll[:, L:].mean().item()
qp = torch.arange(L, 2 * L - 1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - L + 1].mean(dim=(0, 2))
del out; gc.collect()
top = [(int(x) // NH, int(x) % NH, float(ind.flatten()[x])) for x in ind.flatten().topk(8).indices]
print(f"\n[induction] repeated-seq loss: first half {first:.2f} -> second half {second:.2f} "
      f"(collapse {first - second:.2f} nats)")
print(f"[induction] top heads (score): " + ", ".join(f"L{l}H{h}={s:.2f}" for l, h, s in top))
print(f"[induction] heads with score>0.2: {(ind > 0.2).sum().item()} / {NL * NH}")

# ---------- (2) MLP U-map ----------
def chunk(o): return tokz.encode(raw[o:o + 7000])[:T_C]
TRAIN = [chunk(o) for o in range(0, 400000, 50000)]        # 8 fit chunks
TEST = [chunk(o) for o in (540000, 600000, 660000)]

print("\n[mlp] fitting per-token lookup tables for all layers...")
SUM = [collections.defaultdict(lambda: None) for _ in range(NL)]
CNT = [collections.Counter() for _ in range(NL)]
TOT = [None] * NL; NTOK = 0
cap = {}
hooks = [model.model.layers[l].mlp.register_forward_hook(
    (lambda mod, inp, outp, l=l: cap.__setitem__(l, (outp[0] if isinstance(outp, tuple) else outp)[0].detach().float().cpu())))
    for l in range(NL)]
with torch.no_grad():
    for sq in TRAIN:
        model(torch.tensor([sq]).to(DEV))
        for l in range(NL):
            o = cap[l]
            TOT[l] = o.sum(0) if TOT[l] is None else TOT[l] + o.sum(0)
            for i, t in enumerate(sq):
                s = SUM[l][t]
                SUM[l][t] = o[i].clone() if s is None else s + o[i]
                CNT[l][t] += 1
        NTOK += len(sq)
for hk in hooks: hk.remove()
MEAN = [TOT[l] / NTOK for l in range(NL)]
LUT = [{t: SUM[l][t] / CNT[l][t] for t in SUM[l]} for l in range(NL)]

def nll_of(seq, sub_layer=None):
    hk = []
    if sub_layer is not None:
        lut = LUT[sub_layer]; mean = MEAN[sub_layer]
        rep = torch.stack([lut.get(t, mean) for t in seq]).to(DEV).to(DTYPE)
        def mhook(mod, inp, outp, rep=rep):
            return rep.unsqueeze(0) if not isinstance(outp, tuple) else (rep.unsqueeze(0),) + outp[1:]
        hk.append(model.model.layers[sub_layer].mlp.register_forward_hook(mhook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for h in hk: h.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

ni = [nll_of(s) for s in TEST]
print(f"[mlp] intact held-out NLL: {sum(ni) / len(ni):.3f}")
print(f"[mlp] token-lookup solo cost per layer (nats):")
costs = []
for l in range(NL):
    c = sum(nll_of(s, l) - ni[k] for k, s in enumerate(TEST)) / len(TEST)
    costs.append(c)
    print(f"  L{l:2d}: {c:+.3f}", flush=True)
lo = sorted(range(NL), key=lambda l: costs[l])[:5]
hi = sorted(range(NL), key=lambda l: costs[l], reverse=True)[:5]
print(f"[mlp] cheapest layers: {[(l, round(costs[l],3)) for l in lo]}")
print(f"[mlp] most expensive:  {[(l, round(costs[l],3)) for l in hi]}")
print(f"[mlp] free-or-negative (<=0.01): {sum(c <= 0.01 for c in costs)} / {NL}")
print(f"\nU-map check: are the expensive layers the ENDS (near 0 and {NL-1}) and the "
      f"cheap ones the MIDDLE?")
