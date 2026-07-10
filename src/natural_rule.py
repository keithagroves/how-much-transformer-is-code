"""Chapter 2, step 3: is the induction circuit literally our copy rule?

The rule, in this project's language:
    RULE: if current token appeared before, predict the token that followed
          its most recent occurrence.   (an order-2 lookback rule)

On NATURAL text, bucket every position by the rule:
    A. rule fires and its prediction is CORRECT (next token really is the copy)
    B. rule fires but its prediction is WRONG
    C. rule cannot fire (token never seen before in context)

Then ablate the induction heads (ind score > 0.2) and measure damage
(delta log-prob of the true next token) per bucket. If the heads implement
the rule, damage should concentrate in bucket A.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
DH = model.config.head_dim

s = torch.load("head_scores.pt")
INDUCTION = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.2]
print(f"ablating {len(INDUCTION)} heads with ind score > 0.2")

TEXT = open("structured_corpus.txt").read()[:6000] + " " + """
The Dursleys had everything they wanted, but they also had a secret, and
their greatest fear was that somebody would discover it. They didn't think
they could bear it if anyone found out about the Potters. Mrs Potter was
Mrs Dursley's sister, but they hadn't met for several years; in fact,
Mrs Dursley pretended she didn't have a sister, because her sister and her
good-for-nothing husband were as unDursleyish as it was possible to be.
""" * 3

ids = torch.tensor([tokz.encode(TEXT)[:1000]]).to(DEV)
T = ids.shape[1]

def logprobs(off):
    hooks = []
    by_layer = {}
    for l, h in off: by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        def hook(mod, inp, hs=hs):
            x = inp[0].clone()
            for h in hs: x[..., h*DH:(h+1)*DH] = 0
            return (x,) + inp[1:]
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
    for hk in hooks: hk.remove()
    return lp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1).cpu()   # [T-1]

base, abl = logprobs([]), logprobs(INDUCTION)
dmg = base - abl

seq = ids[0].tolist()
last = {}
buckets = {"A rule fires, correct": [], "B rule fires, wrong": [], "C rule can't fire": []}
for i in range(1, T - 1):                       # predicting token i+1 from pos i
    cur, nxt = seq[i], seq[i + 1]
    if cur in last:
        pred = seq[last[cur] + 1]
        buckets["A rule fires, correct" if pred == nxt else "B rule fires, wrong"].append(i)
    else:
        buckets["C rule can't fire"].append(i)
    last[seq[i - 1]] = i - 1                    # only positions strictly before i are visible

print(f"\n{'bucket':<24}{'n':>6}{'mean damage (nats)':>22}")
for name, pos in buckets.items():
    d = dmg[torch.tensor(pos)] if pos else torch.tensor([0.0])
    print(f"{name:<24}{len(pos):>6}{d.mean().item():>22.3f}")

a, b = len(buckets["A rule fires, correct"]), len(buckets["B rule fires, wrong"])
print(f"\ncopy-rule accuracy when it fires: {a}/{a+b} = {a/(a+b):.0%}")
