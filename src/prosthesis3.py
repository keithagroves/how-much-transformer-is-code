"""Chapter 2, step 5c: prosthesis v3 -- substitute at the SITE.

Logit-level substitution recovered verbatim copying (93%) but 0% of the
circuit's natural-text value: downstream layers consume the heads' output.
So v3 replaces each circuit head's output VECTOR in place:

    rule:   cand[i] = follower of most recent longest suffix match (order o)
    synth:  head_out[l,h][i] ~= M[l,h] @ [embed(cand[i]); onehot(o); 1]
    write:  hook o_proj input, overwrite the head's 128-d slice, let the
            rest of the network run unchanged

M is ridge-fitted per head on intact activations from train chunks (natural +
repeated random so all orders appear), then FROZEN. Controls:
    zero    = plain ablation (floor)
    mean    = replace with each head's constant mean output vector
              (recovery here = activation statistics, not content)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, T, LAM = 8, 1000, 5.0

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
DH = model.config.head_dim
V = model.config.vocab_size
EMB = model.model.embed_tokens.weight.detach().float().cpu()

s = torch.load("head_scores.pt")
HEADS = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.2]
LAYERS = sorted({l for l, _ in HEADS})
print(f"substituting {len(HEADS)} heads across layers {LAYERS}")

raw = open("ministral_corpus.txt").read()
chunks = lambda offs: [tokz.encode(raw[o:o + 12000])[:T] for o in offs]
train_seqs = chunks([0, 50000, 100000])
test_seq = chunks([200000])[0]
torch.manual_seed(1)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist())
train_seqs += [mkrnd(), mkrnd(), mkrnd()]
rnd_test = [mkrnd(), mkrnd(), mkrnd()]

def rule(seq):
    """per position i: (order, most-recent follower of longest match) or (0,-1)"""
    out = []
    for i in range(len(seq)):
        best = (0, -1)
        for o in range(min(MAXO, i + 1), 0, -1):
            key = tuple(seq[i - o + 1 : i + 1])
            for j in range(i - 1, o - 2, -1):
                if tuple(seq[j - o + 1 : j + 1]) == key: best = (o, seq[j + 1]); break
            if best[0]: break
        out.append(best)
    return out

def features(seq):
    R = rule(seq)
    X = torch.zeros(len(seq), EMB.shape[1] + MAXO + 1)
    fire = torch.zeros(len(seq), dtype=torch.bool)
    for i, (o, c) in enumerate(R):
        if o == 0: continue
        X[i, : EMB.shape[1]] = EMB[c]
        X[i, EMB.shape[1] + o - 1] = 1.0
        X[i, -1] = 1.0
        fire[i] = True
    return X, fire

# ---------- capture intact head outputs on train ----------
def capture(seq):
    rec = {}
    hooks = []
    for l in LAYERS:
        def hook(mod, inp, l=l):
            rec[l] = inp[0][0].detach().float().cpu()
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        model(torch.tensor([seq]).to(DEV))
    for hk in hooks: hk.remove()
    return rec

print("capturing intact activations + fitting ridge maps...")
Xs, fires, recs = [], [], []
for sq in train_seqs:
    X, f = features(sq)
    Xs.append(X); fires.append(f); recs.append(capture(sq))
Xall = torch.cat([x[f] for x, f in zip(Xs, fires)])
print(f"  {Xall.shape[0]} firing train positions")

XtX = Xall.T @ Xall + LAM * torch.eye(Xall.shape[1])
M = {}          # (l,h) -> [F,128] map
MEAN = {}       # (l,h) -> [128] mean output (control)
for l, h in HEADS:
    Y = torch.cat([r[l][:, h*DH:(h+1)*DH][f] for r, f in zip(recs, fires)])
    M[(l, h)] = torch.linalg.solve(XtX, Xall.T @ Y)
    MEAN[(l, h)] = torch.cat([r[l][:, h*DH:(h+1)*DH] for r in recs]).mean(0)

# ---------- run with substitution ----------
def run_sub(seq, mode):
    """mode: 'intact' | 'zero' | 'mean' | 'synth'"""
    if mode != "intact":
        X, fire = features(seq)
        pred = {}
        for l, h in HEADS:
            if mode == "synth":
                p = X @ M[(l, h)]
                p[~fire] = 0
            elif mode == "mean":
                p = MEAN[(l, h)].expand(len(seq), DH).clone()
            else:
                p = torch.zeros(len(seq), DH)
            pred[(l, h)] = p.to(DEV)
    hooks = []
    if mode != "intact":
        for l in LAYERS:
            hs = [h for ll, h in HEADS if ll == l]
            def hook(mod, inp, l=l, hs=hs):
                x = inp[0].clone()
                for h in hs: x[0, :, h*DH:(h+1)*DH] = pred[(l, h)]
                return (x,) + inp[1:]
            hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

for name, seqs in [("train natural", [train_seqs[0]]), ("held-out natural", [test_seq]),
                   ("repeated random", rnd_test)]:
    nll = {m: sum(run_sub(sq, m) for sq in seqs) / len(seqs)
           for m in ("intact", "zero", "mean", "synth")}
    gap = nll["zero"] - nll["intact"]
    print(f"{name:<18} intact {nll['intact']:.3f}  zero {nll['zero']:.3f}  "
          f"mean {nll['mean']:.3f} ({(nll['zero']-nll['mean'])/gap:.0%})  "
          f"synth {nll['synth']:.3f} ({(nll['zero']-nll['synth'])/gap:.0%} recovered)")
