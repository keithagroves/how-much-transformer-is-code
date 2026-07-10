"""Chapter 2, step 5d: prosthesis v4 -- replace QK with the rule, keep OV.

An attention head factorizes:  QK decides WHERE to look; OV decides WHAT the
attended position writes. v3 showed the write-content can't be synthesized
from raw embeddings. v4 keeps the model's own V (and o_proj) and substitutes
only the learned attention pattern with our code rule:

    j*[i] = position of the follower of the most recent longest suffix match
    head_out[i] = V_head(hidden[j*[i]])          (hard one-hot attention)
    no match -> zero

ZERO fitted parameters. If this recovers the gaps, the induction heads' QK
circuit IS the backoff rule -- the learned part we set out to substitute.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAXO, T = 8, 1000

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
DH, NH, NKV = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
GROUP = NH // NKV
V = cfg.vocab_size

s = torch.load("head_scores.pt")
HEADS = [(l, h) for l in range(28) for h in range(16) if s["ind"][l, h] > 0.2]
LAYERS = sorted({l for l, _ in HEADS})
print(f"v4: rule-attention for {len(HEADS)} heads, OV kept; layers {LAYERS}")

raw = open("ministral_corpus.txt").read()
test_seq = tokz.encode(raw[200000:212000])[:T]
train_seq = tokz.encode(raw[:12000])[:T]          # "train" only in name: nothing is fitted
torch.manual_seed(1)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, V - 1000, (50,)).tolist())
rnd_test = [mkrnd(), mkrnd(), mkrnd()]

def rule_pos(seq):
    """per position i: index of follower of most recent longest match, or -1"""
    out = []
    for i in range(len(seq)):
        best = -1
        for o in range(min(MAXO, i + 1), 0, -1):
            key = tuple(seq[i - o + 1 : i + 1])
            for j in range(i - 1, o - 2, -1):
                if tuple(seq[j - o + 1 : j + 1]) == key: best = j + 1; break
            if best >= 0: break
        out.append(best)
    return torch.tensor(out)

def run(seq, mode):
    """mode: 'intact' | 'zero' | 'rule'"""
    jstar = rule_pos(seq).to(DEV)
    vcache = {}
    hooks = []
    if mode != "intact":
        for l in LAYERS:
            attn = model.model.layers[l].self_attn
            def vhook(mod, inp, outp, l=l):
                vcache[l] = outp[0].detach()          # [T, NKV*DH]
            hooks.append(attn.v_proj.register_forward_hook(vhook))
            hs = [h for ll, h in HEADS if ll == l]
            def ohook(mod, inp, l=l, hs=hs):
                x = inp[0].clone()                    # [1, T, NH*DH]
                v = vcache[l]                         # [T, NKV*DH]
                ok = jstar >= 0
                src = jstar.clamp(min=0)
                for h in hs:
                    g = h // GROUP
                    sub = v[src, g*DH:(g+1)*DH]       # value at rule position
                    sub[~ok] = 0
                    x[0, :, h*DH:(h+1)*DH] = sub if mode == "rule" else 0
                return (x,) + inp[1:]
            hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

for name, seqs in [("natural A", [train_seq]), ("natural B (held-out)", [test_seq]),
                   ("repeated random", rnd_test)]:
    nll = {m: sum(run(sq, m) for sq in seqs) / len(seqs) for m in ("intact", "zero", "rule")}
    gap = nll["zero"] - nll["intact"]
    print(f"{name:<22} intact {nll['intact']:.3f}  zero {nll['zero']:.3f}  "
          f"rule {nll['rule']:.3f}   gap recovered {(nll['zero']-nll['rule'])/gap:.0%}")
