"""Nested replacement-DSL frontier for attention heads (forward-only).

Measures the marginal value of expanding the allowed forward-pass instruction
set at identical component budgets:
  D0 DELETE      : no replacement function (zero head output)
  D1 POSITIONAL  : BOS, self, and fixed offsets 1..16
  D2 STRUCTURAL  : D1 plus punctuation/sentence/line structure
  D3 CONTENT     : D2 plus duplicate-token and induction-follower rules

The head set and fitted coefficients are held fixed.  Only the permitted
instructions change.  Results are held-out attention-only damage versus intact.
"""
import csv
import torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt")
rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS_ALL = sorted(nat_c, key=lambda k: nr[k] + rr[k])

POSITIONAL = {"bos", "self"} | {f"off{i}" for i in range(1, 17)}
STRUCTURAL = POSITIONAL | {"punct", "psent", "sent", "sstart", "lstart"}
CONTENT = STRUCTURAL | {"dup", "rule"}
DSLS = {
    "D0_delete": set(),
    "D1_positional": POSITIONAL,
    "D2_structural": STRUCTURAL,
    "D3_content": CONTENT,
}

raw = open("ministral_corpus.txt").read()
starts = [o for o in range(40000, min(len(raw)-10000, 1000000), 80000)
          if not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o+8000])[:300] for o in starts]

HOLD = {"A": None, "zero": False}
vcache, hooks = {}, []
for l in sorted({l for l, _ in HEADS_ALL}):
    attn = model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(
        lambda m, i, o, l=l: vcache.__setitem__(l, o[0])))
    def ohook(mod, inp, l=l):
        if HOLD["A"] is None or l not in HOLD["A"]:
            return None
        x = inp[0].clone()
        for h, A in HOLD["A"][l]:
            g = h // GROUP
            if HOLD["zero"]:
                x[0, :, h*DH:(h+1)*DH] = 0
            else:
                x[0, :, h*DH:(h+1)*DH] = A.to(DEV) @ vcache[l][:, g*DH:(g+1)*DH]
        return (x,) + inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(ohook))

print("compiling shared instruction columns...", flush=True)
BASE = [RR.code_attn(seq) for seq in eval_chunks]

def program(seq, base, heads, allowed):
    """Compile the permitted DSL instructions into per-head attention."""
    n = len(seq); by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    out = {}
    for l, hs in by_layer.items():
        out[l] = []
        for h in hs:
            M = torch.zeros(n, n)
            for name, weight in RR.W[(l, h)].items():
                if name in allowed and weight > 1e-4:
                    M += weight * base[name]
            if M.sum() == 0:
                M[:, 0] = 1
            out[l].append((h, M / M.sum(-1, keepdim=True).clamp(min=1e-9)))
    return out

def loss(seq, A=None, zero=False):
    HOLD["A"], HOLD["zero"] = A, zero
    with torch.no_grad():
        logits = model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        value = -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
    HOLD["A"], HOLD["zero"] = None, False
    return value

intact = sum(loss(seq) for seq in eval_chunks) / len(eval_chunks)
print(f"intact {intact:.4f}")
print(f"{'heads':>6} {'DSL':<15} {'vocab':>5} {'scalars':>8} {'damage':>9} {'gain_vs_delete':>15}")
rows = []
for k in [40, 80, 120, 160, 200, 256]:
    heads = HEADS_ALL[:k]
    # A placeholder program identifies the head slices for zero ablation.
    A0 = [program(seq, base, heads, POSITIONAL) for seq, base in zip(eval_chunks, BASE)]
    delete_damage = sum(loss(seq, A, zero=True) for seq, A in zip(eval_chunks, A0)) / len(eval_chunks) - intact
    rows.append((k, "D0_delete", 0, 0, 0, delete_damage, 0.0))
    print(f"{k:>6} {'D0_delete':<15} {0:>5} {0:>8} {delete_damage:>+9.3f} {0.0:>+15.3f}", flush=True)
    for level, (name, allowed) in enumerate(list(DSLS.items())[1:], start=1):
        programs = [program(seq, base, heads, allowed) for seq, base in zip(eval_chunks, BASE)]
        damage = sum(loss(seq, A) for seq, A in zip(eval_chunks, programs)) / len(eval_chunks) - intact
        gain = delete_damage - damage
        scalars = sum(1 for head in heads for key, value in RR.W[head].items()
                      if key in allowed and value > 1e-4)
        rows.append((k, name, level, len(allowed), scalars, damage, gain))
        print(f"{k:>6} {name:<15} {len(allowed):>5} {scalars:>8} {damage:>+9.3f} {gain:>+15.3f}", flush=True)

for hook in hooks:
    hook.remove()
with open("dsl_frontier_results.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["heads", "dsl", "level", "vocabulary_size", "fitted_scalars",
                "damage_nats", "gain_vs_delete_nats"])
    w.writerows(rows)
print("wrote dsl_frontier_results.csv")
