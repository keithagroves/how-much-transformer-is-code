"""Cross-family replication of the FIRST-ENTITY PRIMACY rule (probe_coref.py):
does the induction head select the earliest-introduced entity on Pythia too?

Lean, attention-only (no residual hook, no rank-1 fit): for each induction head,
take its TRUE attention over entity candidates, and check which nameable rule's
top pick matches the head's argmax-attended entity.
    usage: python3 probe_coref_x.py [EleutherAI/pythia-410m | Qwen/Qwen3-0.6B] [KCAND]
"""
import collections, gc, math, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-410m"
KCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L = 320, 50

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
print(f"{MODEL}  {NL}L x {NH}H  KCAND={KCAND}")

torch.manual_seed(0)
sq = torch.randint(1000, V - 1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
qp = torch.arange(L, 2 * L - 1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - L + 1].mean(dim=(0, 2))
del out; gc.collect()
HEADS = [(int(x) // NH, int(x) % NH) for x in ind.flatten().topk(6).indices]
print(f"heads: {HEADS}")

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)
def chunk(o): return tokz.encode(raw[o:o + 6000])[:T_C]
TEST = [chunk(o) for o in (540000, 580000, 620000, 660000, 700000)]

RULES = ["recency", "earliest", "salience"]
HIT = {hd: {r: 0.0 for r in RULES + ["random"]} for hd in HEADS}
N = {hd: 0 for hd in HEADS}

for seq in TEST:
    n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
    with torch.no_grad():
        o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    seen = collections.Counter(); freq_at = []
    for i in range(n):
        freq_at.append(seen.copy())
        if is_entity(seq[i]): seen[seq[i]] += 1
    for (l, h) in HEADS:
        A = o.attentions[l][0, h].float()
        for i in range(10, n):
            cand = [j for j in ent if j < i][-KCAND:]
            if len(cand) < 3: continue
            w = A[i, cand]
            if w.sum() < 0.15: continue
            truth = int(w.argmax())
            fq = freq_at[i]
            rec = max(range(len(cand)), key=lambda k: cand[k])
            ear = min(range(len(cand)), key=lambda k: cand[k])
            sal = max(range(len(cand)), key=lambda k: (fq.get(seq[cand[k]], 0), cand[k]))
            picks = {"recency": rec, "earliest": ear, "salience": sal}
            for r, p in picks.items():
                HIT[(l, h)][r] += int(p == truth)
            HIT[(l, h)]["random"] += 1.0 / len(cand)
            N[(l, h)] += 1
    del o; gc.collect()

print(f"\ntop-1 agreement with head's selected entity")
print(f"{'head':>9}{'n':>6}" + "".join(f"{r:>10}" for r in ["random"] + RULES))
agg = collections.defaultdict(list)
for hd in HEADS:
    if not N[hd]: continue
    row = {r: HIT[hd][r] / N[hd] for r in ["random"] + RULES}
    for r, v in row.items(): agg[r].append(v)
    print(f"{str(hd):>9}{N[hd]:>6}" + "".join(f"{row[r]:>10.3f}" for r in ["random"] + RULES),
          flush=True)
mean = lambda v: sum(v) / len(v)
print(f"{'POOLED':>9}{'':>6}" + "".join(f"{mean(agg[r]):>10.3f}" for r in ["random"] + RULES))
