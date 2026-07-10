"""Reviewer #4 (part 2): does COREFERENCE-PROPER name the head's entity selection,
once the sink is removed? Uses a real coref model (fastcoref), not literal token
match. Rule "coref": at query i, find the coref cluster of the referent currently
in focus (the mention at i, else the most recent mention before i), and predict
the most recent entity candidate in that same cluster — i.e. attend to the last
name-mention of whoever is being talked about now. Same sink gate as probe_coref2.
    usage: python3 probe_coref_proper.py [model] [KCAND]
"""
import collections, gc, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastcoref import FCoref

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
KCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 320
coref = FCoref(device="cpu")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
print(f"{MODEL}  KCAND={KCAND}  (sink-controlled + coreference-proper)")

torch.manual_seed(0)
sq = torch.randint(1000, V - 1000, (8, 50)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
qp = torch.arange(50, 99); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - 49].mean(dim=(0, 2))
del out; gc.collect()
HEADS = [(int(x) // NH, int(x) % NH) for x in ind.flatten().topk(6).indices]
print(f"heads: {HEADS}")

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)

OFFS = (540000, 580000, 620000, 660000, 700000)
texts = [raw[o:o + 6000] for o in OFFS]
encs = [tokz(t, return_offsets_mapping=True, add_special_tokens=False) for t in texts]
preds = coref.predict(texts=[t[:e["offset_mapping"][:T_C][-1][1] + 5] for t, e in zip(texts, encs)])

CHUNKS = []
for text, enc, pr in zip(texts, encs, preds):
    ids_seq, offs = enc["input_ids"][:T_C], enc["offset_mapping"][:T_C]
    clus = [-1] * len(ids_seq)
    for ci, cl in enumerate(pr.get_clusters(as_strings=False)):
        for (sa, sb) in cl:
            for k, (a, b) in enumerate(offs):
                if a < sb and sa < b: clus[k] = ci
    CHUNKS.append((ids_seq, clus))

HIT = {hd: collections.defaultdict(float) for hd in HEADS}
N = {hd: 0 for hd in HEADS}
COV = {hd: 0 for hd in HEADS}

for ids_seq, clus in CHUNKS:
    seq = ids_seq; n = len(seq)
    entset = set(j for j in range(n) if is_entity(seq[j]))
    # referent in focus at each position: current cluster else last-seen cluster
    focus = [-1] * n; last = -1
    for k in range(n):
        if clus[k] != -1: last = clus[k]
        focus[k] = last
    with torch.no_grad():
        o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    for (l, h) in HEADS:
        A = o.attentions[l][0, h].float().cpu()
        for i in range(10, n):
            cand = [j for j in range(1, i) if j in entset][-KCAND:]
            if len(cand) < 3: continue
            row = A[i].clone(); row[0] = 0.0
            s = row[1:i + 1].sum().item()
            if s < 1e-6: continue
            row = row / s
            if sum(row[j].item() for j in cand) < 0.15: continue
            truth = max(range(len(cand)), key=lambda k: row[cand[k]].item())
            rec = max(range(len(cand)), key=lambda k: cand[k])
            cur = focus[i]
            same = [k for k in range(len(cand)) if cur != -1 and clus[cand[k]] == cur]
            m = HIT[(l, h)]
            m["recency"] += int(rec == truth)
            m["random"] += 1.0 / len(cand)
            if same:
                cf = max(same, key=lambda k: cand[k])   # most recent same-referent mention
                m["coref"] += int(cf == truth)
                COV[(l, h)] += 1
            N[(l, h)] += 1
    del o; gc.collect()

print(f"\ntop-1 agreement with head's context-selected entity (sink-controlled)")
print(f"{'head':>9}{'n':>6}{'random':>8}{'recency':>8}{'coref':>8}{'cfCov':>7}")
agg = collections.defaultdict(list)
for hd in HEADS:
    if not N[hd]: continue
    rnd = HIT[hd]["random"] / N[hd]; rc = HIT[hd]["recency"] / N[hd]
    cf = HIT[hd]["coref"] / COV[hd] if COV[hd] else float("nan")
    cov = COV[hd] / N[hd]
    agg["random"].append(rnd); agg["recency"].append(rc); agg["cov"].append(cov)
    if COV[hd]: agg["coref"].append(cf)
    print(f"{str(hd):>9}{N[hd]:>6}{rnd:>8.3f}{rc:>8.3f}{cf:>8.3f}{cov:>7.2f}", flush=True)
mean = lambda v: sum(v) / len(v) if v else float("nan")
print(f"{'POOLED':>9}{'':>6}{mean(agg['random']):>8.3f}{mean(agg['recency']):>8.3f}"
      f"{mean(agg['coref']):>8.3f}{mean(agg['cov']):>7.2f}")
print("\nread: coref >> recency AND >> random => coreference names the selection"
      "\n      (residue shrinks). coref ~ recency/random => it does not.")
