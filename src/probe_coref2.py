"""Which-entity selection, sink-controlled. After primacy was unmasked as the
attention sink (probe_primacy_control.py), the honest question remains: once the
sink is removed, does ANY nameable rule predict which entity an induction head
selects, or is the selection genuinely smooth?

Method: zero out BOS and renormalize each attention row over the real context
(positions 1..i); keep only positions with >=0.15 of that CONTEXT mass on entity
candidates. Selected entity = argmax over candidates. Score top-1 agreement of:
  recency    most recent entity
  earliest   first entity in window     (the retracted primacy rule / sink proxy)
  salience   most-mentioned entity so far
  corefRepeat most recent RE-mention (entity seen before)   -- discourse tracking
  corefTok   most recent prior mention of the CURRENT token -- literal coreference
             (scored only where defined; coverage reported)
  random     1/n_candidates
If corefRepeat/corefTok clearly beat recency AND the (now sink-free) earliest
baseline, the selection has a name. If nothing beats recency/random, it is smooth.
    usage: python3 probe_coref2.py [model] [KCAND]
"""
import collections, gc, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
KCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L = 320, 50

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
print(f"{MODEL}  {NL}L x {NH}H  KCAND={KCAND}  (BOS removed, context-renormalized)")

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

RULES = ["recency", "earliest", "salience", "corefRepeat", "corefTok"]
HIT = {hd: collections.defaultdict(float) for hd in HEADS}
N = {hd: 0 for hd in HEADS}
COV = {hd: 0 for hd in HEADS}                 # positions where corefTok is defined

for seq in TEST:
    n = len(seq)
    entset = set(j for j in range(n) if is_entity(seq[j]))
    seen = collections.Counter(); freq_at = []
    for i in range(n):
        freq_at.append(seen.copy())
        if i in entset: seen[seq[i]] += 1
    with torch.no_grad():
        o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    for (l, h) in HEADS:
        A = o.attentions[l][0, h].float().cpu()
        for i in range(10, n):
            cand = [j for j in range(1, i) if j in entset][-KCAND:]
            if len(cand) < 3: continue
            row = A[i].clone(); row[0] = 0.0            # remove sink
            s = row[1:i + 1].sum().item()
            if s < 1e-6: continue
            row = row / s
            cmass = sum(row[j].item() for j in cand)
            if cmass < 0.15: continue                    # genuine context entity attn
            truth = max(range(len(cand)), key=lambda k: row[cand[k]].item())
            fq = freq_at[i]
            picks = {
                "recency":  max(range(len(cand)), key=lambda k: cand[k]),
                "earliest": min(range(len(cand)), key=lambda k: cand[k]),
                "salience": max(range(len(cand)), key=lambda k: (fq.get(seq[cand[k]], 0), cand[k])),
            }
            rep = [k for k in range(len(cand)) if fq.get(seq[cand[k]], 0) >= 1]
            picks["corefRepeat"] = max(rep, key=lambda k: cand[k]) if rep else picks["recency"]
            same = [k for k in range(len(cand)) if seq[cand[k]] == seq[i]]
            m = HIT[(l, h)]
            for r in ["recency", "earliest", "salience", "corefRepeat"]:
                m[r] += int(picks[r] == truth)
            m["random"] += 1.0 / len(cand)
            if same:                                     # corefTok defined here only
                m["corefTok"] += int(max(same, key=lambda k: cand[k]) == truth)
                COV[(l, h)] += 1
            N[(l, h)] += 1
    del o; gc.collect()

print(f"\ntop-1 agreement with head's context-selected entity")
print(f"{'head':>9}{'n':>6}{'random':>8}{'recency':>8}{'earliest':>9}{'salien':>8}"
      f"{'cRepeat':>8}{'cTok':>7}{'cTokCov':>8}")
agg = collections.defaultdict(list)
for hd in HEADS:
    if not N[hd]: continue
    v = {r: HIT[hd][r] / N[hd] for r in ["random"] + RULES[:-1]}
    ctok = HIT[hd]["corefTok"] / COV[hd] if COV[hd] else float("nan")
    cov = COV[hd] / N[hd]
    for r, x in v.items(): agg[r].append(x)
    if COV[hd]: agg["corefTok"].append(ctok)
    agg["cov"].append(cov)
    print(f"{str(hd):>9}{N[hd]:>6}{v['random']:>8.3f}{v['recency']:>8.3f}"
          f"{v['earliest']:>9.3f}{v['salience']:>8.3f}{v['corefRepeat']:>8.3f}"
          f"{ctok:>7.3f}{cov:>8.2f}", flush=True)
mean = lambda x: sum(x) / len(x) if x else float("nan")
print(f"{'POOLED':>9}{'':>6}{mean(agg['random']):>8.3f}{mean(agg['recency']):>8.3f}"
      f"{mean(agg['earliest']):>9.3f}{mean(agg['salience']):>8.3f}"
      f"{mean(agg['corefRepeat']):>8.3f}{mean(agg['corefTok']):>7.3f}{mean(agg['cov']):>8.2f}")
print("\nread: a rule >> recency AND >> earliest(now sink-free) => selection has a name.")
print("      nothing beats recency/random => the which-entity selection is smooth.")
