"""Primacy confound control: is the head's attention SPECIFICALLY on the first
entity, or a smooth early-position blob (attention sink) where the earliest
entity merely sits? Advisor's #1 concern -- entity vs position.

For each induction head + query position (entity-mass>=0.15, >=3 candidates),
compare the head's real attention on the first-entity position against three
positional baselines:
  a_first    attention on the earliest entity in the window
  a_bos      attention on BOS (position 0) -- the classic sink
  base_local mean attention on NON-entity tokens within +/-8 of that entity
             (same early neighborhood -> isolates content from position)
  base_all   mean attention on all non-entity, non-BOS keys <= i (smooth floor)
and report how often the head's GLOBAL argmax (excluding BOS) lands on an entity
and specifically on the first entity.

Read: a_first >> base_local (spike above same-position non-entities) and high
P(argmax on entity) => entity-anchored, primacy is a content rule. a_first ~
base_local, mass dominated by BOS => positional residue / sink in costume.
    usage: python3 probe_primacy_control.py [model] [KCAND]
"""
import collections, gc, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
KCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
THRESH = int(sys.argv[3]) if len(sys.argv) > 3 else 10   # first entity must be >= this position
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L, BAND = 320, 50, 8

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

# Every metric below is computed on attention RENORMALIZED over the real context
# (positions 1..i, i.e. BOS/sink removed), and only on samples where the first
# entity is genuinely mid-passage (ear >= THRESH). This asks: once the sink is
# set aside and the first entity is NOT near the start, does the head still put
# its attention there (entity primacy) or elsewhere (sink was doing the work)?
M = {hd: collections.defaultdict(float) for hd in HEADS}
N = {hd: 0 for hd in HEADS}
SKIP = {hd: 0 for hd in HEADS}
for seq in TEST:
    n = len(seq)
    entset = set(j for j in range(n) if is_entity(seq[j]))
    with torch.no_grad():
        o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    for (l, h) in HEADS:
        A = o.attentions[l][0, h].float().cpu()
        for i in range(THRESH + 2, n):
            cand = [j for j in range(i) if j in entset][-KCAND:]
            if len(cand) < 3: continue
            ear = min(cand)
            raw_row = A[i]
            if raw_row[cand].sum().item() < 0.15: continue
            if ear < THRESH:                        # first entity too close to sink -> skip
                SKIP[(l, h)] += 1; continue
            row = raw_row.clone(); bos = row[0].item()
            row[0] = 0.0
            s = row[1:i + 1].sum().item()
            if s < 1e-6: continue
            row = row / s                            # renormalize over real context
            nb = [j for j in range(max(1, ear - BAND), min(i, ear + BAND + 1))
                  if j not in entset]
            am = int(row[1:i].argmax()) + 1          # argmax over context (BOS already 0)
            m = M[(l, h)]
            m["bos_raw"] += bos                       # sink share, pre-renorm
            m["a_first"] += row[ear].item()
            m["base_local"] += (sum(row[j].item() for j in nb) / len(nb)) if nb else 0.0
            m["argmax_entity"] += int(am in entset)
            m["argmax_first"] += int(am == ear)
            N[(l, h)] += 1
    del o; gc.collect()

cols = ["bos_raw", "a_first", "base_local", "argmax_entity", "argmax_first"]
print(f"\nrenormalized over context (BOS removed); first entity forced to pos>={THRESH}")
print(f"{'head':>9}{'n':>6}{'skip<thr':>9}{'bosRaw':>8}{'a_first':>9}{'baseLoc':>9}"
      f"{'f/loc':>7}{'amEnt':>7}{'amFirst':>9}")
agg = collections.defaultdict(list)
for hd in HEADS:
    if not N[hd]: continue
    v = {c: M[hd][c] / N[hd] for c in cols}
    ratio = v["a_first"] / v["base_local"] if v["base_local"] > 1e-9 else float("inf")
    for c in cols: agg[c].append(v[c])
    agg["ratio"].append(ratio)
    print(f"{str(hd):>9}{N[hd]:>6}{SKIP[hd]:>9}{v['bos_raw']:>8.2f}{v['a_first']:>9.3f}"
          f"{v['base_local']:>9.4f}{ratio:>7.1f}"
          f"{v['argmax_entity']:>7.2f}{v['argmax_first']:>9.2f}", flush=True)
mean = lambda x: sum(x) / len(x)
print(f"{'POOLED':>9}{'':>6}{'':>9}{mean(agg['bos_raw']):>8.2f}{mean(agg['a_first']):>9.3f}"
      f"{mean(agg['base_local']):>9.4f}{mean(agg['ratio']):>7.1f}"
      f"{mean(agg['argmax_entity']):>7.2f}{mean(agg['argmax_first']):>9.2f}")
print("\namFirst (argmax over context lands on the mid-passage first entity):")
print("  high  => entity-anchored, primacy survives the sink control")
print("  ~0    => sink in costume; the first entity only won by sitting near BOS")
