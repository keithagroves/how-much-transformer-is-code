"""Reviewer #4 (part 1): does the CURRENT-CLAUSE SUBJECT name the head's entity
selection, once the attention sink is removed? Scored through the same sink gate
as probe_coref2 (BOS zeroed, attention renormalized over real context, argmax
over entity candidates = the head's actual pick), with recency/random baselines.

spaCy dependency parse -> clause subjects (nsubj/nsubjpass); their char spans are
mapped onto the model's BPE token positions. Rule "clauseSubj" predicts the most
recent entity candidate that is a subject token (the current grammatical agent);
coverage = fraction of scored positions where such a candidate exists.
    usage: python3 probe_syntax.py [model] [KCAND]
"""
import collections, gc, sys, torch, spacy
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
KCAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C = 320
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size
print(f"{MODEL}  {NL}L x {NH}H  KCAND={KCAND}  (sink-controlled + clause-subject)")

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

def prep(o):
    """Return (ids[:T_C], subj_mask[:T_C]) with BPE positions aligned to spaCy subjects."""
    text = raw[o:o + 6000]
    enc = tokz(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"][:T_C], enc["offset_mapping"][:T_C]
    doc = nlp(text[:offs[-1][1] + 5])
    spans = [(t.idx, t.idx + len(t)) for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    subj = [False] * len(ids)
    for k, (a, b) in enumerate(offs):
        for (sa, sb) in spans:
            if a < sb and sa < b:          # char-span overlap
                subj[k] = True; break
    return ids, subj

CHUNKS = [prep(o) for o in (540000, 580000, 620000, 660000, 700000)]

RULES = ["recency", "clauseSubj"]
HIT = {hd: collections.defaultdict(float) for hd in HEADS}
N = {hd: 0 for hd in HEADS}
COV = {hd: 0 for hd in HEADS}

for ids_seq, subj in CHUNKS:
    seq = ids_seq; n = len(seq)
    entset = set(j for j in range(n) if is_entity(seq[j]))
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
            subj_c = [k for k in range(len(cand)) if subj[cand[k]]]
            m = HIT[(l, h)]
            m["recency"] += int(rec == truth)
            m["random"] += 1.0 / len(cand)
            if subj_c:                       # clause-subject rule defined here
                cs = max(subj_c, key=lambda k: cand[k])   # most recent subject entity
                m["clauseSubj"] += int(cs == truth)
                COV[(l, h)] += 1
            N[(l, h)] += 1
    del o; gc.collect()

print(f"\ntop-1 agreement with head's context-selected entity (sink-controlled)")
print(f"{'head':>9}{'n':>6}{'random':>8}{'recency':>8}{'clauseSubj':>11}{'csCov':>7}")
agg = collections.defaultdict(list)
for hd in HEADS:
    if not N[hd]: continue
    rnd = HIT[hd]["random"] / N[hd]; rc = HIT[hd]["recency"] / N[hd]
    cs = HIT[hd]["clauseSubj"] / COV[hd] if COV[hd] else float("nan")
    cov = COV[hd] / N[hd]
    agg["random"].append(rnd); agg["recency"].append(rc)
    if COV[hd]: agg["cs"].append(cs)
    agg["cov"].append(cov)
    print(f"{str(hd):>9}{N[hd]:>6}{rnd:>8.3f}{rc:>8.3f}{cs:>11.3f}{cov:>7.2f}", flush=True)
mean = lambda v: sum(v) / len(v) if v else float("nan")
print(f"{'POOLED':>9}{'':>6}{mean(agg['random']):>8.3f}{mean(agg['recency']):>8.3f}"
      f"{mean(agg['cs']):>11.3f}{mean(agg['cov']):>7.2f}")
print("\nread: clauseSubj >> recency AND >> random => current grammatical subject names the"
      "\n      selection. clauseSubj ~ recency/random => it does not (residue stays smooth).")
