"""Probe, coreference rung: is the head's entity SELECTION nameable as a
linguistic rule (recency / salience / coreference), or does it need the learned
rank-1 direction?

probe_queryedit showed the rank-1 direction Q causally steers which entity the
head attends to. Open question (advisor): is Q just re-encoding a readable
discourse feature? Test which predictor's top pick matches the head's TRUE
selected entity (argmax of its real attention over entity candidates):

  recency    most recent entity                 (positional, the weak baseline)
  salience   most-mentioned entity so far        (discourse-salient / protagonist)
  coref      most recent ESTABLISHED entity      (prior mention of a tracked name)
  Qlearned   the fitted rank-1 selector          (reference: the learned ceiling)
  random     1/n_candidates                        (floor)

If salience/coref >> recency and approach Qlearned, the "which candidate"
residue is largely nameable and the smooth part shrinks. If they stay near
recency while Qlearned is far above, selection is genuinely not a simple
discourse rule and the low-rank direction is irreducibly needed.
"""
import collections, gc, math, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L = 320, 50
KCAND = int(sys.argv[1]) if len(sys.argv) > 1 else 20   # entity-window cap

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size, cfg.vocab_size

# ---- induction heads ----
torch.manual_seed(0)
sq = torch.randint(1000, V - 1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad():
    out = model(ids, output_attentions=True)
qp = torch.arange(L, 2 * L - 1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qp, qp - L + 1].mean(dim=(0, 2))
del out; gc.collect()
HEADS = [(int(x) // NH, int(x) % NH) for x in ind.flatten().topk(6).indices]
LAYERS = sorted({l for l, _ in HEADS})
print(f"heads: {HEADS}")

resid = {}
hooks = [model.model.layers[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())),
    with_kwargs=True) for l in LAYERS]

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)
def chunk(o): return tokz.encode(raw[o:o + 6000])[:T_C]
TRAIN = [chunk(o) for o in range(0, 520000, 40000)]
TEST = [chunk(o) for o in (540000, 580000, 620000, 660000, 700000)]


# ---- fit the rank-1 selector Q (reference ceiling), as in probe_rank2 ----
def collect(seqs):
    data = {hd: [] for hd in HEADS}
    for seq in seqs:
        n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
        with torch.no_grad():
            o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu(); H = resid[l][0].float().cpu()
            for i in range(10, n):
                cand = [j for j in ent if j < i][-KCAND:]
                if len(cand) < 3: continue
                w = A[i, cand]
                if w.sum() < 0.15: continue
                rec = torch.tensor([-math.log(i - j) for j in cand])
                data[(l, h)].append((H[i], H[cand], w / w.sum(), rec))
        del o; gc.collect()
    return data


print("fitting rank-1 selectors (for Q reference)...")
TR = collect(TRAIN)
SEL = {}
for hd in HEADS:
    s = TR[hd]
    if len(s) < 20: continue
    a = torch.zeros(1, requires_grad=True)
    P = (torch.randn(D, 1) * 0.02).requires_grad_()
    Q = (torch.randn(D, 1) * 0.02).requires_grad_()
    opt = torch.optim.Adam([a, P, Q], lr=0.05, weight_decay=1e-3)
    for _ in range(250):
        loss = 0.0
        for hi, Hc, dist, rec in s:
            lg = a * rec + (hi @ P) @ (Hc @ Q).T
            loss = loss - (dist * F.log_softmax(lg, -1)).sum()
        (loss / len(s)).backward(); opt.step(); opt.zero_grad()
    SEL[hd] = (a.detach(), P.detach()[:, 0], Q.detach()[:, 0])
print(f"  fit {len(SEL)}/{len(HEADS)} heads")


# ---- which predictor picks the entity the head actually selected? ----
RULES = ["recency", "earliest", "salience", "coref", "estFirst", "Qlearned"]
HIT = {hd: {r: 0 for r in RULES + ["random"]} for hd in SEL}
N = {hd: 0 for hd in SEL}

for seq in TEST:
    n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
    with torch.no_grad():
        o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    # running entity-token frequency up to each position
    seen = collections.Counter()
    freq_at = []                              # freq_at[i][token] = count in seq[:i]
    for i in range(n):
        freq_at.append(seen.copy())
        if is_entity(seq[i]): seen[seq[i]] += 1
    for (l, h) in SEL:
        a, P, Q = SEL[(l, h)]
        Q = Q.to(DEV)
        A = o.attentions[l][0, h].float()
        H = resid[l][0].float()
        for i in range(10, n):
            cand = [j for j in ent if j < i][-KCAND:]
            if len(cand) < 3: continue
            w = A[i, cand]
            if w.sum() < 0.15: continue
            truth = int(w.argmax())                       # entity the head selected
            fq = freq_at[i]
            # nameable predictors -> index into cand
            rec = max(range(len(cand)), key=lambda k: cand[k])            # most recent
            ear = min(range(len(cand)), key=lambda k: cand[k])            # earliest in window
            sal = max(range(len(cand)),
                      key=lambda k: (fq.get(seq[cand[k]], 0), cand[k]))   # most-mentioned
            est = [k for k in range(len(cand)) if fq.get(seq[cand[k]], 0) >= 1]
            cor = max(est, key=lambda k: cand[k]) if est else rec         # recent established
            esf = min(est, key=lambda k: cand[k]) if est else ear         # earliest established
            ql = int((H[torch.tensor(cand, device=DEV)] @ Q).argmax())    # learned selector
            picks = {"recency": rec, "earliest": ear, "salience": sal,
                     "coref": cor, "estFirst": esf, "Qlearned": ql}
            for r, p in picks.items():
                HIT[(l, h)][r] += int(p == truth)
            HIT[(l, h)]["random"] += 1.0 / len(cand)
            N[(l, h)] += 1
    del o; gc.collect()


print(f"\ntop-1 agreement with the head's actually-selected entity (held-out)")
print(f"{'head':>9}{'n':>6}" + "".join(f"{r:>10}" for r in ["random"] + RULES))
agg = collections.defaultdict(list)
for hd in SEL:
    if not N[hd]: continue
    row = {r: HIT[hd][r] / N[hd] for r in ["random"] + RULES}
    for r, v in row.items(): agg[r].append(v)
    print(f"{str(hd):>9}{N[hd]:>6}" + "".join(f"{row[r]:>10.3f}" for r in ["random"] + RULES),
          flush=True)
mean = lambda v: sum(v) / len(v)
print(f"{'POOLED':>9}{'':>6}" + "".join(f"{mean(agg[r]):>10.3f}" for r in ["random"] + RULES))
print("\nread: salience/coref >> recency and near Qlearned => selection is a nameable"
      "\n      discourse rule. Far below Qlearned => low-rank direction irreducible.")
for hk in hooks:
    hk.remove()
