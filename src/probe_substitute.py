"""Probe, stage 2: does 'rules + one learned direction' recover more DOWNSTREAM
function than the discrete-rule ceiling?

Per induction head, fit rank-1 entity-selection (recency + one content
direction) on train chunks. Then on held-out chunks substitute the head's
attention and measure next-token NLL under:
    zero       head off (floor)
    rule-only  keep TRUE attention on rule cols (+BOS); drop the rest
               == the discrete-ceiling condition
    rule+probe rule cols PLUS the head's true entity-mass redistributed over
               entity candidates by the learned rank-1 pattern
    intact     (ceiling, 100%)

Report % of (zero - intact) gap recovered. If rule+probe >> rule-only, the
diffuse part is largely recency + a low-dimensional entity-selection direction.
"""
import collections, gc, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, L, KCAND, R = 320, 50, 20, 1

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
tokz = AutoTokenizer.from_pretrained(MODEL)
cfg = model.config
NL, NH, D = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.hidden_size
DH, GROUP = cfg.head_dim, cfg.num_attention_heads // cfg.num_key_value_heads
V = cfg.vocab_size

torch.manual_seed(0)
sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
with torch.no_grad(): out = model(ids, output_attentions=True)
qpos = torch.arange(L, 2*L-1); ind = torch.zeros(NL, NH)
for l, att in enumerate(out.attentions):
    ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
del out; gc.collect()
HEADS = [(l, h) for l in range(NL) for h in range(NH) if ind[l, h] > 0.2]
BYL = {}
for l, h in HEADS: BYL.setdefault(l, []).append(h)
LAYERS = sorted(BYL)
print(f"heads: {len(HEADS)} with induction score > 0.2")

resid = {}
hooks = [model.model.layers[l].register_forward_pre_hook(
    (lambda mod, args, kwargs, l=l: resid.__setitem__(l, args[0].detach())), with_kwargs=True)
    for l in LAYERS]

raw = open("ministral_corpus.txt").read()
frq = collections.Counter(tokz.encode(raw[:300000]))
def is_entity(t):
    d = tokz.decode([t]).strip()
    return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)
def match_cols(seq):
    occ = {}; cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p+1 < len(seq): cols[i].add(p+1)
        occ.setdefault(seq[i], []).append(i)
    return cols
def chunk(o): return tokz.encode(raw[o:o+6000])[:T_C]

TRAIN = [chunk(o) for o in range(0, 520000, 40000)]
TEST  = [chunk(o) for o in (540000, 580000, 620000, 660000, 700000)]

# ---- fit rank-1 entity selector per head (recency + one direction) ----
def collect(seqs):
    data = {hd: [] for hd in HEADS}
    for seq in seqs:
        n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
        with torch.no_grad(): o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        for (l, h) in HEADS:
            A = o.attentions[l][0, h].float().cpu(); H = resid[l][0].float().cpu()
            for i in range(10, n):
                cand = [j for j in ent if j < i][-KCAND:]
                if len(cand) < 3: continue
                w = A[i, cand]
                if w.sum() < 0.15: continue
                rec = torch.tensor([-torch.tensor(float(i-j)).log() for j in cand])
                data[(l, h)].append((H[i], H[cand], w/w.sum(), rec))
        del o; gc.collect()
    return data

print("fitting per-head rank-1 selectors...")
TR = collect(TRAIN)
SEL = {}
for hd in HEADS:
    s = TR[hd]
    if len(s) < 20: continue                      # not enough entity attention to fit; rule-only
    a = torch.zeros(1, requires_grad=True)
    P = (torch.randn(D, R)*0.02).requires_grad_(); Q = (torch.randn(D, R)*0.02).requires_grad_()
    opt = torch.optim.Adam([a, P, Q], lr=0.05, weight_decay=1e-3)
    for _ in range(250):
        loss = 0.0
        for hi, Hc, dist, rec in s:
            lg = a*rec + (hi @ P) @ (Hc @ Q).T
            loss = loss - (dist * F.log_softmax(lg, -1)).sum()
        (loss/len(s)).backward(); opt.step(); opt.zero_grad()
    SEL[hd] = (a.detach(), P.detach(), Q.detach())
print(f"  fit selectors for {len(SEL)}/{len(HEADS)} heads (rest rule-only)")

# ---- substitution measurement ----
vcache = {}
def vhooks():
    hk = []
    for l in LAYERS:
        def h(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hk.append(model.model.layers[l].self_attn.v_proj.register_forward_hook(h))
    return hk

def measure(seq, mode, ATT=None, rule=None, Abuilt=None):
    hooks = []
    if mode != "intact":
        if mode in ("rule", "probe"): hooks += vhooks()
        for l, hs in BYL.items():
            def oh(mod, inp, l=l, hs=hs):
                x = inp[0].clone()
                for h in hs:
                    if mode == "zero":
                        x[0, :, h*DH:(h+1)*DH] = 0
                    else:
                        g = h // GROUP; vh = vcache[l][:, g*DH:(g+1)*DH].float()
                        x[0, :, h*DH:(h+1)*DH] = (Abuilt[(l, h)] @ vh).to(x.dtype)
                return (x,) + inp[1:]
            hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(oh))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

def build(seq, ATT, probe):
    n = len(seq); ent = [j for j in range(n) if is_entity(seq[j])]
    cols = match_cols(seq)
    rule = torch.zeros(n, n)
    for i, cs in enumerate(cols):
        for j in cs: rule[i, j] = 1
    rule[:, 0] = 1
    Ab = {}
    for (l, h) in HEADS:
        A = ATT[l][h]
        M = (A * rule).clone()                                   # keep true weights on rule cols
        if probe and (l, h) in SEL:
            a, P, Q = SEL[(l, h)]; H = resid[l][0].float().cpu()
            for i in range(2, n):
                cand = [j for j in ent if j < i and rule[i, j] == 0][-KCAND:]
                if len(cand) < 2: continue
                m_ent = A[i, cand].sum().item()                  # head's true entity mass
                if m_ent < 0.05: continue
                rec = torch.tensor([-torch.tensor(float(i-j)).log() for j in cand])
                lg = a*rec + (H[i] @ P) @ (H[cand] @ Q).T
                w = F.softmax(lg, -1) * m_ent
                for k, j in enumerate(cand): M[i, j] += w[k]
        rs = M.sum(-1, keepdim=True)                             # renormalize rows -> proper
        M = torch.where(rs > 1e-6, M / rs.clamp(min=1e-6), M)    # convex combination
        Ab[(l, h)] = M.to(DEV)
    return Ab

print(f"\n{'chunk':>6}{'gap':>8}{'rule rec':>10}{'probe rec':>11}")
S = {"gap": 0.0, "rule": 0.0, "probe": 0.0}
for c, seq in enumerate(TEST):
    with torch.no_grad(): o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
    ATT = {l: o.attentions[l][0].float().cpu() for l in BYL}; del o
    ni = measure(seq, "intact"); nz = measure(seq, "zero")
    Ar = build(seq, ATT, probe=False); nr = measure(seq, "rule", Abuilt=Ar)
    Ap = build(seq, ATT, probe=True);  np_ = measure(seq, "probe", Abuilt=Ap)
    gap = nz - ni
    S["gap"] += gap; S["rule"] += (nz - nr); S["probe"] += (nz - np_)
    print(f"{c:>6}{gap:>8.3f}{(nz-nr):>10.3f}{(nz-np_):>11.3f}", flush=True)
    del ATT; gc.collect()
print(f"\npooled ablation gap: {S['gap']:.3f} nats over {len(TEST)} chunks")
print(f"discrete ceiling (rule-only): {S['rule']/S['gap']:.0%}")
print(f"rule + rank-1 entity probe:   {S['probe']/S['gap']:.0%}")
for hk in hooks: hk.remove()
