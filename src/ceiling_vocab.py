"""Falsification test: is the diffuse residue un-nameable, or just un-nameable
in our narrow rule vocabulary?

Re-run the substitution-ceiling measurement with an EXPANDED set of nameable
positions. Baseline vocabulary = {exact-match followers, BOS}. We add
entity-anchored rules a cheap parser could implement:
    +recent  : the single most recent entity position before the query
    +entities: all recent entity positions (capitalized / rare content tokens)

Ceiling = keep the head's TRUE attention on the nameable columns, drop the
rest, run head output through the model's own value path, measure recovered
fraction of the (zero - intact) gap. Pooled over 8 chunks per model. If
+entities recovers materially more than baseline on fiction (where the residue
is largest), the ceiling was partly a ceiling on our rule LANGUAGE.
"""
import collections, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, NCHUNK, L = 512, 8, 50

raw = open("ministral_corpus.txt").read()
def chunks_for(tokz):
    return [tokz.encode(raw[30000 + 70000*i: 30000 + 70000*i + 9000])[:T_C]
            for i in range(NCHUNK)]

class Arch:
    def __init__(s, m):
        s.m = m; s.neox = hasattr(m, "gpt_neox"); c = m.config
        s.NL, s.NH = c.num_hidden_layers, c.num_attention_heads
        if s.neox:
            s.DH = c.hidden_size // s.NH; s.G = 1; s.layers = m.gpt_neox.layers
        else:
            s.DH = c.head_dim; s.G = s.NH // c.num_key_value_heads; s.layers = m.model.layers
        s.vc = {}
    def vhooks(s, ls):
        hk = []
        for l in ls:
            if s.neox:
                def h(mod, inp, outp, l=l):
                    q3 = outp.view(outp.shape[0], outp.shape[1], s.NH, 3*s.DH)
                    s.vc[l] = q3[0,:,:,2*s.DH:].detach()
                hk.append(s.layers[l].attention.query_key_value.register_forward_hook(h))
            else:
                def h(mod, inp, outp, l=l): s.vc[l] = outp[0].detach()
                hk.append(s.layers[l].self_attn.v_proj.register_forward_hook(h))
        return hk
    def v(s, l, h):
        if s.neox: return s.vc[l][:, h, :]
        g = h // s.G; return s.vc[l][:, g*s.DH:(g+1)*s.DH]
    def op(s, l): return s.layers[l].attention.dense if s.neox else s.layers[l].self_attn.o_proj

def match_cols(seq):
    occ = {}; cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p+1 < len(seq): cols[i].add(p+1)
        occ.setdefault(seq[i], []).append(i)
    return cols

def measure(name, dtype):
    print(f"\n===== {name} =====", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="eager", dtype=dtype).to(DEV).eval()
    tokz = AutoTokenizer.from_pretrained(name)
    A = Arch(model); V = model.config.vocab_size
    frq = collections.Counter(tokz.encode(raw[:300000]))
    def is_entity(t):
        d = tokz.decode([t]).strip()
        return bool(d) and d.isalpha() and (d[0].isupper() or frq.get(t, 0) < 30)

    torch.manual_seed(0)
    sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
    with torch.no_grad(): out = model(ids, output_attentions=True)
    qpos = torch.arange(L, 2*L-1); ind = torch.zeros(A.NL, A.NH)
    for l, att in enumerate(out.attentions):
        ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
    del out; gc.collect()
    HEADS = [(l, h) for l in range(A.NL) for h in range(A.NH) if ind[l, h] > 0.2]
    BYL = {}
    for l, h in HEADS: BYL.setdefault(l, []).append(h)
    print(f"induction heads: {len(HEADS)}")

    def run(seq, mode, ATT=None, mask=None):
        hooks = []
        if mode != "intact":
            if mode == "keep": hooks += A.vhooks(BYL.keys())
            for l, hs in BYL.items():
                def oh(mod, inp, l=l, hs=hs):
                    x = inp[0].clone()
                    for h in hs:
                        if mode == "zero": x[0,:,h*A.DH:(h+1)*A.DH] = 0
                        else: x[0,:,h*A.DH:(h+1)*A.DH] = ((ATT[l][h]*mask) @ A.v(l,h).float()).to(x.dtype)
                    return (x,)+inp[1:]
                hooks.append(A.op(l).register_forward_pre_hook(oh))
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
        for hk in hooks: hk.remove()
        return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

    VOCABS = ["match+BOS", "+recent-entity", "+all-entities", "+random-N (control)"]
    S = {v: {"gap": [], "rec": []} for v in VOCABS}   # per-chunk, for bootstrap
    g = torch.Generator().manual_seed(0)
    for seq in chunks_for(tokz):
        if len(seq) < T_C: continue
        n = len(seq)
        with torch.no_grad(): o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
        ATT = {l: o.attentions[l][0].float() for l in BYL}; del o
        cols = match_cols(seq)
        ent = set(j for j in range(n) if is_entity(seq[j]))
        base = torch.zeros(n, n)
        for i, cs in enumerate(cols):
            for j in cs: base[i, j] = 1
        base[:, 0] = 1
        recent = base.clone(); allent = base.clone(); rand = base.clone()
        for i in range(1, n):
            prev_ent = [j for j in range(i) if j in ent]
            if prev_ent:
                recent[i, prev_ent[-1]] = 1
                for j in prev_ent[-20:]: allent[i, j] = 1
            k = min(len(prev_ent[-20:]) if prev_ent else 0, i)   # match the entity count
            if k > 0:
                pick = torch.randperm(i, generator=g)[:k]
                for j in pick.tolist(): rand[i, j] = 1
        masks = {"match+BOS": base.to(DEV), "+recent-entity": recent.to(DEV),
                 "+all-entities": allent.to(DEV), "+random-N (control)": rand.to(DEV)}
        ni = run(seq, "intact"); nz = run(seq, "zero")
        for v in VOCABS:
            nk = run(seq, "keep", ATT, masks[v])
            S[v]["gap"].append(nz - ni); S[v]["rec"].append(nz - nk)
        del ATT; gc.collect(); torch.mps.empty_cache()

    # cluster bootstrap over chunks: CI on each recovered fraction (ratio of sums)
    import torch as _t
    NB = 5000; nc = len(S[VOCABS[0]]["gap"])
    gen = _t.Generator().manual_seed(0)
    idxs = [_t.randint(0, nc, (nc,), generator=gen) for _ in range(NB)]
    def frac(v, idx):
        rec = _t.tensor(S[v]["rec"]); gap = _t.tensor(S[v]["gap"])
        return (rec[idx].sum() / gap[idx].sum()).item()
    print(f"{'vocabulary':<20}{'recovered':>11}{'   95% CI':>16}")
    for v in VOCABS:
        pt = frac(v, _t.arange(nc))
        bs = sorted(frac(v, ix) for ix in idxs)
        print(f"{v:<20}{pt:>11.0%}   [{bs[int(.025*NB)]:.0%}, {bs[int(.975*NB)]:.0%}]", flush=True)
    # paired difference: all-entities minus random-N (the key "real signal" test)
    diffs = sorted(frac("+all-entities", ix) - frac("+random-N (control)", ix) for ix in idxs)
    dpt = frac("+all-entities", _t.arange(nc)) - frac("+random-N (control)", _t.arange(nc))
    print(f"{'entities - random':<20}{dpt:>+11.0%}   [{diffs[int(.025*NB)]:+.0%}, {diffs[int(.975*NB)]:+.0%}]"
          f"   {'(excludes 0)' if diffs[int(.025*NB)] > 0 else '(spans 0)'}", flush=True)
    del model; gc.collect(); torch.mps.empty_cache()

for name, dt in [("Qwen/Qwen3-0.6B", torch.float32), ("EleutherAI/pythia-410m", torch.float32)]:
    measure(name, dt)
