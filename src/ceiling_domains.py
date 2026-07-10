"""Causal test on the EVAL axis (salvages the twin hypothesis without the
induction phase-change gamble): does the rule-nameable fraction rise with the
regularity of the TEXT, holding the model fixed?

Models with real induction heads (Qwen3-0.6B, Pythia-410M) measured on three
domains of increasing structure:
    fiction    (ministral narrative — entity-heavy, our hard case)
    wikitext   (encyclopedic — verbatim entity/term repeats)
    structured (the Act-II schemas — maximally rule-like)
8 chunks each, pooled gaps. Prediction: structured > wikitext > fiction.
"""
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, NCHUNK, L = 600, 8, 50

fiction = open("ministral_corpus.txt").read()
structured = open("structured_corpus.txt").read()
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
wiki = "\n".join(r["text"] for r in ds if r["text"].strip())

DOMAINS = {
    "fiction":    lambda i: fiction[30000 + 90000*i: 30000 + 90000*i + 9000],
    "wikitext":   lambda i: wiki[i*30000:(i+1)*30000],
    "structured": lambda i: structured[i*9000:(i+1)*9000] * 2,   # ensure length
}

def match_cols(seq):
    occ = {}; cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p+1 < len(seq): cols[i].add(p+1)
        occ.setdefault(seq[i], []).append(i)
    return cols

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

def measure(name, dtype):
    print(f"\n===== {name} =====", flush=True)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager", dtype=dtype).to(DEV).eval()
    tokz = AutoTokenizer.from_pretrained(name)
    A = Arch(model); V = model.config.vocab_size
    torch.manual_seed(0)
    sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
    with torch.no_grad(): out = model(ids, output_attentions=True)
    qpos = torch.arange(L, 2*L-1); ind = torch.zeros(A.NL, A.NH)
    for l, att in enumerate(out.attentions):
        ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
    del out; gc.collect(); torch.mps.empty_cache()
    HEADS = [(l,h) for l in range(A.NL) for h in range(A.NH) if ind[l,h] > 0.2]
    BYL = {}
    for l,h in HEADS: BYL.setdefault(l, []).append(h)
    print(f"induction heads: {len(HEADS)}")

    def nll(seq, mode, ATT=None, mask=None):
        hooks = []
        if mode != "intact":
            if mode in ("masked","inverse"): hooks += A.vhooks(BYL.keys())
            for l, hs in BYL.items():
                def oh(mod, inp, l=l, hs=hs):
                    x = inp[0].clone()
                    for h in hs:
                        if mode == "zero": x[0,:,h*A.DH:(h+1)*A.DH] = 0
                        else:
                            m = mask if mode=="masked" else torch.tril(1-mask.cpu()).to(DEV)
                            x[0,:,h*A.DH:(h+1)*A.DH] = ((ATT[l][h]*m) @ A.v(l,h).float()).to(x.dtype)
                    return (x,)+inp[1:]
                hooks.append(A.op(l).register_forward_pre_hook(oh))
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
        for hk in hooks: hk.remove()
        return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

    for dom, txt in DOMAINS.items():
        rows = []
        for i in range(NCHUNK):
            seq = tokz.encode(txt(i))[:T_C]
            if len(seq) < T_C: continue
            with torch.no_grad(): o = model(torch.tensor([seq]).to(DEV), output_attentions=True)
            ATT = {l: o.attentions[l][0].float() for l in BYL}; del o
            n = len(seq); cols = match_cols(seq)
            mask = torch.zeros(n, n)
            for a, cs in enumerate(cols):
                for j in cs: mask[a, j] = 1
            mask[:, 0] = 1; maskD = mask.to(DEV)
            ni = nll(seq, "intact"); nz = nll(seq, "zero")
            nm = nll(seq, "masked", ATT, maskD); nv = nll(seq, "inverse", ATT, maskD)
            rows.append((ni, nz, nm, nv)); del ATT; gc.collect(); torch.mps.empty_cache()
        t = torch.tensor(rows); gap = t[:,1]-t[:,0]; rm = t[:,1]-t[:,2]
        pm = rm.sum()/gap.sum()
        # cluster bootstrap over chunks: CI on the pooled ratio-of-sums
        g = torch.Generator().manual_seed(0); B = 5000; n = len(rows); bs = []
        for _ in range(B):
            idx = torch.randint(0, n, (n,), generator=g)
            bs.append((rm[idx].sum()/gap[idx].sum()).item())
        bs.sort(); lo, hi = bs[int(.025*B)], bs[int(.975*B)]
        print(f"  {dom:<11} gap {gap.mean():.3f}±{gap.std():.3f}  masked-to-rule {pm:.0%}  95%CI [{lo:.0%},{hi:.0%}]", flush=True)
    del model; gc.collect(); torch.mps.empty_cache()

for name, dt in [("Qwen/Qwen3-0.6B", torch.float32), ("EleutherAI/pythia-410m", torch.float32)]:
    measure(name, dt)
