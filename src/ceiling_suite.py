"""Ceiling robustness suite: is the ~30/70 split an artifact?

Confounds addressed:
  1. shared eval text  -> two domains: ministral fiction AND WikiText-103 val
  2. single-sequence noise -> 8 chunks per domain, pooled + per-chunk stats
  3. scale             -> Qwen3-0.6B, Pythia-410M, Qwen3-1.7B (4x step)

For each model: find induction heads (score>0.2 on repeated random), then per
chunk measure intact / zero-ablated / true-attention-masked-to-rule-cols(+BOS)
/ inverse. Recovery pooled across chunks: sum(nz-nx)/sum(nz-ni).
All inference; no training anywhere.
"""
import gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, NCHUNK, L = 600, 8, 50

MODELS = [("Qwen/Qwen3-0.6B", torch.float32),
          ("EleutherAI/pythia-410m", torch.float32),
          ("Qwen/Qwen3-1.7B", torch.float16)]

raw = open("ministral_corpus.txt").read()
MIN_STARTS = [30000 + 90000 * i for i in range(NCHUNK)]
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
WIKI = "\n".join(r["text"] for r in ds if r["text"].strip())

def match_cols(seq):
    occ = {}
    cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p + 1 < len(seq): cols[i].add(p + 1)
        occ.setdefault(seq[i], []).append(i)
    return cols

class Arch:
    def __init__(self, model):
        self.m = model
        self.neox = hasattr(model, "gpt_neox")
        cfg = model.config
        self.NL, self.NH = cfg.num_hidden_layers, cfg.num_attention_heads
        if self.neox:
            self.DH = cfg.hidden_size // self.NH
            self.GROUP = 1
            self.layers = model.gpt_neox.layers
        else:
            self.DH = cfg.head_dim
            self.GROUP = self.NH // cfg.num_key_value_heads
            self.layers = model.model.layers
        self.vcache = {}

    def vhooks(self, ls):
        hks = []
        for l in ls:
            if self.neox:
                def hook(mod, inp, outp, l=l):
                    q3 = outp.view(outp.shape[0], outp.shape[1], self.NH, 3 * self.DH)
                    self.vcache[l] = q3[0, :, :, 2*self.DH:].detach()      # [T,NH,DH]
                hks.append(self.layers[l].attention.query_key_value.register_forward_hook(hook))
            else:
                def hook(mod, inp, outp, l=l):
                    self.vcache[l] = outp[0].detach()                       # [T,NKV*DH]
                hks.append(self.layers[l].self_attn.v_proj.register_forward_hook(hook))
        return hks

    def v_of(self, l, h):
        if self.neox: return self.vcache[l][:, h, :]
        g = h // self.GROUP
        return self.vcache[l][:, g*self.DH:(g+1)*self.DH]

    def out_proj(self, l):
        return self.layers[l].attention.dense if self.neox else self.layers[l].self_attn.o_proj

def measure(name, dtype):
    print(f"\n===== {name} =====", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, attn_implementation="eager", dtype=dtype).to(DEV).eval()
    tokz = AutoTokenizer.from_pretrained(name)
    A = Arch(model)
    V = model.config.vocab_size

    # induction heads
    torch.manual_seed(0)
    seq = torch.randint(1000, V - 1000, (8, L))
    ids = torch.cat([seq, seq], dim=1).to(DEV)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    qpos = torch.arange(L, 2*L - 1)
    ind = torch.zeros(A.NL, A.NH)
    for l, att in enumerate(out.attentions):
        a = att.float().cpu()
        ind[l] = a[:, :, qpos, qpos - L + 1].mean(dim=(0, 2))
    del out; gc.collect(); torch.mps.empty_cache()
    HEADS = [(l, h) for l in range(A.NL) for h in range(A.NH) if ind[l, h] > 0.2]
    BYL = {}
    for l, h in HEADS: BYL.setdefault(l, []).append(h)
    print(f"induction heads (>0.2): {len(HEADS)} / {A.NL*A.NH}")

    def nll(sq, mode, ATT=None, maskD=None):
        hooks = []
        if mode != "intact":
            if mode in ("masked", "inverse"):
                hooks += A.vhooks(BYL.keys())
            for l, hs in BYL.items():
                def ohook(mod, inp, l=l, hs=hs):
                    x = inp[0].clone()
                    for h in hs:
                        if mode == "zero":
                            x[0, :, h*A.DH:(h+1)*A.DH] = 0
                        else:
                            m = maskD if mode == "masked" else (torch.tril(1 - maskD.cpu()).to(DEV))
                            x[0, :, h*A.DH:(h+1)*A.DH] = ((ATT[l][h] * m) @ A.v_of(l, h).float()).to(x.dtype)
                    return (x,) + inp[1:]
                hooks.append(A.out_proj(l).register_forward_pre_hook(ohook))
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0, :-1].float(), -1)
        for hk in hooks: hk.remove()
        return -lp.gather(-1, torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item()

    def ceiling_chunk(sq):
        n = len(sq)
        with torch.no_grad():
            out = model(torch.tensor([sq]).to(DEV), output_attentions=True)
        ATT = {l: out.attentions[l][0].float() for l in BYL}
        del out
        cols = match_cols(sq)
        mask = torch.zeros(n, n)
        for i, cs in enumerate(cols):
            for j in cs: mask[i, j] = 1
        mask[:, 0] = 1
        maskD = mask.to(DEV)
        ni = nll(sq, "intact")
        nz = nll(sq, "zero")
        nm = nll(sq, "masked", ATT, maskD)
        nv = nll(sq, "inverse", ATT, maskD)
        del ATT; gc.collect(); torch.mps.empty_cache()
        return ni, nz, nm, nv

    for domain, text_of in [("ministral", lambda i: raw[MIN_STARTS[i]:MIN_STARTS[i]+9000]),
                            ("wikitext", lambda i: WIKI[i*30000:(i+1)*30000])]:
        rows = []
        for i in range(NCHUNK):
            sq = tokz.encode(text_of(i))[:T_C]
            if len(sq) < T_C: continue
            rows.append(ceiling_chunk(sq))
        t = torch.tensor(rows)                       # [n, 4] = ni, nz, nm, nv
        gap = (t[:, 1] - t[:, 0])
        rm = (t[:, 1] - t[:, 2]); rv = (t[:, 1] - t[:, 3])
        pm, pv = rm.sum()/gap.sum(), rv.sum()/gap.sum()
        ratios_m = rm/gap; ratios_v = rv/gap
        print(f"{domain:<10} gap {gap.mean():.3f}±{gap.std():.3f}  "
              f"masked {pm:.0%} (per-chunk {ratios_m.mean():.0%}±{ratios_m.std():.0%})  "
              f"inverse {pv:.0%} (per-chunk {ratios_v.mean():.0%}±{ratios_v.std():.0%})", flush=True)

    del model; gc.collect(); torch.mps.empty_cache()

for name, dtype in MODELS:
    measure(name, dtype)
