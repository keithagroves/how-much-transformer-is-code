"""Twin experiment, step 3: measure the substitution ceiling on both twins.

The causal test. Two models, identical architecture, differ only in training
data (TinyStories vs lawful schemas). Prediction from the domain-dependence
finding: the STRUCTURED twin's induction heads should be far more
rule-nameable (its world is verbatim-repeatable slots) than the NATURAL
twin's (entity-heavy narrative -> diffuse tracking).

Uses each model's OWN validation text (in-distribution), pooled over 8 chunks,
so gaps are honest and ratios are of sums, not single-chunk noise.
"""
import json, torch
import torch.nn.functional as F
from train_twin import GPT, rope

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T_C, NCHUNK, L = 256, 8, 40

def load(arm):
    ck = torch.load(f"twin_model_{arm}.pt", map_location=DEV)
    m = GPT(ck["cfg"]).to(DEV).eval()
    m.load_state_dict(ck["state"])
    return m, ck["cfg"]

def match_cols(seq):
    occ = {}
    cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p + 1 < len(seq): cols[i].add(p + 1)
        occ.setdefault(seq[i], []).append(i)
    return cols

def attach(model):
    """capture per-block attention weights + v; allow overriding attn matrix."""
    store = {}
    def mk(bi, blk):
        def fwd(x, pos):
            B, T, D = x.shape
            q, k, v = blk.qkv(blk.ln1(x)).split(D, -1)
            q = q.view(B, T, blk.h, blk.dh).transpose(1, 2)
            k = k.view(B, T, blk.h, blk.dh).transpose(1, 2)
            v = v.view(B, T, blk.h, blk.dh).transpose(1, 2)
            q, k = rope(q, pos), rope(k, pos)
            scores = (q @ k.transpose(-2, -1)) / (blk.dh ** 0.5)
            mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
            A = scores.softmax(-1)                          # [B,h,T,T]
            store[bi] = {"A": A.detach(), "v": v.detach()}
            ov = store[bi].get("override")
            if ov is not None:
                A = ov
            o = A @ v
            x = x + blk.proj(o.transpose(1, 2).reshape(B, T, D))
            h = blk.ln2(x)
            x = x + blk.w3(F.silu(blk.w1(h)) * blk.w2(h))
            return x
        return fwd
    for bi, blk in enumerate(model.blocks):
        blk.forward = mk(bi, blk)
    return store

def nll(model, seq):
    with torch.no_grad():
        lg = model(torch.tensor([seq]).to(DEV))[0, :-1].float()
    return F.cross_entropy(lg, torch.tensor(seq[1:]).to(DEV)).item()

def induction_heads(model, cfg, store):
    torch.manual_seed(0)
    V = cfg["vocab"]
    seq = torch.randint(50, V-50, (L,)).tolist()
    seq = seq + seq
    for s in store.values(): s.pop("override", None)
    nll(model, seq)
    qpos = torch.arange(L, 2*L-1)
    heads = []
    scores = {}
    for bi, s in store.items():
        A = s["A"][0].float().cpu()                         # [h,T,T]
        sc = A[:, qpos, qpos - L + 1].mean(1)               # per head
        for h in range(len(sc)):
            scores[(bi, h)] = float(sc[h])
            if sc[h] > 0.2: heads.append((bi, h))
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    return heads, top

def ceiling(model, cfg, store, val_ids, label):
    heads, top = induction_heads(model, cfg, store)
    byl = {}
    for b, h in heads: byl.setdefault(b, []).append(h)
    print(f"  [{label}] induction heads: {len(heads)}  top "
          + ", ".join(f"L{b}.H{h}={s:.2f}" for (b, h), s in top))
    if not heads:
        print(f"  [{label}] no induction heads > 0.2 — skipping ceiling"); return
    rows = []
    for c in range(NCHUNK):
        seq = val_ids[c*T_C:(c+1)*T_C].tolist()
        if len(seq) < T_C: break
        for s in store.values(): s.pop("override", None)
        ni = nll(model, seq)
        A0 = {b: store[b]["A"].clone() for b in byl}         # true attention
        n = len(seq)
        cols = match_cols(seq)
        mask = torch.zeros(n, n)
        for i, cs in enumerate(cols):
            for j in cs: mask[i, j] = 1
        mask[:, 0] = 1
        maskD = mask.to(DEV)

        def run(m):
            for b, hs in byl.items():
                A = A0[b].clone()
                for h in hs: A[0, h] = A0[b][0, h] * m
                store[b]["override"] = A
            r = nll(model, seq)
            for b in byl: store[b].pop("override", None)
            return r
        # zero = mask everything off
        nz = run(torch.zeros(n, n, device=DEV))
        nm = run(maskD)
        nv = run(torch.tril(1 - mask).to(DEV))
        rows.append((ni, nz, nm, nv))
    t = torch.tensor(rows)
    gap = t[:, 1] - t[:, 0]
    pm = (t[:, 1] - t[:, 2]).sum() / gap.sum()
    pv = (t[:, 1] - t[:, 3]).sum() / gap.sum()
    print(f"  [{label}] gap {gap.mean():.3f}±{gap.std():.3f}  "
          f"masked-to-rule {pm:.0%}  inverse {pv:.0%}", flush=True)
    return float(pm), float(pv), float(gap.mean())

if __name__ == "__main__":
    results = {}
    for arm in ("natural", "structured"):
        model, cfg = load(arm)
        store = attach(model)
        val = torch.tensor(json.load(open(f"twin_{arm}.json"))[-100_000:], dtype=torch.long)
        print(f"\n=== {arm} twin (val loss {torch.load(f'twin_model_{arm}.pt')['val']:.3f}) ===")
        results[arm] = ceiling(model, cfg, store, val, arm)
    print("\n=== TWIN VERDICT ===")
    for arm, r in results.items():
        if r: print(f"  {arm:<11} masked-to-rule {r[0]:.0%}  (gap {r[2]:.3f})")
