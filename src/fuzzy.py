"""Chapter 2, step 7: FUZZY induction -- equivalence classes, not token IDs.

Inspection showed off-rule attention is induction over fuzzy matches:
case/space folding ('his'~' His'), tokenization variants (' Below'~'Below'),
morphological families (' wasn'~' hadn' -> copy the ''t'), name prefixes.

Token similarity (hand-set, no fitting):
    1.0  same id
    0.9  same normalized string (lower, strip space/punct)
    cos  input-embedding cosine, if > SIM_TH (within-sequence neighbors)

Match strength between context at i and occurrence at j-1: sum of per-token
similarities over the common suffix (stop when sim < SIM_TH), capped MAXO.

Part 1  fuzzy CEILING: true attention masked to fuzzy-match columns (+BOS)
Part 2  fuzzy PROSTHESIS: v5 soft rule attention with fuzzy strengths,
        reusing v5's fitted alpha=1 and gains (nothing refitted)
"""
import torch
import prosthesis5 as P5
import ceiling as C

SIM_TH = 0.55
MAXO, DEV, DH, GROUP = P5.MAXO, P5.DEV, P5.DH, P5.GROUP
HEADS, LAYERS = P5.HEADS, P5.LAYERS
GAINS = [0.25, 0.5, 0.5, 0.75, 0.75, 0.5, 1.0, 0.5]
ALPHA = 1.0

EMB = P5.model.model.embed_tokens.weight.detach().float().cpu()

def sim_lookup(seq):
    """[U,U] similarity over the sequence's unique tokens."""
    uniq = sorted(set(seq))
    idx = {t: k for k, t in enumerate(uniq)}
    E = EMB[torch.tensor(uniq)]
    E = E / E.norm(dim=-1, keepdim=True)
    S = (E @ E.T).clamp(min=0)
    norm = [P5.tokz.decode([t]).strip().lower().strip('.,"\'') for t in uniq]
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            if norm[a] and norm[a] == norm[b]:
                S[a, b] = S[b, a] = max(S[a, b], 0.9)
    S.fill_diagonal_(1.0)
    S[S < SIM_TH] = 0
    return S, idx

def fuzzy_table(seq):
    """[(i, j, strength)] follower positions j with fuzzy match strength."""
    S, idx = sim_lookup(seq)
    n = len(seq)
    ids = [idx[t] for t in seq]
    out = []
    for i in range(1, n):
        for j in range(1, i + 1):                     # j = follower position
            s0 = S[ids[i], ids[j - 1]]
            if s0 <= 0: continue
            strength, k = float(s0), 1
            while (k < MAXO and i - k >= 0 and j - 1 - k >= 0):
                s = S[ids[i - k], ids[j - 1 - k]]
                if s <= 0: break
                strength += float(s); k += 1
            out.append((i, j, strength))
    return out

FCACHE = {}
def get_table(seq):
    if id(seq) not in FCACHE: FCACHE[id(seq)] = fuzzy_table(seq)
    return FCACHE[id(seq)]

def fuzzy_attn(seq):
    """v5-style soft attention from fuzzy strengths, gated per strength bucket."""
    n = len(seq)
    A = torch.zeros(n, n)
    smax = torch.zeros(n)
    for i, j, st in get_table(seq):
        A[i, j] = torch.tensor(ALPHA * st).exp()
        smax[i] = max(smax[i], st)
    Z = A.sum(-1, keepdim=True).clamp(min=1e-9)
    g = torch.zeros(MAXO + 1); g[1:] = torch.tensor(GAINS)
    bucket = smax.round().long().clamp(max=MAXO)
    return (A / Z) * g[bucket].unsqueeze(-1)

def fuzzy_cols(seq):
    cols = [set() for _ in range(len(seq))]
    for i, j, st in get_table(seq):
        if st >= 0.9: cols[i].add(j)
    return cols

def run_soft_fuzzy(seq):
    A = fuzzy_attn(seq).to(DEV)
    vcache, hooks = {}, []
    for l in LAYERS:
        attn = P5.model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        hs = [h for ll, h in HEADS if ll == l]
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A @ vcache[l][:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(P5.model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    seq = P5.test_seq
    ni, nz = P5.run(seq, "intact"), P5.run(seq, "zero")
    print(f"natural B held-out: intact {ni:.3f}  zero {nz:.3f}")

    # part 1: fuzzy ceiling -- reuse ceiling.run_masked with fuzzy columns
    C.rule_cols = fuzzy_cols
    nm = C.run_masked(seq, "true-masked")
    print(f"  true attn masked to FUZZY cols  {nm:.3f}   ceiling {(nz-nm)/(nz-ni):.0%}  (exact-col ceiling was 32%)")

    # part 2: fuzzy prosthesis
    nf = run_soft_fuzzy(seq)
    print(f"  fuzzy soft-rule prosthesis      {nf:.3f}   recovered {(nz-nf)/(nz-ni):.0%}  (exact v5 was 30%)")
