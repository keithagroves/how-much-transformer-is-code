"""Chapter 2, step 6: ceiling for rule-attention substitution.

Take each circuit head's TRUE attention matrix (intact run), mask it to the
positions our rule can name -- match-follower columns (+ BOS sink) -- and run
with that. Variants:
    true-full      sanity: A_true @ V == intact head output (must be ~100%)
    true-masked    A_true restricted to rule positions + BOS
    true-inverse   A_true restricted to everything EXCEPT rule positions
Recovery of true-masked = the ceiling any code rule that picks these
positions can reach. true-inverse = what flows through positions no
induction rule would ever name.
"""
import torch
import prosthesis5 as P5

model, DEV, DH, GROUP, MAXO = P5.model, P5.DEV, P5.DH, P5.GROUP, P5.MAXO
HEADS, LAYERS = P5.HEADS, P5.LAYERS

def rule_cols(seq):
    """per row i: set of follower columns the rule names"""
    tab = P5.match_table(seq)
    cols = [set() for _ in range(len(seq))]
    for i, j, m in tab: cols[i].add(j)
    return cols

def run_masked(seq, variant):
    ids = torch.tensor([seq]).to(DEV)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    n = len(seq)
    cols = rule_cols(seq)
    mask = torch.zeros(n, n)
    for i, cs in enumerate(cols):
        for j in cs: mask[i, j] = 1
    mask[:, 0] = 1                                   # BOS/first-token sink
    if variant == "true-inverse":
        mask = 1 - mask
        mask = torch.tril(mask)
    if variant == "true-full":
        mask = torch.tril(torch.ones(n, n))
    mask = mask.to(DEV)

    A = {}                                           # (l,h) -> masked true attention
    for l in LAYERS:
        att = out.attentions[l][0]                   # [H, T, T]
        for ll, h in HEADS:
            if ll == l: A[(l, h)] = att[h] * mask

    vcache, hooks = {}, []
    for l in LAYERS:
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        hs = [h for ll, h in HEADS if ll == l]
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A[(l, h)] @ vcache[l][:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    for name, seqs in [("natural B (held-out)", [P5.test_seq]),
                       ("repeated random", P5.rnd_test[:1])]:
        ni = sum(P5.run(sq, "intact") for sq in seqs) / len(seqs)
        nz = sum(P5.run(sq, "zero") for sq in seqs) / len(seqs)
        print(f"{name}:  intact {ni:.3f}  zero {nz:.3f}")
        for v in ("true-full", "true-masked", "true-inverse"):
            nv = sum(run_masked(sq, v) for sq in seqs) / len(seqs)
            print(f"    {v:<13} {nv:.3f}   recovered {(nz-nv)/(nz-ni):.0%}")
