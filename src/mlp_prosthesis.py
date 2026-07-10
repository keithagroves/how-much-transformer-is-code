"""Chapter 2, step 10: the MLP front.

Hypothesis: many MLP layers act as token-keyed key-value memories, so their
code substitute is a LOOKUP TABLE: token id -> mean output vector (fitted on
train chunks; unseen token -> layer mean). Layers where the lookup is cheap
are context-independent enrichment; expensive layers do real composition.

Measures: solo substitution cost per layer (train), then cheapest-first
cumulative frontier on held-out natural + repeated random.
"""
import torch
import replace_all as RA

model, DEV, T = RA.model, RA.DEV, RA.T
tokz, NL = RA.tokz, RA.NL
train_seq, test_seq, rnd_test = RA.train_seq, RA.test_seq, RA.rnd_test

raw = open("ministral_corpus.txt").read()
starts = [0, 20000, 50000, 80000, 110000, 140000, 260000, 290000]
fit_chunks = [tokz.encode(raw[o:o + 8000])[:900] for o in starts]

# ---------- fit lookup tables ----------
print("capturing MLP outputs on fit chunks...")
SUM = [dict() for _ in range(NL)]     # layer -> {token: (sum_vec, count)}
MEAN = [None] * NL
cap = {}
hooks = []
for l in range(NL):
    def hook(mod, inp, outp, l=l): cap[l] = outp[0].detach().float().cpu()
    hooks.append(model.model.layers[l].mlp.register_forward_hook(hook))
tot = [None] * NL; cnt = 0
for sq in fit_chunks:
    with torch.no_grad():
        model(torch.tensor([sq]).to(DEV))
    for l in range(NL):
        o = cap[l]                                   # [T, 1024]
        tot[l] = o.sum(0) if tot[l] is None else tot[l] + o.sum(0)
        d = SUM[l]
        for i, t in enumerate(sq):
            if t in d: d[t][0].add_(o[i]); d[t][1] += 1
            else: d[t] = [o[i].clone(), 1]
    cnt += len(sq)
for hk in hooks: hk.remove()
for l in range(NL):
    MEAN[l] = (tot[l] / cnt).to(torch.float16)
    SUM[l] = {t: (v / n).to(torch.float16) for t, (v, n) in SUM[l].items()}
print(f"lookup tables: {len(SUM[0]):,} tokens seen, {NL} layers")

def lookup_matrix(seq, l):
    return torch.stack([SUM[l].get(t, MEAN[l]) for t in seq]).float().to(DEV)

def run_sub(seq, layers):
    LUT = {l: lookup_matrix(seq, l) for l in layers}
    hooks = []
    for l in layers:
        def hook(mod, inp, outp, l=l):
            return LUT[l].unsqueeze(0)
        hooks.append(model.model.layers[l].mlp.register_forward_hook(hook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    ni_tr = run_sub(train_seq, [])
    print(f"train intact {ni_tr:.3f}; solo cost per MLP layer:")
    costs = {}
    for l in range(NL):
        costs[l] = run_sub(train_seq, [l]) - ni_tr
        print(f"  L{l:>2}: {costs[l]:+.4f}", flush=True)
    torch.save({"costs": costs}, "mlp_solo_costs.pt")

    order = sorted(costs, key=lambda l: costs[l])
    ni_nat, ni_rnd = run_sub(test_seq, []), run_sub(rnd_test, [])
    print(f"\nheld-out intact: natural {ni_nat:.3f}  rnd {ni_rnd:.3f}")
    print(f"{'k MLPs':>7}{'natural':>9}{'+nats':>8}{'rnd':>9}{'+nats':>8}")
    for k in (2, 4, 6, 8, 12, 16, 20):
        nc = run_sub(test_seq, order[:k])
        nr = run_sub(rnd_test, order[:k])
        print(f"{k:>7}{nc:>9.3f}{nc-ni_nat:>+8.3f}{nr:>9.3f}{nr-ni_rnd:>+8.3f}", flush=True)
