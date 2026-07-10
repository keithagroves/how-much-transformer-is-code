"""Chapter 2, step 8d: richer template library.

Adds codeable column types the first library lacked:
    dup     duplicate-token: uniform over earlier occurrences of a token
            similar to the CURRENT token (known head type)
    sstart  first token of the current sentence
    sent    uniform over the current sentence so far
    psent   uniform over the previous sentence
    lstart  first token of the current line
Then refit all heads (least squares), re-measure solo costs, and rebuild the
frontier on held-out text.
"""
import torch
import replace_all as RA

model, DEV, DH, GROUP, NL, NH, T = RA.model, RA.DEV, RA.DH, RA.GROUP, RA.NL, RA.NH, RA.T
tokz = RA.tokz
train_seq, test_seq, rnd_test = RA.train_seq, RA.test_seq, RA.rnd_test

SENT_END = {t for t in range(300) if any(c in tokz.decode([t]) for c in ".!?\n")}
LINE_END = {t for t in range(300) if "\n" in tokz.decode([t])}

def rich_columns(seq):
    cols, _ = RA.template_columns(seq)
    n = len(seq)
    # token similarity for dup (reuse fuzzy machinery's string/emb classes)
    uniq = sorted(set(seq)); idx = {t: k for k, t in enumerate(uniq)}
    E = RA.EMB[torch.tensor(uniq)]; E = E / E.norm(dim=-1, keepdim=True)
    S = (E @ E.T).clamp(min=0)
    norm = [tokz.decode([t]).strip().lower().strip('.,"\'') for t in uniq]
    bystr = {}
    for k, s_ in enumerate(norm): bystr.setdefault(s_, []).append(k)
    for ks in bystr.values():
        for a in ks:
            for b in ks:
                if a != b: S[a, b] = max(S[a, b], torch.tensor(0.9))
    S.fill_diagonal_(1.0)
    ids = [idx[t] for t in seq]
    dup = torch.zeros(n, n)
    occ = {}
    for i in range(n):
        for j in occ.get(True, []):
            pass
    for i in range(1, n):
        for j in range(max(0, 0), i):
            if S[ids[i], ids[j]] >= 0.85: dup[i, j] = 1
    cols["dup"] = dup
    # sentence / line structure
    sid = torch.zeros(n, dtype=torch.long); lid = torch.zeros(n, dtype=torch.long)
    s = l = 0
    for i in range(n):
        sid[i] = s; lid[i] = l
        if seq[i] in SENT_END: s += 1
        if seq[i] in LINE_END: l += 1
    sstart = torch.zeros(n, n); sent = torch.zeros(n, n)
    psent = torch.zeros(n, n); lstart = torch.zeros(n, n)
    first_of_sent = {}
    first_of_line = {}
    for i in range(n):
        first_of_sent.setdefault(int(sid[i]), i)
        first_of_line.setdefault(int(lid[i]), i)
    for i in range(1, n):
        cs = int(sid[i])
        sstart[i, first_of_sent[cs]] = 1
        lstart[i, first_of_line[int(lid[i])]] = 1
        same = (sid[:i] == cs)
        if same.any(): sent[i, :i][same] = 1
        if cs > 0:
            prev = (sid[:i] == cs - 1)
            if prev.any(): psent[i, :i][prev] = 1
    cols.update(dup=dup, sstart=sstart, sent=sent, psent=psent, lstart=lstart)
    return cols

def code_attn(seq):
    cols = rich_columns(seq)
    return {k: v / v.sum(-1, keepdim=True).clamp(min=1e-9) for k, v in cols.items()}

W = {}
def run_sub(seq, heads, base):
    n = len(seq)
    A = {}
    for (l, h) in heads:
        M = torch.zeros(n, n)
        for k, wk in W[(l, h)].items():
            if wk > 1e-4: M += wk * base[k]
        A[(l, h)] = (M / M.sum(-1, keepdim=True).clamp(min=1e-9)).to(DEV)
    by_layer = {}
    for l, h in heads: by_layer.setdefault(l, []).append(h)
    vcache, hooks = {}, []
    for l, hs in by_layer.items():
        attn = model.model.layers[l].self_attn
        def vhook(mod, inp, outp, l=l): vcache[l] = outp[0].detach()
        hooks.append(attn.v_proj.register_forward_hook(vhook))
        def ohook(mod, inp, l=l, hs=hs):
            x = inp[0].clone()
            for h in hs:
                g = h // GROUP
                x[0, :, h*DH:(h+1)*DH] = A[(l, h)] @ vcache[l][:, g*DH:(g+1)*DH]
            return (x,) + inp[1:]
        hooks.append(attn.o_proj.register_forward_pre_hook(ohook))
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1).cpu()
    for hk in hooks: hk.remove()
    return -lp.gather(-1, torch.tensor(seq[1:]).unsqueeze(-1)).mean().item()

if __name__ == "__main__":
    print("building rich templates + fitting...")
    base_tr = code_attn(train_seq)
    names = list(base_tr.keys())
    rows = slice(50, T)
    X = torch.stack([base_tr[k][rows].flatten() for k in names])
    XXt = X @ X.T + 1e-4 * torch.eye(len(names))
    with torch.no_grad():
        out = model(torch.tensor([train_seq]).to(DEV), output_attentions=True)
    atts = [a[0].float().cpu() for a in out.attentions]
    del out
    r2s = []
    for l in range(NL):
        for h in range(NH):
            y = atts[l][h][rows].flatten()
            w = torch.linalg.solve(XXt, X @ y).clamp(min=0)
            W[(l, h)] = {k: float(wk) for k, wk in zip(names, w)}
            pred = (w.unsqueeze(-1) * X).sum(0)
            r2s.append(1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum())
    r2s = torch.tensor(r2s)
    print(f"attention R^2: median {r2s.median():.2f}, >=0.8: {(r2s>=0.8).sum()} heads")
    torch.save(W, "rich_templates.pt")

    ni_tr = run_sub(train_seq, [], base_tr)
    print(f"measuring solo costs ({NL*NH} forwards)...")
    costs = {}
    for l in range(NL):
        for h in range(NH):
            costs[(l, h)] = run_sub(train_seq, [(l, h)], base_tr) - ni_tr
    torch.save(costs, "rich_solo_costs.pt")

    order = sorted(costs, key=lambda k: costs[k])
    base_te, base_rn = code_attn(test_seq), code_attn(rnd_test)
    ni_nat = run_sub(test_seq, [], base_te)
    ni_rnd = run_sub(rnd_test, [], base_rn)
    print(f"\nheld-out intact: natural {ni_nat:.3f}  rnd {ni_rnd:.3f}")
    print(f"{'k':>5}{'natural':>9}{'+nats':>8}{'rnd':>9}{'+nats':>8}")
    for k in (32, 64, 96, 128, 160, 192, 224, 288):
        heads = order[:k]
        nc = run_sub(test_seq, heads, base_te)
        nr = run_sub(rnd_test, heads, base_rn)
        print(f"{k:>5}{nc:>9.3f}{nc-ni_nat:>+8.3f}{nr:>9.3f}{nr-ni_rnd:>+8.3f}", flush=True)
