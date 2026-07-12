"""Reviewer circularity check on D2 ≈ D3: heads entered the substituted set by
solo cost under the FULL DSL, so the set is enriched for positionally-replaceable
heads, and "content operators add nothing in this set" could be true by
construction. Clean test: re-select per tier — rank every head by its solo cost
substituting under THAT tier's instructions only, give each tier its own
cheapest-k, and rebuild the (attention-only, unhealed, held-out) frontier.
If D3-selected/D3-substituted still ties D2-selected/D2-substituted, the
positional finding is not a selection artifact.
Run from src/ (fiction corpus + fiction rich_templates.pt).
"""
import csv, torch
import replace_rich as RR
import replace_all as RA

tokz, train_seq, test_seq = RA.tokz, RA.train_seq, RA.test_seq
NL, NH = RA.NL, RA.NH

W_FULL = torch.load("rich_templates.pt")
POSITIONAL = {"bos", "self"} | {f"off{i}" for i in range(1, 17)}
STRUCTURAL = POSITIONAL | {"punct", "psent", "sent", "sstart", "lstart"}
CONTENT = STRUCTURAL | {"dup", "rule"}
TIERS = [("D1_positional", POSITIONAL), ("D2_structural", STRUCTURAL), ("D3_content", CONTENT)]

def masked(allowed):
    return {k: {n: (w if n in allowed else 0.0) for n, w in v.items()} for k, v in W_FULL.items()}

base_tr = RR.code_attn(train_seq)
base_te = RR.code_attn(test_seq)
ni_tr = None
results = {}
orders = {}
for name, allowed in TIERS:
    RR.W.clear(); RR.W.update(masked(allowed))
    if ni_tr is None:
        ni_tr = RR.run_sub(train_seq, [], base_tr)
        ni_te = RR.run_sub(test_seq, [], base_te)
        print(f"train intact {ni_tr:.4f}  test intact {ni_te:.4f}", flush=True)
    costs = {}
    for l in range(NL):
        for h in range(NH):
            costs[(l, h)] = RR.run_sub(train_seq, [(l, h)], base_tr) - ni_tr
        if l % 7 == 6: print(f"  {name} solo scan through L{l}", flush=True)
    orders[name] = sorted(costs, key=lambda k: costs[k])
    print(f"{name}: solo scan done", flush=True)

rows = []
KS = [40, 80, 120, 160, 200, 224]
print(f"\nheld-out frontier, each tier with ITS OWN selection (unhealed, attention-only)")
print(f"{'k':>5}" + "".join(f"{n.split('_')[0]:>12}" for n, _ in TIERS))
for k in KS:
    vals = []
    for name, allowed in TIERS:
        RR.W.clear(); RR.W.update(masked(allowed))
        d = RR.run_sub(test_seq, orders[name][:k], base_te) - ni_te
        vals.append(d)
    rows.append([k] + [round(v, 4) for v in vals])
    print(f"{k:>5}" + "".join(f"{v:>+12.3f}" for v in vals), flush=True)

# overlap of selected sets at k=160
s1, s3 = set(orders["D1_positional"][:160]), set(orders["D3_content"][:160])
print(f"\nselection overlap D1 vs D3 at k=160: {len(s1 & s3)}/160")
with open("dsl_tier_selection.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["k"] + [n for n, _ in TIERS]); w.writerows(rows)
print("wrote dsl_tier_selection.csv")
print("read: if D3 ~= D2 with per-tier selection, 'content buys nothing' is not a selection artifact;")
print("      if D3 pulls ahead, the original frontier understated content because selection excluded")
print("      the heads content helps.")
