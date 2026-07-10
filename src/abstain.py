"""Experiment E: neutral as an ABSTENTION BAND on the polarity axis.

Rule:  neutral if |polarity| < m   else  sign(polarity).
Fit the margin m on the ORIGINAL 72 (frozen), apply to held-out. Then test the
key claim: does the abstention band coincide with ministral's OWN uncertainty
(its low self-agreement sentences)?
"""
import json, numpy as np
import model

# --- fit margin m on the original corpus only ---
emb0 = np.load("emb.npy")
y0 = np.array([d["label"] for d in json.load(open("labels.json"))])
pol0, _ = model.scores(emb0)
best, m = -1, 0.0
for cand in np.linspace(0, np.abs(pol0).max(), 121):
    pred = np.where(np.abs(pol0) < cand, "neutral",
                    np.where(pol0 > 0, "positive", "negative"))
    acc = (pred == y0).mean()
    if acc > best:
        best, m = acc, cand
print(f"fitted abstention margin m={m:.3f} on original 72 (train fidelity {best:.3f})")

def abstain_predict(pol):
    return np.where(np.abs(pol) < m, "neutral",
                    np.where(pol > 0, "positive", "negative"))

# --- apply frozen to held-out ---
r = json.load(open("consistency.json"))
E = np.load("emb_test.npy")
pol, _ = model.scores(E)
maj = np.array([x["maj"] for x in r])
agree = np.array([x["agree"] for x in r])
old_pred = np.array([x["pred"] for x in r])          # evaluativeness-gate surrogate
new_pred = abstain_predict(pol)

def fid(mask, pred): return (pred[mask] == maj[mask]).mean()
allm = np.ones(len(maj), bool); uni = agree == 1.0
print("\n=== held-out fidelity vs ministral majority: gate vs polarity-band ===")
print(f"  {'subset':<18}{'eval-gate':>10}{'polarity-band':>16}")
for name, msk in [("all", allm), ("unanimous(7/7)", uni), ("ambiguous(<6/7)", agree < 6/7)]:
    print(f"  {name:<18}{fid(msk,old_pred):>10.3f}{fid(msk,new_pred):>16.3f}")

# --- THE key test: does the abstention band = ministral's uncertainty zone? ---
inband = np.abs(pol) < m
print("\n=== does |polarity| track ministral's self-agreement? ===")
print(f"  mean ministral agreement INSIDE  band (surrogate abstains): {agree[inband].mean():.3f}  (n={inband.sum()})")
print(f"  mean ministral agreement OUTSIDE band (surrogate commits):  {agree[~inband].mean():.3f}  (n={(~inband).sum()})")
from scipy.stats import spearmanr
rho, p = spearmanr(np.abs(pol), agree)
print(f"  Spearman(|polarity|, agreement) = {rho:+.3f}  (p={p:.1e})")

amb = agree < 6/7
print(f"\n  of ministral's {amb.sum()} ambiguous sentences, {(amb & inband).sum()} fall in the abstention band "
      f"({(amb & inband).sum()/amb.sum():.0%} caught)")
print(f"  of the {(~amb).sum()} confident sentences, {((~amb) & inband).sum()} fall in the band "
      f"({((~amb) & inband).sum()/(~amb).sum():.0%})")
