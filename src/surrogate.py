"""Hand-authored surrogate: readable rules over TWO named, interpretable axes
of qwen's latent space, fit to replicate ministral's sentiment token.

  polarity(emb)      = emb . unit(proto_pos - proto_neg)     # which way
  evaluativeness(emb)= emb . unit((proto_pos+proto_neg)/2 - proto_neu)  # opinion at all?

Decision (the whole model, in three lines):
    if evaluativeness < t_eval:  neutral
    elif polarity > t_pol:       positive
    else:                        negative

Everything below just *fits* the two prototypes and the two scalar thresholds,
then measures how faithfully these rules reproduce ministral -- train and LOO.
"""
import json, numpy as np
from itertools import product

emb = np.load("emb.npy")
data = json.load(open("labels.json"))
y = np.array([d["label"] for d in data])
classes = ["negative", "neutral", "positive"]
unit = lambda v: v / np.linalg.norm(v)

def fit_axes(E, Y):
    p = {c: unit(E[Y == c].mean(0)) for c in classes}
    a_pol = unit(p["positive"] - p["negative"])
    a_eval = unit((p["positive"] + p["negative"]) / 2 - p["neutral"])
    return a_pol, a_eval

def scores(E, axes):
    a_pol, a_eval = axes
    return E @ a_pol, E @ a_eval

def predict(pol, ev, t_pol, t_eval):
    out = np.where(ev < t_eval, "neutral",
                   np.where(pol > t_pol, "positive", "negative"))
    return out

def fit_thresholds(pol, ev, Y):
    # Both boundaries are derived, not searched -> stable across folds.
    # polarity boundary = 0, the geometric midpoint of the pos/neg poles.
    # neutrality gate  = midpoint of mean-evaluativeness for neutral vs. the
    #                    evaluative (pos u neg) examples.
    t_pol = 0.0
    ev_neu = ev[Y == "neutral"].mean()
    ev_op = ev[np.isin(Y, ["positive", "negative"])].mean()
    t_eval = (ev_neu + ev_op) / 2
    return t_pol, t_eval

# ---- fit on all data (for the readable, reported rule) ----
axes = fit_axes(emb, y)
pol, ev = scores(emb, axes)
t_pol, t_eval = fit_thresholds(pol, ev, y)
train_pred = predict(pol, ev, t_pol, t_eval)
train_fid = (train_pred == y).mean()

# ---- honest LOO: refit axes + thresholds without example i ----
loo_pred = []
for i in range(len(emb)):
    m = np.ones(len(emb), bool); m[i] = False
    ax = fit_axes(emb[m], y[m])
    pol_i, ev_i = scores(emb[m], ax)
    tp, te = fit_thresholds(pol_i, ev_i, y[m])
    ph, eh = scores(emb[i:i+1], ax)
    loo_pred.append(predict(ph, eh, tp, te)[0])
loo_pred = np.array(loo_pred)
loo_fid = (loo_pred == y).mean()

print("=== the fitted rule (all data) ===")
print(f"  if evaluativeness < {t_eval:+.3f}: neutral")
print(f"  elif polarity     > {t_pol:+.3f}: positive")
print(f"  else                            : negative\n")
print(f"train fidelity to ministral = {train_fid:.3f}  ({(train_pred==y).sum()}/{len(y)})")
print(f"LOO-CV fidelity to ministral = {loo_fid:.3f}  ({(loo_pred==y).sum()}/{len(y)})")
print("  (ceiling was 0.944 for both nearest-proto and logistic regression)\n")

print("=== where the fitted rule disagrees with ministral (all-data model) ===")
for i in np.where(train_pred != y)[0]:
    print(f"  said {train_pred[i]:<8} truth {y[i]:<8} pol={pol[i]:+.3f} ev={ev[i]:+.3f} | {data[i]['text']}")
