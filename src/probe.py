"""Geometry probe: understand qwen's latent organization of sentiment BEFORE
authoring rules. Prints prototypes' separation, the pos-neg axis, where neutral
sits, top discriminative dims, and achievable-fidelity ceilings.
"""
import json, numpy as np
from collections import Counter

emb = np.load("emb.npy")                       # (72,1024), L2-normalized
data = json.load(open("labels.json"))
y = np.array([d["label"] for d in data])
classes = ["negative", "neutral", "positive"]

def unit(v): return v / np.linalg.norm(v)

# class prototypes (normalized mean)
proto = {c: unit(emb[y == c].mean(0)) for c in classes}
print("=== prototype cosine similarities (how distinct are the class centers) ===")
for a in classes:
    print("  " + a[:3], " ".join(f"{b[:3]}={proto[a]@proto[b]:+.3f}" for b in classes))

# sentiment axis = positive - negative
axis = unit(proto["positive"] - proto["negative"])
proj = emb @ axis
print("\n=== projection onto (positive - negative) axis, mean +/- std ===")
for c in classes:
    p = proj[y == c]
    print(f"  {c:<9} {p.mean():+.3f} +/- {p.std():.3f}   [{p.min():+.3f}, {p.max():+.3f}]")

# nearest-prototype accuracy (train) and LOO
def nearest_proto_pred(E, protos):
    P = np.vstack([protos[c] for c in classes])
    return np.array(classes)[(E @ P.T).argmax(1)]

train_pred = nearest_proto_pred(emb, proto)
print(f"\n=== nearest-prototype fidelity (train) = {(train_pred==y).mean():.3f} ===")

loo_correct = 0
for i in range(len(emb)):
    mask = np.ones(len(emb), bool); mask[i] = False
    pr = {c: unit(emb[mask & (y == c)].mean(0)) for c in classes}
    loo_correct += nearest_proto_pred(emb[i:i+1], pr)[0] == y[i]
print(f"=== nearest-prototype fidelity (LOO-CV) = {loo_correct/len(emb):.3f} ===")

# ceiling: logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
clf = LogisticRegression(max_iter=2000, C=1.0)
loo_lr = (cross_val_predict(clf, emb, y, cv=LeaveOneOut()) == y).mean()
print(f"=== logistic-regression ceiling (LOO-CV) = {loo_lr:.3f} ===")

# top discriminative dims along the axis
top = np.argsort(-np.abs(axis))[:12]
print("\n=== top 12 dims carrying the sentiment axis (dim: weight) ===")
print("  " + "  ".join(f"{d}:{axis[d]:+.3f}" for d in top))

# where do ministral's calls disagree with a pure axis threshold?
print("\n=== axis value for the tricky #16 (speechless->neutral) ===")
i16 = next(i for i,d in enumerate(data) if "speechless" in d["text"])
print(f"  proj={proj[i16]:+.3f}  label={y[i16]}  text={data[i16]['text']!r}")
