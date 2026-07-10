"""The frozen surrogate. Fits the two named axes + principled thresholds on the
ORIGINAL corpus (emb.npy / labels.json) and exposes predict() for new embeddings.
Import this anywhere; fitting happens once at import from the original data only.
"""
import json, numpy as np

_CLASSES = ["negative", "neutral", "positive"]
_unit = lambda v: v / np.linalg.norm(v)

def _fit(emb, y):
    proto = {c: _unit(emb[y == c].mean(0)) for c in _CLASSES}
    a_pol = _unit(proto["positive"] - proto["negative"])
    a_eval = _unit((proto["positive"] + proto["negative"]) / 2 - proto["neutral"])
    ev = emb @ a_eval
    t_eval = (ev[y == "neutral"].mean() +
              ev[np.isin(y, ["positive", "negative"])].mean()) / 2
    return a_pol, a_eval, 0.0, t_eval  # t_pol pinned at 0

# fit once, on the original corpus only
_emb = np.load("emb.npy")
_y = np.array([d["label"] for d in json.load(open("labels.json"))])
A_POL, A_EVAL, T_POL, T_EVAL = _fit(_emb, _y)

def scores(emb):
    return emb @ A_POL, emb @ A_EVAL

def predict(emb):
    emb = np.atleast_2d(emb)
    pol, ev = scores(emb)
    return np.where(ev < T_EVAL, "neutral",
                    np.where(pol > T_POL, "positive", "negative"))

if __name__ == "__main__":
    print(f"frozen surrogate: t_pol={T_POL:+.3f} t_eval={T_EVAL:+.3f}")
    p = predict(_emb)
    print(f"self-check train fidelity = {(p == _y).mean():.3f}")
