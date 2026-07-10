"""Experiment D: the tunability knob. Sweep the abstention margin m and trace
precision vs coverage on the held-out set. This is the 'business requirement'
dial: pick a precision floor, read off the coverage you keep; the rest abstains
(route to a human / a bigger model).

Metrics (held-out, vs ministral majority label):
  coverage  = fraction of sentences the surrogate COMMITS on (|polarity| >= m)
  precision = of committed predictions, fraction matching ministral majority
              (committing on a neutral-majority sentence counts as a miss)
"""
import json, numpy as np, model
r = json.load(open("consistency.json"))
E = np.load("emb_test.npy")
pol, _ = model.scores(E)
maj = np.array([x["maj"] for x in r])
N = len(maj)

def eval_m(m):
    committed = np.abs(pol) >= m
    if committed.sum() == 0:
        return 0.0, float("nan")
    pred = np.where(pol[committed] > 0, "positive", "negative")
    prec = (pred == maj[committed]).mean()
    return committed.mean(), prec

print(f"{'margin m':>9}{'coverage':>10}{'precision':>11}{'committed':>11}")
grid = np.linspace(0, np.abs(pol).max() * 0.9, 19)
for m in grid:
    cov, prec = eval_m(m)
    print(f"{m:>9.3f}{cov:>10.0%}{prec:>11.3f}{int(round(cov*N)):>8}/{N}")

# operating points: smallest m achieving a precision floor
print("\n=== operating points (min margin hitting a precision floor) ===")
fine = np.linspace(0, np.abs(pol).max(), 400)
rows = []
for floor in [0.90, 0.95, 0.99, 1.00]:
    hit = None
    for m in fine:
        cov, prec = eval_m(m)
        if cov > 0 and prec >= floor:
            hit = (m, cov, prec); break
    if hit:
        print(f"  precision >= {floor:.2f}:  m={hit[0]:.3f}  keep {hit[1]:.0%} coverage  (actual prec {hit[2]:.3f})")
        rows.append([float(floor), float(hit[0]), float(hit[1]), float(hit[2])])

# dump curve for the writeup
curve = [{"m": float(m), "coverage": float(eval_m(m)[0]), "precision": float(eval_m(m)[1])}
         for m in fine if not np.isnan(eval_m(m)[1])]
json.dump({"curve": curve, "operating_points": rows, "N": N}, open("sweep.json", "w"))
print(f"\nsaved sweep.json ({len(curve)} points)")
