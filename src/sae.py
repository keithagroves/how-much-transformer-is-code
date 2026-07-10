"""Discovered basis: sparse dictionary learning over vocab embeddings ->
candidate directions -> coherence gate (vs random-direction null) -> gemma
auto-naming -> discovered_scales.json.

The gate matters: ANY direction's extremes look thematic in a dense semantic
space (we proved this with random directions), so a discovered direction only
counts if its top words are MORE mutually similar than random directions' are.
"""
import json, re, sys, numpy as np, requests

K = 64          # dictionary size
TOP = 8         # words defining a component
Z_KEEP = 3.0    # coherence z-score vs random-null needed to survive

words = json.load(open("vocab_words.json"))
E = np.load("vocab_emb.npy").astype(np.float64)
n, d = E.shape
print(f"vocab {n} x {d}", file=sys.stderr)

# ---- learn sparse dictionary ----
from sklearn.decomposition import MiniBatchDictionaryLearning
dl = MiniBatchDictionaryLearning(n_components=K, alpha=0.5, batch_size=1024,
                                 max_iter=60, random_state=0, positive_code=True,
                                 fit_algorithm="cd", transform_algorithm="lasso_cd")
codes = dl.fit_transform(E)                     # (n, K) sparse activations
D = dl.components_                              # (K, d) directions
D = D / np.linalg.norm(D, axis=1, keepdims=True)
print(f"dictionary learned; mean nonzeros/word: {(codes>1e-6).sum(1).mean():.1f}", file=sys.stderr)

def top_words(j, k=TOP):
    idx = np.argsort(-codes[:, j])[:k]
    return [words[i] for i in idx], codes[idx, j]

def coherence(ws):
    V = E[[words.index(w) for w in ws]]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    return S[np.triu_indices(len(ws), 1)].mean()

# ---- null: coherence of top words of random directions ----
rng = np.random.default_rng(1)
null = []
for _ in range(200):
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    idx = np.argsort(-(E @ r))[:TOP]
    V = E[idx]; V = V/np.linalg.norm(V,axis=1,keepdims=True)
    S = V @ V.T
    null.append(S[np.triu_indices(TOP,1)].mean())
mu, sd = float(np.mean(null)), float(np.std(null))
print(f"random-direction null coherence: {mu:.3f} +/- {sd:.3f}", file=sys.stderr)

# ---- gate components ----
survivors = []
for j in range(K):
    if (codes[:, j] > 1e-6).sum() < TOP: continue
    ws, act = top_words(j)
    c = coherence(ws)
    z = (c - mu) / sd
    if z >= Z_KEEP:
        survivors.append({"comp": j, "z": round(float(z), 1),
                          "coherence": round(float(c), 3), "words": ws})
survivors.sort(key=lambda s: -s["z"])
print(f"survivors: {len(survivors)}/{K} pass z>={Z_KEEP}", file=sys.stderr)

# ---- auto-name via gemma ----
def name_of(ws):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma4:e4b", "stream": False,
        "options": {"temperature": 0.2, "num_predict": 24},
        "prompt": ("These words share one property or theme: "
                   + ", ".join(ws) +
                   ". Name that property in 1-3 lowercase words. Reply with only the name.")},
        timeout=300)
    nm = r.json().get("response", "").strip().lower()
    return re.sub(r"[^a-z -]", "", nm)[:40] or "unnamed"

for s in survivors:
    s["name"] = name_of(s["words"])
    print(f"  z={s['z']:>5}  {s['name']:<28} {', '.join(s['words'])}")

json.dump(survivors, open("discovered_scales.json", "w"), indent=2)
print(f"\nwrote discovered_scales.json ({len(survivors)} named directions)")
