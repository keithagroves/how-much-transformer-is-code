"""'Nesses' -- graded semantic dimensions as named axes in qwen space.

A ness = unit(mean(emb(high_seeds)) - mean(emb(low_seeds))). Any word or whole
statement scores as its projection on that axis. Scores are calibrated so the
low-seed mean = -1 and high-seed mean = +1 (readable units).

This file provides the machinery + two example axes (meanness, formality) and,
run as a script, eyeball checks: a word gradient and statement scores.
"""
import json, os, numpy as np, requests

CONTRASTS = {
    "meanness": (
        # high (mean) side
        "cruel vicious spiteful nasty heartless callous brutal malicious".split(),
        # low (kind) side
        "kind gentle caring warm compassionate tender considerate loving".split(),
    ),
    "formality": (
        "herewith pursuant accordingly notwithstanding respectfully endeavour".split(),
        "yeah gonna stuff okay folks kinda".split(),
    ),
}

def _embed(texts):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b", "input": list(texts)}, timeout=300)
    E = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

class Ness:
    def __init__(self, name, high_seeds, low_seeds):
        self.name = name
        hi, lo = _embed(high_seeds), _embed(low_seeds)
        axis = hi.mean(0) - lo.mean(0)
        self.axis = axis / np.linalg.norm(axis)
        # calibrate: low-seed mean -> -1, high-seed mean -> +1
        h, l = float(hi.mean(0) @ self.axis), float(lo.mean(0) @ self.axis)
        self.mid, self.half = (h + l) / 2, (h - l) / 2
    def score_vecs(self, V):
        return ((V @ self.axis) - self.mid) / self.half
    def score(self, texts):
        return self.score_vecs(_embed(texts))

def load_axes():
    return {name: Ness(name, hi, lo) for name, (hi, lo) in CONTRASTS.items()}

if __name__ == "__main__":
    axes = load_axes()
    m = axes["meanness"]

    words = ("sneered snapped barked mocked taunted said remarked replied "
             "whispered smiled soothed comforted praised hugged").split()
    s = m.score(words)
    print("=== word gradient on the meanness axis (calibrated: kind=-1 .. mean=+1) ===")
    for w, v in sorted(zip(words, s), key=lambda t: -t[1]):
        bar = "#" * int(max(0, (v + 1) * 12))
        print(f"  {w:<11} {v:+.2f}  {bar}")

    stmts = [
        "You are a pathetic waste of space and everyone laughs at you.",
        "Honestly, nobody here likes you at all.",
        "That answer was not quite right, but good try.",
        "The meeting is scheduled for three o'clock.",
        "You did a wonderful job and I am proud of you.",
        "You always know how to make everyone feel welcome.",
    ]
    print("\n=== statement meanness (whole-sentence projection) ===")
    for t, v in zip(stmts, m.score(stmts)):
        print(f"  {v:+.2f}  {t}")
