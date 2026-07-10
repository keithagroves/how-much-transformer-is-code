"""Embed each labeled text with qwen3-embedding:0.6b.

Saves emb.npy (N x D float32) aligned row-for-row with labels.json, so the
surrogate has (embedding, ground-truth token) pairs to fit and be read against.
"""
import json, sys, numpy as np, requests

MODEL = "qwen3-embedding:0.6b"

def embed(text):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": MODEL, "input": text,
    }, timeout=120)
    return np.asarray(r.json()["embeddings"][0], dtype=np.float32)

def main():
    data = json.load(open("labels.json"))
    vecs = []
    for i, d in enumerate(data):
        vecs.append(embed(d["text"]))
        print(f"[{i+1:>2}/{len(data)}] embedded dim={vecs[-1].shape[0]}", file=sys.stderr)
    emb = np.vstack(vecs)
    np.save("emb.npy", emb)
    print(f"\nsaved emb.npy shape={emb.shape} dtype={emb.dtype}", file=sys.stderr)

if __name__ == "__main__":
    main()
