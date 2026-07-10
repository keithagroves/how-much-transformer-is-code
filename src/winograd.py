"""Can 'rules pointing at the space' do Winograd-style pronoun resolution?

Meta-rule (3 lines, legible): substitute each candidate noun for the pronoun;
embed both readings; resolve to the candidate whose reading is most similar to
the original sentence. World knowledge = the geometry, not an enumeration.

Each schema has two polarity versions, so surface cues cancel: a system using
word association alone scores 50%.
Comparison: ministral asked directly (the neural ceiling on this hardware).
"""
import re, numpy as np, requests

# (sentence, candidate A, candidate B, correct)
ITEMS = [
 ("The trophy didn't fit in the suitcase because it was too big.", "trophy", "suitcase", "trophy"),
 ("The trophy didn't fit in the suitcase because it was too small.", "trophy", "suitcase", "suitcase"),
 ("The ball broke the table because it was made of steel.", "ball", "table", "ball"),
 ("The ball broke the table because it was made of cardboard.", "ball", "table", "table"),
 ("The man couldn't lift his son because he was so weak.", "man", "son", "man"),
 ("The man couldn't lift his son because he was so heavy.", "man", "son", "son"),
 ("The truck zoomed past the bus because it was going so fast.", "truck", "bus", "truck"),
 ("The truck zoomed past the bus because it was going so slowly.", "truck", "bus", "bus"),
 ("The cat caught the mouse because it was quick.", "cat", "mouse", "cat"),
 ("The cat caught the mouse because it was slow.", "cat", "mouse", "mouse"),
 ("The fish ate the worm because it was hungry.", "fish", "worm", "fish"),
 ("The fish ate the worm because it was tasty.", "fish", "worm", "worm"),
 ("I poured water from the bottle into the cup until it was full.", "bottle", "cup", "cup"),
 ("I poured water from the bottle into the cup until it was empty.", "bottle", "cup", "bottle"),
 ("The knight killed the dragon because it was evil.", "knight", "dragon", "dragon"),
 ("The knight killed the dragon because it was brave.", "knight", "dragon", "knight"),
]

def embed(texts):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b", "input": texts}, timeout=300)
    E = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def substitute(sent, cand):
    s = re.sub(r"\bit was\b", f"the {cand} was", sent, count=1)
    if s == sent:
        s = re.sub(r"\bhe was\b", f"the {cand} was", sent, count=1)
    return s

def ministral(sent, a, b):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False, "options": {"temperature": 0},
        "prompt": (f'Sentence: "{sent}"\nDoes the pronoun refer to the {a} or the {b}? '
                   f"Answer with exactly one word: {a} or {b}.\nAnswer:")}, timeout=120)
    resp = r.json()["response"].strip().lower()
    return a if a in resp and b not in resp else (b if b in resp else "?")

geo_ok = neu_ok = 0
print(f"{'geo':>4} {'neural':>7}  sentence")
for sent, a, b, gold in ITEMS:
    E = embed([sent, substitute(sent, a), substitute(sent, b)])
    geo = a if E[0] @ E[1] > E[0] @ E[2] else b
    neu = ministral(sent, a, b)
    geo_ok += geo == gold; neu_ok += neu == gold
    print(f"{'Y' if geo==gold else '.':>4} {'Y' if neu==gold else '.':>7}  {sent[:64]} -> {gold}")
n = len(ITEMS)
print(f"\ngeometric substitution rule: {geo_ok}/{n} = {geo_ok/n:.2f}   (chance 0.50)")
print(f"ministral asked directly   : {neu_ok}/{n} = {neu_ok/n:.2f}")
