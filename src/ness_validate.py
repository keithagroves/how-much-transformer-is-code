"""Validate the meanness axis against an independent judge (ministral rates
each statement's meanness 1-5 at temp 0). Report Spearman correlation."""
import re, numpy as np, requests
from scipy.stats import spearmanr
from nesses import load_axes

STMTS = [
    "You are a pathetic waste of space and everyone laughs at you.",
    "Nobody would even notice if you disappeared tomorrow.",
    "You always ruin everything you touch.",
    "Honestly, nobody here likes you at all.",
    "That was a stupid thing to say.",
    "You're not exactly the sharpest tool in the shed.",
    "I've seen better work from a first-year intern.",
    "That answer was not quite right, but good try.",
    "Your essay needs work on structure and clarity.",
    "The meeting is scheduled for three o'clock.",
    "The train departs from platform four every hour.",
    "Could you pass the salt, please?",
    "Thanks for coming today, it means a lot.",
    "You handled that situation with real patience.",
    "You did a wonderful job and I am proud of you.",
    "Your kindness to the new students hasn't gone unnoticed.",
    "You always know how to make everyone feel welcome.",
    "You are the most thoughtful person I have ever met.",
    "Everyone says your cooking is terrible, and they are right.",
    "It takes real talent to fail this consistently.",
]

def judge(text):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0},
        "prompt": ("Rate how mean this statement is on a scale of 1 (very kind) "
                   "to 5 (very mean). Reply with only the number.\n\n"
                   f'Statement: "{text}"\nRating:')}, timeout=120)
    m = re.search(r"[1-5]", r.json()["response"])
    return int(m.group()) if m else None

axes = load_axes()
scores = axes["meanness"].score(STMTS)
ratings = [judge(t) for t in STMTS]

print(f"{'axis':>6}  {'judge':>5}  statement")
for t, s, j in sorted(zip(STMTS, scores, ratings), key=lambda x: -x[1]):
    print(f"{s:>+6.2f}  {j:>5}  {t[:62]}")

rho, p = spearmanr(scores, ratings)
print(f"\nSpearman(axis, ministral 1-5 judge) = {rho:+.3f}  (p={p:.1e}, n={len(STMTS)})")
