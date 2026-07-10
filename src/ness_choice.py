"""Ness-driven word choice: the graded-lexicon + register architecture.

Rule form:   <class rule picks WHAT KIND of word>  +  <register picks WHICH>
Demo: complete  '<context>. "...," she ___ .'  by choosing the SAY-verb whose
meanness score is closest to the context's meanness register.
"""
import numpy as np
from nesses import load_axes, _embed

axes = load_axes()
m = axes["meanness"]

# the graded lexicon for one word class (computed once; this table IS the rule data)
SAY_VERBS = "sneered snapped barked scoffed said replied remarked murmured whispered soothed".split()
lex = dict(zip(SAY_VERBS, m.score(SAY_VERBS)))

print("=== graded lexicon: class SAY, dimension meanness ===")
for w, v in sorted(lex.items(), key=lambda t: -t[1]):
    print(f"  {w:<10} {v:+.2f}")

def choose(context, cls=lex):
    register = float(m.score([context])[0])          # context meanness register
    word = min(cls, key=lambda w: abs(cls[w] - register))
    return register, word

print("\n=== rule: pick SAY-verb closest to the context register ===")
contexts = [
    "You are a pathetic waste of space and everyone laughs at you.",
    "That was a stupid thing to say.",
    "The quarterly report is due on Friday.",
    "That answer was not quite right, but good try.",
    "You did a wonderful job and I am proud of you.",
]
for c in contexts:
    reg, w = choose(c)
    print(f'  register={reg:+.2f} -> she {w.upper()}   | context: "{c[:52]}"')
