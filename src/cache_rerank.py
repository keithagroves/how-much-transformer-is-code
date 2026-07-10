"""Linguist feature test #2: discourse cohesion (the classic 'cache LM').
Content words are bursty -- once mentioned, they recur. Rule: if any candidate
occurred in the recent window, prefer it; else keep frequency order.
Measured on ALL fixed-test positions (not just content-choice ones).
"""
import sys, re, numpy as np
from collections import defaultdict
from semantic_eval import load_tokens, load_rules, MAXN
from generalize import TOK2POS

by, dflt = load_rules("rules_big.txt")
toks = load_tokens()
test = toks[int(len(toks)*0.9):]
is_content = lambda w: re.fullmatch(r"[a-z]{3,}", w) and w not in TOK2POS

def run(window):
    base = cache = 0; flips_r = flips_w = 0
    n = 0
    for i in range(3, len(test)):
        ctx = test[i-3:i]; truth = test[i]
        cands = None
        for o in range(MAXN, 1, -1):
            k = tuple(ctx[-(o-1):])
            if k in by[o]: cands = by[o][k]; break
        cands = cands or [dflt]
        b = cands[0]
        # cache rule: prefer a CONTENT candidate seen in the recent window
        recent = set(t for t in test[max(0, i-window):i] if is_content(t))
        hits = [c for c in cands if c in recent]
        c = hits[0] if hits else b
        base += b == truth; cache += c == truth; n += 1
        if c != b:
            if c == truth: flips_r += 1
            elif b == truth: flips_w += 1
    return base/n, cache/n, flips_r, flips_w, n

print(f"{'window':>7}{'freq top1':>11}{'cache top1':>12}{'fixed':>7}{'broken':>8}")
for w in (20, 50, 100, 200):
    b, c, fr, fw, n = run(w)
    print(f"{w:>7}{b:>11.3f}{c:>12.3f}{fr:>7}{fw:>8}")
