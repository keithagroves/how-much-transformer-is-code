"""Demo: generate text from the rule program, with a legible trace.

Every token is produced by a visible rule: which context fired, at what order,
what the candidates were, and what was sampled. No neural net at generation
time -- the entire model is a table you can read.

  python3 demo.py "she opened the door and"
  python3 demo.py "the storm hit the village" --tokens 50 --temp 0.7
  python3 demo.py "he looked at her and said" --trace
"""
import argparse, re, sys
import numpy as np
from collections import defaultdict, Counter

MAXN = 4; MIN_RULE = 3; TOPC = 8

def build():
    text = open("ministral_corpus.txt", encoding="utf-8").read().replace("<|doc|>", " ")
    toks = re.findall(r"[a-z]+|[^\w\s]", text.lower())
    cc = defaultdict(Counter)
    for i in range(len(toks)):
        for o in range(2, MAXN + 1):
            if i - o + 1 >= 0:
                cc[tuple(toks[i - o + 1:i])][toks[i]] += 1
    rules = {k: c.most_common(TOPC) for k, c in cc.items() if sum(c.values()) >= MIN_RULE}
    print(f"[rulebook: {len(rules):,} rules from {len(toks):,} tokens]\n", file=sys.stderr)
    return rules

NO_SPACE_BEFORE = set(",.!?;:)'’”")
NO_SPACE_AFTER = set("(“‘")

def detok(tokens):
    out = ""
    cap = True
    for t in tokens:
        if t in NO_SPACE_BEFORE or (out and out[-1] in NO_SPACE_AFTER):
            out += t
        else:
            out += (" " if out else "") + t
        if cap and t.isalpha():
            out = out[:-len(t)] + t.capitalize()
            cap = False
        if t in ".!?":
            cap = True
    return out

def generate(rules, seed, n=40, temp=0.8, trace=False, rng=None):
    rng = rng or np.random.default_rng()
    toks = re.findall(r"[a-z]+|[^\w\s]", seed.lower())
    lines = []
    for step in range(n):
        cands, order = None, 0
        for o in range(MAXN, 1, -1):
            k = tuple(toks[-(o - 1):])
            if len(k) == o - 1 and k in rules:
                cands, order = rules[k], o
                break
        if cands is None:
            cands, order = [(",", 1)], 0
        words = [w for w, _ in cands]
        counts = np.array([c for _, c in cands], dtype=np.float64)
        p = counts ** (1.0 / max(temp, 0.05)); p /= p.sum()
        pick = rng.choice(len(words), p=p)
        if trace:
            ctx = " ".join(toks[-(order - 1):]) if order else "(no rule)"
            shown = "  ".join(f"{w}:{int(c)}" for w, c in cands[:5])
            lines.append(f"  [{step:>2}] order-{order} after {ctx!r:<24} -> {words[pick]!r:<12} from {{{shown}}}")
        toks.append(words[pick])
    if trace:
        print("\n".join(lines))
    return detok(toks)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--seed-rng", type=int, default=None)
    a = ap.parse_args()
    rules = build()
    rng = np.random.default_rng(a.seed_rng)
    print(generate(rules, a.seed, a.tokens, a.temp, a.trace, rng))
