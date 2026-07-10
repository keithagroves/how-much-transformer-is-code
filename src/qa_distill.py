"""Q&A distillation: distill ministral's question-answering into a rulebook.

Facts stay LEXICAL (no slotting) -- the bindings ARE the payload:
  context ('of','france','is') -> 'paris'  is the fact, stored as a rule.
Runtime answers questions with zero LLM calls. Arithmetic included as a
boundary probe: seen sums should be memorized, unseen sums should fail.

  usage: python3 qa_distill.py collect   # ask ministral, save qa_pairs
         python3 qa_distill.py eval      # build rulebook, answer, score
"""
import json, os, re, sys, requests
from collections import defaultdict, Counter

MAXN = 7   # entity must stay in view: legs/sums need 6; margin 7

COUNTRIES = ("france germany italy spain portugal japan china india brazil canada egypt "
    "kenya norway sweden poland greece turkey thailand vietnam peru chile cuba ireland "
    "scotland austria hungary finland denmark morocco argentina").split()
UNSEEN_COUNTRIES = "mexico russia australia iceland colombia".split()
ANIMALS = {"spider": 8, "ant": 6, "bee": 6, "dog": 4, "cat": 4, "horse": 4,
           "cow": 4, "chicken": 2, "snake": 0, "octopus": 8, "crab": 10, "fly": 6}
OPPOSITES = "hot big fast up light happy wet hard old tall empty early loud strong clean".split()
UNSEEN_OPP = "cold small quiet".split()
SUMS = [(a, b) for a in (3, 7, 12, 25, 48) for b in (4, 9, 16, 33)]
UNSEEN_SUMS = [(6, 8), (14, 27), (52, 19)]

def questions():
    qs = []
    for c in COUNTRIES: qs.append(("capital", f"What is the capital of {c.title()}?"))
    for a in ANIMALS:   qs.append(("legs", f"How many legs does a {a} have?"))
    for w in OPPOSITES: qs.append(("opposite", f"What is the opposite of {w}?"))
    for a, b in SUMS:   qs.append(("sum", f"What is {a} plus {b}?"))
    return qs

def ask(q):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0, "num_predict": 50},
        "prompt": f"{q} Answer with only the single word or number, nothing else."}, timeout=120)
    a = r.json().get("response", "").strip()
    a = re.split(r"(?<=[.!])\s", a)[0]          # first sentence only
    return re.sub(r"[*#_`]+", "", a).strip()

tok = lambda s: re.findall(r"[a-z0-9]+|[^\w\s]", s.lower())

if __name__ == "__main__" and sys.argv[1:2] == ["collect"]:
    pairs = []
    qs = questions()
    for i, (fam, q) in enumerate(qs):
        a = ask(q)
        pairs.append({"family": fam, "q": q, "a": a})
        print(f"[{i+1:>3}/{len(qs)}] {q}  ->  {a}", file=sys.stderr)
    json.dump(pairs, open("qa_pairs.json", "w"), indent=2)
    print(f"saved {len(pairs)} pairs")
    sys.exit()

# ---------------- eval ----------------
pairs = json.load(open("qa_pairs.json"))
stream = []
for p in pairs:
    stream += ["q", ":"] + tok(p["q"]) + ["a", ":"] + tok(p["a"]) + ["<end>"]

rules = defaultdict(Counter)
for i in range(len(stream)):
    for o in range(2, MAXN+1):
        if i-o+1 >= 0: rules[tuple(stream[i-o+1:i])][stream[i]] += 1
rb = {k: c.most_common(1)[0][0] for k, c in rules.items()}   # MIN_RULE=1: facts occur once
print(f"rulebook: {len(rb):,} rules from {len(pairs)} Q-A pairs", file=sys.stderr)

def answer(q, maxlen=25):
    ctx = ["q", ":"] + tok(q) + ["a", ":"]
    out = []
    for _ in range(maxlen):
        w = None
        for o in range(MAXN, 1, -1):
            k = tuple(ctx[-(o-1):])
            if k in rb: w = rb[k]; break
        if w is None or w == "<end>": break
        out.append(w); ctx.append(w)
    return " ".join(out)

def norm(s): return " ".join(tok(s))

fid = Counter(); tot = Counter()
for p in pairs:
    got = answer(p["q"])
    ok = got == norm(p["a"])
    fid[p["family"]] += ok; tot[p["family"]] += 1
print("\n=== fidelity to ministral on TRAINED questions (rule answers, no LLM) ===")
for f in tot:
    print(f"  {f:<9} {fid[f]}/{tot[f]}  ({fid[f]/tot[f]:.0%})")

print("\n=== boundary: UNSEEN questions (should fail -- honesty check) ===")
for c in UNSEEN_COUNTRIES[:3]:
    print(f"  Q: capital of {c.title()}?  ->  {answer(f'What is the capital of {c.title()}?')!r}")
for w in UNSEEN_OPP[:2]:
    print(f"  Q: opposite of {w}?        ->  {answer(f'What is the opposite of {w}?')!r}")
for a, b in UNSEEN_SUMS[:3]:
    print(f"  Q: {a} plus {b}?           ->  {answer(f'What is {a} plus {b}?')!r}")

print("\n=== sample trained answers ===")
for p in pairs[::31][:6]:
    print(f"  Q: {p['q']}")
    print(f"     rule answer: {answer(p['q'])!r}")
