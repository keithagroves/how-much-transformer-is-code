"""Arbitrary graded scales ("nesses"), created on demand and validated before
acceptance.

  python3 scales.py create urgency          # gemma seeds it, split-half validated
  python3 scales.py create spookiness
  python3 scales.py list
  python3 scales.py gradient urgency now later whenever immediately someday
  python3 scales.py score urgency "Drop everything, the server is on fire."

A scale = seed contrast sets -> unit(mean(high) - mean(low)) axis in qwen space,
calibrated so seed means sit at +/-1. VALIDITY GATE: axis built from half the
seeds must place >=80% of the HELD-OUT seeds on their correct side; otherwise
the scale is rejected (some concepts don't embed as one direction -- we refuse
to pretend).  Accepted scales persist in scales.json.
"""
import json, os, re, sys, numpy as np, requests

REGISTRY = "scales.json"

def _embed(texts):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "qwen3-embedding:0.6b", "input": list(texts)}, timeout=300)
    E = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)

def gen_seeds(name):
    """gemma4 proposes seed words for both ends of the scale."""
    def ask(end):
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b", "stream": False,
            "options": {"temperature": 0.4, "num_predict": 200},
            "prompt": (f"List 12 single English words that strongly express "
                       f"{end} {name}. Lowercase, one word per line, no "
                       f"numbering, no punctuation, no explanations.")}, timeout=300)
        words = re.findall(r"^[a-z]+$", r.json()["response"], re.M)
        return list(dict.fromkeys(words))[:12]
    return ask("HIGH"), ask("very LOW (the opposite end of)")

def validate(high, low):
    """Split-half: axis from even-indexed seeds, test odd-indexed held-out."""
    Eh, El = _embed(high), _embed(low)
    tr_h, te_h = Eh[::2], Eh[1::2]
    tr_l, te_l = El[::2], El[1::2]
    axis = tr_h.mean(0) - tr_l.mean(0); axis /= np.linalg.norm(axis)
    mid = (tr_h.mean(0) @ axis + tr_l.mean(0) @ axis) / 2
    sh, sl = te_h @ axis - mid, te_l @ axis - mid
    acc = (np.concatenate([sh > 0, sl < 0])).mean()
    d = (sh.mean() - sl.mean()) / (np.concatenate([sh, sl]).std() + 1e-9)
    return float(acc), float(d)

def build(high, low):
    Eh, El = _embed(high), _embed(low)
    axis = Eh.mean(0) - El.mean(0); axis /= np.linalg.norm(axis)
    h, l = float(Eh.mean(0) @ axis), float(El.mean(0) @ axis)
    return axis, (h + l) / 2, (h - l) / 2

def registry():
    return json.load(open(REGISTRY)) if os.path.exists(REGISTRY) else {}

def save_scale(name, high, low, acc, d):
    reg = registry()
    reg[name] = {"high": high, "low": low, "heldout_acc": acc, "cohens_d": d}
    json.dump(reg, open(REGISTRY, "w"), indent=2)

# reference distribution: ~120 ordinary words; z-calibration makes 0 = "a
# typical word" and units = std-devs, comparable across scales (seed-midpoint
# calibration breaks when a seed set is off-center, e.g. a bad LOW list).
REF_WORDS = ("time year people way day man thing woman life child world school "
             "state family student group country problem hand part place case "
             "week company system program question work government number night "
             "point home water room mother area money story fact month lot right "
             "study book eye job word business issue side kind head house service "
             "friend father power hour game line end member law car city community "
             "name president team minute idea body information back parent face "
             "others level office door health person art war history party result "
             "change morning reason research girl guy moment air teacher force "
             "education foot boy age policy everything process music market sense "
             "nation plan college interest death experience effect use class").split()
_REF_CACHE = {}

class Scale:
    def __init__(self, name):
        spec = registry()[name]
        self.name = name
        self.axis, _, _ = build(spec["high"], spec["low"])
        if "ref" not in _REF_CACHE:
            _REF_CACHE["ref"] = _embed(REF_WORDS)
        ref = _REF_CACHE["ref"] @ self.axis
        self.mu, self.sd = float(ref.mean()), float(ref.std())
    def score(self, texts):
        return ((_embed(texts) @ self.axis) - self.mu) / self.sd

def create(name):
    high, low = gen_seeds(name)
    print(f"seeds high: {' '.join(high)}")
    print(f"seeds low : {' '.join(low)}")
    if len(high) < 8 or len(low) < 8:
        print("REJECTED: seed generation too thin"); return False
    acc, d = validate(high, low)
    print(f"split-half validation: held-out accuracy={acc:.2f}  separation d={d:.2f}")
    if acc < 0.80:
        print(f"REJECTED: '{name}' does not embed as a single direction"); return False
    save_scale(name, high, low, acc, d)
    print(f"ACCEPTED -> scales.json ('{name}')")
    return True

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "create":
        create(sys.argv[2])
    elif cmd == "list":
        for n, s in registry().items():
            print(f"  {n:<14} heldout_acc={s['heldout_acc']:.2f}  d={s['cohens_d']:.2f}")
    elif cmd == "gradient":
        sc = Scale(sys.argv[2]); words = sys.argv[3:]
        for w, v in sorted(zip(words, sc.score(words)), key=lambda t: -t[1]):
            print(f"  {w:<14} {v:+.2f}")
    elif cmd == "score":
        sc = Scale(sys.argv[2])
        for t in sys.argv[3:]:
            print(f"  {float(sc.score([t])[0]):+.2f}  {t}")
