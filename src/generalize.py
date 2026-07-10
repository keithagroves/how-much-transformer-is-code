"""Step 3: reorganize the flat rule list into GRAMMAR PATTERNS + EXCEPTIONS.

Tag each context token with a coarse part-of-speech (closed-class lists, self-
contained). Build POS-pattern rules (POS-signature -> top next token). Then a
concrete lexical rule survives only as an EXCEPTION if it disagrees with its POS
pattern; everything a pattern already predicts is deleted (lossless on seen
contexts, and patterns add coverage on unseen ones).
"""
import re, sys
from collections import defaultdict, Counter

MAXN = 4
CLASSES = {
 "DET": "the a an this that these those his her its their my your our some any no every each all such another".split(),
 "PREP": "of in on at to from by with for about into over under after before between through upon without within against toward towards among during than as".split(),
 "PRON": "i he she it we they you me him them us who which what whom whose".split(),
 "CONJ": "and but or nor yet so because though although while if when whereas since unless".split(),
 "AUX": "was were is are am be been being had have has do does did would could should will shall may might must can".split(),
}
TOK2POS = {w: p for p, ws in CLASSES.items() for w in ws}
def pos(tok):
    if tok in TOK2POS: return TOK2POS[tok]
    if re.fullmatch(r"[^\w\s]", tok): return tok        # punctuation = its own class
    return "OPEN"
def sig(ctx): return tuple(pos(t) for t in ctx)

def load_tokens(path="austen_corpus.txt"):
    t = open(path, encoding="utf-8", errors="ignore").read()
    a = t.find("*** START"); t = t[t.find("\n", a)+1:] if a != -1 else t
    return re.findall(r"[a-z]+|[^\w\s]", t.lower())

if __name__ == "__main__":
    toks = load_tokens()
    train, test = toks[:int(len(toks)*0.9)], toks[int(len(toks)*0.9):]
    
    lex_ctx, pat_ctx = defaultdict(Counter), defaultdict(Counter)
    for i in range(len(train)):
        for order in range(2, MAXN+1):
            if i-order+1 < 0: continue
            c = tuple(train[i-order+1:i])
            lex_ctx[c][train[i]] += 1
            pat_ctx[(order, sig(c))][train[i]] += 1
    
    MIN = 10
    lex = {c: n.most_common(1)[0][0] for c, n in lex_ctx.items() if sum(n.values()) >= MIN}
    pat = {k: n.most_common(1)[0][0] for k, n in pat_ctx.items() if sum(n.values()) >= MIN}
    default = Counter(train).most_common(1)[0][0]
    
    # compress: keep a lexical rule only if it disagrees with its POS pattern
    exceptions = {c: t for c, t in lex.items() if pat.get((len(c)+1, sig(c))) != t}
    redundant = len(lex) - len(exceptions)
    
    def predict(context, rules_lex, rules_pat):
        for order in range(MAXN, 1, -1):                 # specific lexical rules first
            key = tuple(context[-(order-1):])
            if key in rules_lex: return rules_lex[key]
        for order in range(MAXN, 1, -1):                 # then POS patterns as backoff
            p = rules_pat.get((order, sig(tuple(context[-(order-1):]))))
            if p is not None: return p
        return default
    
    _pairs = [(test[max(0,i-3):i], test[i]) for i in range(3, len(test))]
    def acc(rules_lex, rules_pat):
        return sum(predict(c, rules_lex, rules_pat) == t for c, t in _pairs)/len(_pairs)
    
    o2pat = {k: v for k, v in pat.items() if k[0] == 2}
    print(f"flat pruned      : {len(lex):>6,} lexical rules            top1={acc(lex, {}):.3f}")
    print(f"grammar+exceptions: {len(exceptions):>6,} exc + {len(pat):,} patterns   top1={acc(exceptions, pat):.3f}"
          f"   ({redundant:,} rules = {redundant/len(lex)*100:.0f}% were just grammar)")
    print(f"patterns ONLY     : {len(pat):>6,} POS patterns             top1={acc({}, pat):.3f}")
    print(f"order-2 grammar   : {len(o2pat):>6,} patterns (see below)     top1={acc({}, o2pat):.3f}")
    
    print("\n=== the POS-pattern layer (order-2, the compact grammar core) ===")
    o2 = sorted(((k[1][0], t, sum(pat_ctx[k].values())) for k, t in pat.items() if k[0]==2),
                key=lambda r:-r[2])
    for p, t, n in o2:
        print(f"  {p:<6} -> {t!r:<9} ({n:,}x)")
