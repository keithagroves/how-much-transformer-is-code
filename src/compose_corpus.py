"""Compositional testbed: negation. A 2x2 design over
   lexical polarity {pos word, neg word}  x  {plain, negated}.

'great' and 'not great' are lexically near-identical but oppositely labeled, so
the target function is an INTERACTION (lex XOR neg), which no single linear
direction can express. Metadata (lex, neg) is controlled by construction.
"""
import json, itertools, random
random.seed(7)

ASPECTS = ["food", "service", "hotel room", "movie", "phone", "staff",
           "view", "coffee", "seat", "delivery", "hotel", "meal"]
POS = ["excellent", "wonderful", "fantastic", "great", "superb", "amazing"]
NEG = ["terrible", "awful", "horrible", "disappointing", "dreadful", "lousy"]

def plain(a, w):    return f"The {a} was {w}."
def negate(a, w):   return f"The {a} was not {w}."

rows = []
for lex, words in [(+1, POS), (-1, NEG)]:
    pairs = list(itertools.product(ASPECTS, words))
    random.shuffle(pairs)
    for a, w in pairs[:24]:                 # 24 per lexical polarity, per form
        rows.append({"text": plain(a, w),  "lex": lex, "neg": 0})
    for a, w in pairs[24:48]:
        rows.append({"text": negate(a, w), "lex": lex, "neg": 1})

random.shuffle(rows)
json.dump(rows, open("compose.json", "w"), indent=2)
from collections import Counter
print(f"{len(rows)} sentences; cells (lex,neg):",
      dict(Counter((r['lex'], r['neg']) for r in rows)))
for r in rows[:6]:
    print("  ", r["lex"], r["neg"], r["text"])
