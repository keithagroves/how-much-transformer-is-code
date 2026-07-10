"""Author-independent held-out set: glm-4.7-flash writes novel sentences.
glm does NOT assign labels (ministral does that later) -- it only supplies text
in a style neither Claude nor the original corpus produced. Dedup vs corpus.
"""
import json, re, requests
from corpus import TEXTS

PROMPT = (
    "Write 60 diverse short English sentences drawn from everyday life, spanning "
    "many domains: technology, food, travel, work, sports, health, home, weather, "
    "relationships, shopping, education, and services. Vary the sentiment: about a "
    "third clearly positive, a third clearly negative, and a third purely factual "
    "and neutral. Keep each to one line, 6-14 words. No numbering, no quotation "
    "marks, no headers. Output only the sentences, one per line."
)

def generate():
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma4:e4b", "stream": False,
        "options": {"temperature": 0.9, "num_predict": 1600},
        "prompt": PROMPT,
    }, timeout=600)
    return r.json()["response"]

def clean(raw):
    lines = []
    seen = set(t.lower().strip(" .") for t in TEXTS)
    for ln in raw.splitlines():
        ln = ln.strip()
        ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln)      # strip bullets/numbers
        ln = ln.strip(' "\'')
        if not ln or len(ln.split()) < 4:
            continue
        key = ln.lower().strip(" .")
        if key in seen:                                # no overlap w/ original
            continue
        seen.add(key)
        lines.append(ln)
    return lines

if __name__ == "__main__":
    import os
    acc, seen = [], set(t.lower().strip(" .") for t in TEXTS)
    if os.path.exists("test_texts.json"):                 # accumulate across runs
        for t in json.load(open("test_texts.json")):
            k = t.lower().strip(" .")
            if k not in seen:
                seen.add(k); acc.append(t)
    for rnd in range(5):
        if len(acc) >= 55:
            break
        for ln in clean(generate()):
            k = ln.lower().strip(" .")
            if k not in seen:
                seen.add(k); acc.append(ln)
        print(f"round {rnd+1}: {len(acc)} unique so far")
    json.dump(acc, open("test_texts.json", "w"), indent=2)
    print(f"total {len(acc)} novel sentences (deduped vs corpus + across rounds)")
