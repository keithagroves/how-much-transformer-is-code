"""Structured-input corpus: ministral fills rigid schemas instead of free
narration. Tests the thesis that constraining input structure moves text
generation into the regime where rule systems win.

Three schemas (weather report / product listing / match recap), rigid sentence
frames, slot variation only. Appends to structured_corpus.txt.

  usage: python3 gen_structured.py [target_words]
"""
import os, random, re, sys, requests

OUT = "structured_corpus.txt"

SCHEMAS = {
"weather": ("Write a weather report following EXACTLY this structure, one sentence per line:\n"
    "1. '<City>' will be <condition> on <day>.\n"
    "2. Temperatures will reach <number> degrees by <time of day>.\n"
    "3. Winds will blow from the <direction> at <number> miles per hour.\n"
    "4. Residents should <one short piece of advice>.\n"
    "Invent the details. Plain text only, no headers, exactly 4 sentences."),
"product": ("Write a product listing following EXACTLY this structure, one sentence per line:\n"
    "1. The <product name> is a <adjective> <product type> for <audience>.\n"
    "2. It features <feature one>, <feature two>, and <feature three>.\n"
    "3. It weighs <number> <unit> and comes in <color> and <color>.\n"
    "4. It costs <number> dollars and ships within <number> days.\n"
    "Invent the details. Plain text only, no headers, exactly 4 sentences."),
"recap": ("Write a sports match recap following EXACTLY this structure, one sentence per line:\n"
    "1. The <team name> defeated the <team name> by a score of <number> to <number>.\n"
    "2. <Player name> scored <number> points in the <ordinal> half.\n"
    "3. The turning point came when <short event clause>.\n"
    "4. The next match is on <day> against the <team name>.\n"
    "Invent the details. Plain text only, no headers, exactly 4 sentences."),
}

def gen(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0.8, "num_predict": 220}, "prompt": prompt}, timeout=300)
    return r.json().get("response", "")

def wc(path):
    if not os.path.exists(path): return 0
    return len(re.findall(r"[a-z]+", open(path, encoding="utf-8").read().lower()))

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 80_000
    names = list(SCHEMAS)
    done = open(OUT, encoding="utf-8").read().count("\n<|doc|>\n") if os.path.exists(OUT) else 0
    have = wc(OUT)
    print(f"at {have:,} words, target {target:,}", file=sys.stderr)
    while have < target:
        schema = names[done % len(names)]
        text = re.sub(r"[*#_>`]+", " ", gen(SCHEMAS[schema])).strip()
        if len(text.split()) > 15:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(text + "\n<|doc|>\n")
        done += 1
        have = wc(OUT)
        if done % 20 == 0: print(f"  {have:,}/{target:,}", file=sys.stderr)
    print("target reached", file=sys.stderr)
