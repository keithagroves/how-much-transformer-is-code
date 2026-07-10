"""Data factory: ministral generates the training corpus (resumable).

Diverse seed prompts -> raw continuations at temp 0.8, appended to
ministral_corpus.txt until the word target is reached. Re-run to grow further.

  usage: python3 gen_ministral.py [target_words]
"""
import itertools, json, os, re, sys, requests

OUT = "ministral_corpus.txt"
TOPICS = ["a family dinner", "a walk in the city", "an argument between friends",
    "a job interview", "a storm at sea", "a child learning to read", "moving to a new house",
    "a broken machine", "a wedding", "an unexpected letter", "a long train journey",
    "cooking a difficult meal", "an old photograph", "a lost dog", "the first day of school",
    "a garden in spring", "a hospital waiting room", "a small shop closing down",
    "two neighbors quarreling", "a birthday surprise", "a mountain hike gone wrong",
    "an inheritance dispute", "learning to swim", "a late-night phone call"]
STYLES = ["Write a short story about", "Describe in plain prose",
          "Write a scene with dialogue about", "Tell a simple story about"]

def generate(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0.8, "num_predict": 700},
        "prompt": prompt}, timeout=600)
    return r.json().get("response", "")

def wordcount(path):
    if not os.path.exists(path): return 0
    return len(re.findall(r"[a-z]+", open(path, encoding="utf-8").read().lower()))

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    have = wordcount(OUT)
    print(f"corpus at {have:,} words, target {target:,}", file=sys.stderr)
    combos = itertools.cycle((s, t) for t in TOPICS for s in STYLES)
    # skip already-used combos deterministically by count of separators
    done = open(OUT, encoding="utf-8").read().count("\n<|doc|>\n") if os.path.exists(OUT) else 0
    for _ in range(done): next(combos)
    while have < target:
        style, topic = next(combos)
        text = generate(f"{style} {topic}. Keep it natural and concrete.")
        text = re.sub(r"[*#_>`]+", " ", text)          # strip markdown noise
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(text.strip() + "\n<|doc|>\n")
        have = wordcount(OUT)
        print(f"  {have:,}/{target:,} words", file=sys.stderr)
    print("target reached", file=sys.stderr)
