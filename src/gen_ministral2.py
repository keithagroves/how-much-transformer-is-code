"""Data factory v2: diversified prompts to kill near-duplicate contamination.
72 topics x 7 styles x randomized names/places -> effectively unique prompts.
Appends to ministral_corpus.txt (same file; dedup pass runs at rebuild time).

  usage: python3 gen_ministral2.py [target_words]
"""
import os, random, re, sys, requests

OUT = "ministral_corpus.txt"
TOPICS = ("a family dinner;a walk in the city;an argument between friends;a job interview;"
 "a storm at sea;a child learning to read;moving to a new house;a broken machine;a wedding;"
 "an unexpected letter;a long train journey;cooking a difficult meal;an old photograph;"
 "a lost dog;the first day of school;a garden in spring;a hospital waiting room;"
 "a small shop closing down;two neighbors quarreling;a birthday surprise;"
 "a mountain hike gone wrong;an inheritance dispute;learning to swim;a late-night phone call;"
 "a chess tournament;repairing an old boat;a power outage;the night shift at a diner;"
 "a missed flight;an apology long overdue;a street musician;the last day of harvest;"
 "a locked room;an anonymous gift;a science fair;teaching someone to drive;"
 "a flooded basement;the new manager;a forgotten anniversary;a roadside fruit stand;"
 "an overheard conversation;the town library;a borrowed umbrella;moving day for an elderly parent;"
 "a fishing trip;the school play;a broken promise;a found wallet;the first snowfall;"
 "an old rivalry;a community garden;the wrong bus;a house with a history;quitting a job;"
 "a stray cat;the farmers market;an unlikely friendship;a night at the observatory;"
 "the family business;a misdelivered package;learning an instrument;a bridge under repair;"
 "an election in a small town;the lighthouse keeper;a recipe passed down;a carnival at dusk;"
 "the retirement party;a debt repaid;an empty theater;the last customer;a shared taxi;"
 "the beekeeper's daughter").split(";")
STYLES = ["Write a short story about", "Describe in plain prose",
          "Write a scene with dialogue about", "Tell a simple story about",
          "Write a first-person account of", "Write a letter describing",
          "Narrate an afternoon involving"]
NAMES = "Elena Marcus Priya Tom Agnes Diego Ruth Sam Wei Clara Omar June Felix Nadia Earl".split()
PLACES = ("a coastal village;a mountain town;the city outskirts;a river valley;"
          "a desert crossroads;an island ferry port;a northern suburb;farm country").split(";")

def generate(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "ministral-3:3b", "stream": False,
        "options": {"temperature": 0.85, "num_predict": 700},
        "prompt": prompt}, timeout=600)
    return r.json().get("response", "")

def wordcount(path):
    if not os.path.exists(path): return 0
    return len(re.findall(r"[a-z]+", open(path, encoding="utf-8").read().lower()))

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1_200_000
    have = wordcount(OUT)
    done = open(OUT, encoding="utf-8").read().count("\n<|doc|>\n") if os.path.exists(OUT) else 0
    rng = random.Random(done * 7919)          # unique stream per doc index
    print(f"corpus at {have:,} words, target {target:,}", file=sys.stderr)
    while have < target:
        prompt = (f"{rng.choice(STYLES)} {rng.choice(TOPICS)}, set in {rng.choice(PLACES)}, "
                  f"featuring a character named {rng.choice(NAMES)}. "
                  f"Keep it natural and concrete. Do not use headers or titles.")
        text = re.sub(r"[*#_>`]+", " ", generate(prompt))
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(text.strip() + "\n<|doc|>\n")
        done += 1; rng = random.Random(done * 7919)
        have = wordcount(OUT)
        print(f"  {have:,}/{target:,}", file=sys.stderr)
    print("target reached", file=sys.stderr)
