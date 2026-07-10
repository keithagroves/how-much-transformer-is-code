"""Twin experiment, step 1: build matched corpora + one shared tokenizer.

Natural arm:    TinyStories (real dataset, standard for tiny-model training)
Structured arm: programmatic generator — the three Act-II schemas with random
                slot fills; a perfectly lawful world, no LLM involved

Both trimmed to the same token budget under a shared 8k BPE tokenizer so the
ONLY difference between the twins is the regularity of their training data.
"""
import json, random, sys
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BUDGET = 5_000_000            # tokens per arm
random.seed(0)

# ---------------- structured generator ----------------
CITY = "Riverton Ashford Maplewood Kingsport Dunmore Fairhaven Brookfield Eastvale Norwood Lakemont Hartwell Stonebridge Millbrook Crestwood Baytown Elmsford".split()
COND = "sunny cloudy rainy windy foggy snowy stormy clear overcast humid".split()
DAY = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
TOD = ["noon", "midday", "early afternoon", "late afternoon", "evening", "sunset"]
DIR = "north south east west northeast northwest southeast southwest".split()
ADVICE = ["carry an umbrella", "wear sunscreen", "dress warmly", "stay indoors",
          "drive carefully", "drink plenty of water", "secure loose objects", "plan for delays"]
PROD = "lamp kettle backpack blender keyboard chair blanket speaker bottle toaster desk monitor".split()
ADJ = "sturdy lightweight elegant compact durable modern affordable premium versatile reliable".split()
AUD = ["students", "travelers", "families", "professionals", "campers", "gamers", "chefs", "readers"]
FEAT = ["a long battery life", "a waterproof shell", "adjustable settings", "a quiet motor",
        "fast charging", "a soft grip", "an energy saving mode", "a compact design",
        "wireless connectivity", "easy cleaning"]
COLOR = "black white silver blue red green gray navy beige charcoal".split()
UNIT = ["pounds", "kilograms", "ounces"]
TEAM = "Falcons Tigers Rockets Wolves Sharks Eagles Bears Panthers Hornets Comets Ravens Bison".split()
NAME = "Jordan Casey Morgan Riley Avery Quinn Hayden Parker Reese Dakota Emerson Rowan".split()
ORD = ["first", "second"]
EVENT = ["the goalkeeper saved a penalty", "a late timeout changed the momentum",
         "an interception led to a quick score", "the crowd rallied behind the home side",
         "a substitution sparked the offense", "back to back threes shifted the lead"]

def weather():
    return (f"{random.choice(CITY)} will be {random.choice(COND)} on {random.choice(DAY)}.\n"
            f"Temperatures will reach {random.randint(20, 105)} degrees by {random.choice(TOD)}.\n"
            f"Winds will blow from the {random.choice(DIR)} at {random.randint(3, 45)} miles per hour.\n"
            f"Residents should {random.choice(ADVICE)}.\n")

def product():
    f3 = random.sample(FEAT, 3)
    c2 = random.sample(COLOR, 2)
    return (f"The {random.choice(ADJ).title()} {random.choice(PROD).title()} is a {random.choice(ADJ)} {random.choice(PROD)} for {random.choice(AUD)}.\n"
            f"It features {f3[0]}, {f3[1]}, and {f3[2]}.\n"
            f"It weighs {random.randint(1, 40)} {random.choice(UNIT)} and comes in {c2[0]} and {c2[1]}.\n"
            f"It costs {random.randint(10, 400)} dollars and ships within {random.randint(1, 14)} days.\n")

def recap():
    t1, t2, t3 = random.sample(TEAM, 3)
    s1 = random.randint(60, 120); s2 = s1 - random.randint(2, 30)
    return (f"The {t1} defeated the {t2} by a score of {s1} to {s2}.\n"
            f"{random.choice(NAME)} scored {random.randint(8, 45)} points in the {random.choice(ORD)} half.\n"
            f"The turning point came when {random.choice(EVENT)}.\n"
            f"The next match is on {random.choice(DAY)} against the {t3}.\n")

GENS = [weather, product, recap]
print("generating structured corpus...")
docs = [random.choice(GENS)() for _ in range(160_000)]
structured = "\n".join(docs)
print(f"  structured: {len(structured):,} chars")

# ---------------- TinyStories ----------------
print("downloading TinyStories...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
parts, tot = [], 0
for r in ds:
    parts.append(r["text"]); tot += len(r["text"])
    if tot > 30_000_000: break
natural = "\n\n".join(parts)
print(f"  tinystories: {len(natural):,} chars")

# ---------------- shared tokenizer ----------------
print("training shared 8k BPE...")
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<unk>", "<pad>"])
mix = [natural[i:i+50000] for i in range(0, 8_000_000, 50000)] + \
      [structured[i:i+50000] for i in range(0, min(len(structured), 8_000_000), 50000)]
tok.train_from_iterator(mix, trainer)
tok.save("twin_tokenizer.json")

# ---------------- encode + trim to budget ----------------
def encode_to_budget(text, budget):
    ids, pos, step = [], 0, 2_000_000
    while len(ids) < budget and pos < len(text):
        ids += tok.encode(text[pos:pos+step]).ids
        pos += step
    return ids[:budget]

print("encoding to matched budgets...")
nat_ids = encode_to_budget(natural, BUDGET)
st_ids = encode_to_budget(structured, BUDGET)
print(f"  natural {len(nat_ids):,} tokens | structured {len(st_ids):,} tokens")
json.dump(nat_ids, open("twin_natural.json", "w"))
json.dump(st_ids, open("twin_structured.json", "w"))
print("saved twin_natural.json / twin_structured.json / twin_tokenizer.json")
