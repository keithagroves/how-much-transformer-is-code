"""Warm-start twin: init BOTH arms from Pythia-160m (induction already formed),
continue-train one on natural text and one on lawful schemas, then measure how
rule-nameable each becomes. Sidesteps the from-scratch phase change.

Clean design to separate the WEIGHTS effect from the eval-text effect: measure
a 3x2 grid — {baseline, natural-tuned, structured-tuned} x {natural eval,
structured eval}. If tuning on structure raises nameability on matched eval
ABOVE the untuned baseline on that same eval text, the weights changed, not
just the difficulty of the text.

  python3 pythia_twin.py prep
  python3 pythia_twin.py train natural|structured
  python3 pythia_twin.py measure
"""
import gc, json, math, random, sys, time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tokz = AutoTokenizer.from_pretrained(MODEL)

# ---------------- structured generator (self-contained) ----------------
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
EVENT = ["the goalkeeper saved a penalty", "a late timeout changed the momentum",
         "an interception led to a quick score", "the crowd rallied behind the home side"]

def a_weather():
    return (f"{random.choice(CITY)} will be {random.choice(COND)} on {random.choice(DAY)}.\n"
            f"Temperatures will reach {random.randint(20,105)} degrees by {random.choice(TOD)}.\n"
            f"Winds will blow from the {random.choice(DIR)} at {random.randint(3,45)} miles per hour.\n"
            f"Residents should {random.choice(ADVICE)}.\n")
def a_product():
    f3 = random.sample(FEAT, 3); c2 = random.sample(COLOR, 2)
    return (f"The {random.choice(ADJ).title()} {random.choice(PROD).title()} is a {random.choice(ADJ)} {random.choice(PROD)} for {random.choice(AUD)}.\n"
            f"It features {f3[0]}, {f3[1]}, and {f3[2]}.\n"
            f"It weighs {random.randint(1,40)} {random.choice(UNIT)} and comes in {c2[0]} and {c2[1]}.\n"
            f"It costs {random.randint(10,400)} dollars and ships within {random.randint(1,14)} days.\n")
def a_recap():
    t1, t2, t3 = random.sample(TEAM, 3); s1 = random.randint(60,120); s2 = s1 - random.randint(2,30)
    return (f"The {t1} defeated the {t2} by a score of {s1} to {s2}.\n"
            f"{random.choice(NAME)} scored {random.randint(8,45)} points in the first half.\n"
            f"The turning point came when {random.choice(EVENT)}.\n"
            f"The next match is on {random.choice(DAY)} against the {t3}.\n")

def prep():
    from datasets import load_dataset
    BUDGET = 6_000_000
    random.seed(0)
    print("generating structured...")
    structured = "\n".join(random.choice([a_weather, a_product, a_recap])() for _ in range(200_000))
    print("streaming TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    parts, tot = [], 0
    for r in ds:
        parts.append(r["text"]); tot += len(r["text"])
        if tot > 40_000_000: break
    natural = "\n\n".join(parts)

    def enc(text):
        ids, pos = [], 0
        while len(ids) < BUDGET and pos < len(text):
            ids += tokz.encode(text[pos:pos+2_000_000]); pos += 2_000_000
        return ids[:BUDGET]
    json.dump(enc(natural), open("pyt_natural.json", "w"))
    json.dump(enc(structured), open("pyt_structured.json", "w"))
    print(f"saved pyt_natural.json / pyt_structured.json ({BUDGET:,} tokens each)")

# ---------------- warm-start fine-tune ----------------
def train(arm):
    STEPS, BATCH, CTX, LR = 1500, 4, 512, 1e-5
    ids = torch.tensor(json.load(open(f"pyt_{arm}.json")), dtype=torch.long)
    tr = ids[:-100_000]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    sched = lambda t: min(t/100, 1.0) * 0.5 * (1 + math.cos(math.pi * max(0, t-100)/(STEPS-100)))
    print(f"[{arm}] warm-start fine-tune, {STEPS} steps @ lr {LR}")
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups: g["lr"] = LR * sched(step)
        i = torch.randint(0, len(tr)-CTX-1, (BATCH,))
        x = torch.stack([tr[j:j+CTX] for j in i]).to(DEV)
        out = model(x, labels=x)
        opt.zero_grad(); out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0 or step == STEPS-1:
            print(f"  step {step:>4}  loss {out.loss.item():.3f}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save(model.state_dict(), f"pyt_tuned_{arm}.pt")
    print(f"saved pyt_tuned_{arm}.pt")

# ---------------- ceiling measurement (NeoX) ----------------
T_C, NCHUNK, L = 512, 8, 50
def match_cols(seq):
    occ = {}; cols = [set() for _ in range(len(seq))]
    for i in range(len(seq)):
        for p in occ.get(seq[i], []):
            if p+1 < len(seq): cols[i].add(p+1)
        occ.setdefault(seq[i], []).append(i)
    return cols

def measure():
    NL = AutoModelForCausalLM.from_pretrained(MODEL).config.num_hidden_layers
    cfg = AutoModelForCausalLM.from_pretrained(MODEL).config
    NH, DH = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    V = cfg.vocab_size

    val_nat = torch.tensor(json.load(open("pyt_natural.json"))[-100_000:])
    val_str = torch.tensor(json.load(open("pyt_structured.json"))[-100_000:])
    EVAL = {"nat-eval": val_nat, "str-eval": val_str}

    def load(which):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV).eval()
        if which != "baseline":
            m.load_state_dict(torch.load(f"pyt_tuned_{which}.pt", map_location=DEV))
        return m

    def ceiling_on(model, val, byl):
        rows = []
        vc = {}
        for c in range(NCHUNK):
            seq = val[c*T_C:(c+1)*T_C].tolist()
            if len(seq) < T_C: break
            with torch.no_grad():
                out = model(torch.tensor([seq]).to(DEV), output_attentions=True)
            ATT = {l: out.attentions[l][0].float() for l in byl}; del out
            n = len(seq); cols = match_cols(seq)
            mask = torch.zeros(n, n)
            for a, cs in enumerate(cols):
                for j in cs: mask[a, j] = 1
            mask[:, 0] = 1; maskD = mask.to(DEV)
            def vhooks():
                hk = []
                for l in byl:
                    def h(mod, inp, outp, l=l):
                        q3 = outp.view(outp.shape[0], outp.shape[1], NH, 3*DH)
                        vc[l] = q3[0,:,:,2*DH:].detach()
                    hk.append(model.gpt_neox.layers[l].attention.query_key_value.register_forward_hook(h))
                return hk
            def run(mode):
                hooks = vhooks() if mode in ("masked","inverse") else []
                for l, hs in byl.items():
                    def oh(mod, inp, l=l, hs=hs):
                        x = inp[0].clone()
                        for h in hs:
                            if mode == "zero": x[0,:,h*DH:(h+1)*DH] = 0
                            else:
                                m = maskD if mode=="masked" else torch.tril(1-mask.cpu()).to(DEV)
                                x[0,:,h*DH:(h+1)*DH] = (ATT[l][h]*m) @ vc[l][:, h, :]
                        return (x,)+inp[1:]
                    hooks.append(model.gpt_neox.layers[l].attention.dense.register_forward_pre_hook(oh))
                with torch.no_grad():
                    lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
                for hk in hooks: hk.remove()
                return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
            ni = run("intact") if False else None
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(), -1)
            ni = -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
            nz, nm = run("zero"), run("masked")
            rows.append((ni, nz, nm)); del ATT; gc.collect(); torch.mps.empty_cache()
        t = torch.tensor(rows); gap = t[:,1]-t[:,0]
        return (t[:,1]-t[:,2]).sum()/gap.sum(), gap.mean()

    def induction_of(model):
        torch.manual_seed(0)
        sq = torch.randint(1000, V-1000, (8, L)); ids = torch.cat([sq, sq], 1).to(DEV)
        with torch.no_grad(): out = model(ids, output_attentions=True)
        qpos = torch.arange(L, 2*L-1); ind = torch.zeros(NL, NH)
        for l, att in enumerate(out.attentions):
            ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
        del out; gc.collect(); torch.mps.empty_cache()
        heads = [(l,h) for l in range(NL) for h in range(NH) if ind[l,h] > 0.2]
        byl = {}
        for l,h in heads: byl.setdefault(l, []).append(h)
        return byl, len(heads)

    print(f"\n{'model':<16}{'induction':>10}   " + "".join(f"{e:>22}" for e in EVAL))
    grid = {}
    for which in ("baseline", "natural", "structured"):
        model = load(which)
        byl, nheads = induction_of(model)
        cells = []
        for ename, val in EVAL.items():
            pm, gap = ceiling_on(model, val, byl)
            grid[(which, ename)] = float(pm)
            cells.append(f"{pm:>7.0%} (gap {gap:.2f})")
        print(f"{which:<16}{nheads:>10}   " + "".join(f"{c:>22}" for c in cells), flush=True)
        del model; gc.collect(); torch.mps.empty_cache()

    print("\n=== WEIGHTS EFFECT (nameable fraction, tuned minus baseline, matched eval) ===")
    print(f"  structured-tuned on str-eval:  {grid[('structured','str-eval')]:.0%}  "
          f"vs baseline {grid[('baseline','str-eval')]:.0%}  "
          f"-> {grid[('structured','str-eval')]-grid[('baseline','str-eval')]:+.0%}")
    print(f"  natural-tuned    on nat-eval:  {grid[('natural','nat-eval')]:.0%}  "
          f"vs baseline {grid[('baseline','nat-eval')]:.0%}  "
          f"-> {grid[('natural','nat-eval')]-grid[('baseline','nat-eval')]:+.0%}")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep": prep()
    elif cmd == "train": train(sys.argv[2])
    elif cmd == "measure": measure()
