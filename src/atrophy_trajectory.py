"""Diagnostic: is the induction collapse from training TIME, our DATA's
degeneracy, or a genuine 'regular worlds don't need induction' effect?

Fine-tune Pythia-160m from the same init on a chosen corpus and log the
induction head count (random-repeat probe) at checkpoints. Compare arms:
  structured  — our degenerate schema corpus (collapsed to 3 heads @1500)
  wikitext    — a DIFFERENT natural corpus (control: if it keeps its heads,
                the collapse is specific to the structured corpus, not tuning)

A gradual decay => steady data pressure (and shows where a head-preserving
tune sits). Natural arm already known stable (19/20 @1500).

  python3 atrophy_trajectory.py structured|wikitext
"""
import json, math, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "EleutherAI/pythia-160m"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
ARM = sys.argv[1]
STEPS, BATCH, CTX, LR, L = 1500, 4, 512, 1e-5, 50
CKPTS = [0, 100, 250, 500, 1000, 1500]

tokz = AutoTokenizer.from_pretrained(MODEL)

# ---- corpus ----
if ARM == "wikitext":
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    buf, tot = [], 0
    for r in ds:
        if r["text"].strip(): buf.append(r["text"]); tot += len(r["text"])
        if tot > 40_000_000: break
    ids = torch.tensor(tokz.encode("\n".join(buf))[:6_000_000], dtype=torch.long)
else:
    ids = torch.tensor(json.load(open(f"pyt_{ARM}.json")), dtype=torch.long)
tr = ids[:-100_000]
print(f"[{ARM}] {len(tr):,} train tokens")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_implementation="eager", dtype=torch.float32).to(DEV)
cfg = model.config
NL, NH, V = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size

def induction_count():
    model.eval()
    torch.manual_seed(0)
    sq = torch.randint(1000, V-1000, (8, L)); rep = torch.cat([sq, sq], 1).to(DEV)
    with torch.no_grad(): out = model(rep, output_attentions=True)
    qpos = torch.arange(L, 2*L-1); ind = torch.zeros(NL, NH)
    for l, att in enumerate(out.attentions):
        ind[l] = att.float().cpu()[:, :, qpos, qpos-L+1].mean(dim=(0,2))
    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    nll = -lp.gather(-1, rep[:, 1:].unsqueeze(-1)).squeeze(-1)
    drop = nll[:, :L-1].mean().item() - nll[:, L-1:].mean().item()
    model.train()
    return int((ind > 0.2).sum()), float(ind.max()), drop

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
sched = lambda t: min(t/100, 1.0) * 0.5 * (1 + math.cos(math.pi * max(0, t-100)/(STEPS-100)))
print(f"{'step':>6}{'heads>0.2':>11}{'max score':>11}{'repeat-drop':>13}")
t0 = time.time()
model.train()
for step in range(STEPS + 1):
    if step in CKPTS:
        nh, mx, drop = induction_count()
        print(f"{step:>6}{nh:>11}{mx:>11.2f}{drop:>13.2f}   ({(time.time()-t0)/60:.1f} min)", flush=True)
    if step == STEPS: break
    for g in opt.param_groups: g["lr"] = LR * sched(step)
    i = torch.randint(0, len(tr)-CTX-1, (BATCH,))
    x = torch.stack([tr[j:j+CTX] for j in i]).to(DEV)
    out = model(x, labels=x)
    opt.zero_grad(); out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
print(f"[{ARM}] done")
