"""Reviewer control: the healing asymmetry. Healed-code damage is measured against
the INTACT model, but the healed hybrid fine-tuned 65k norm gains on the training
distribution and the intact model did not. If healing an intact model (no code)
with the identical protocol itself gains on held-out text, the healed-code damage
is understated by that offset.

Here: heal ONLY the 65k RMSNorm gains of the otherwise-intact model on the same
24+2 training chunks, same epochs, then report held-out loss delta vs the unhealed
intact model on the same 8 held-out offsets used by heal_shuffle/heal_holdout.
If ~0, the +0.64 is clean; if not, it is the offset to subtract.
"""
import torch
import replace_all as RA

model, DEV = RA.model, RA.DEV
tokz = RA.tokz
EPOCHS, T_TR, LR = 20, 600, 3e-4

import os as _os
raw = open(_os.environ.get("SUB_CORPUS", "ministral_corpus.txt")).read()
train_starts = [o for o in range(20000, min(len(raw) - 10000, 1000000), 40000)
                if not 185000 <= o <= 215000][:24]
train_chunks = [tokz.encode(raw[o:o + 8000])[:T_TR] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r + r)(torch.randint(1000, RA.V - 1000, (50,)).tolist()) * 3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw) - 10000, 1000000), 80000)
               if o not in set(train_starts) and not 185000 <= o <= 215000][:8]
eval_chunks = [tokz.encode(raw[o:o + 8000])[:300] for o in eval_starts]
print(f"held-out eval offsets (k): {[o//1000 for o in eval_starts]}")

def loss(seq):
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0, :-1].float(), -1)
    return -lp.gather(-1, torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm_params = [p for n_, p in model.named_parameters() if "norm" in n_.lower()]
orig = [p.detach().clone() for p in norm_params]

# unhealed intact held-out loss
base = [loss(sq) for sq in eval_chunks]
print(f"intact (unhealed) held-out loss: {sum(base)/len(base):.4f}")

# heal ONLY norm gains on training chunks, no code installed
for p in norm_params: p.requires_grad_(True)
opt = torch.optim.Adam(norm_params, lr=LR)
model.train()
for ep in range(EPOCHS):
    for sq in train_chunks:
        ids = torch.tensor([sq]).to(DEV)
        out = model(ids, labels=ids)
        opt.zero_grad(); out.loss.backward(); opt.step()
model.eval()
healed = [loss(sq) for sq in eval_chunks]
for p, o in zip(norm_params, orig): p.data.copy_(o)   # restore

deltas = [h - b for h, b in zip(healed, base)]
import random
random.seed(0); B = 5000; n = len(deltas)
bs = sorted(sum(deltas[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
mean = sum(deltas) / n
print(f"intact-heal held-out delta: {mean:+.4f} nats  95% CI [{bs[int(.025*B)]:+.4f}, {bs[int(.975*B)]:+.4f}]")
print("negative = healing the norms alone improves held-out loss (a domain-adaptation offset the")
print("healed-code number gets for free); ~0 = the +0.64 healed-code damage is clean.")
