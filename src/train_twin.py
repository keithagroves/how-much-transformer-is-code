"""Twin experiment, step 2: train one tiny transformer on a chosen arm.

Same architecture, same hyperparameters, same token budget — only the data
differs. A small GPT (Llama-style: RMSNorm, RoPE, SwiGLU, GQA off) sized to
train in a few hours on MPS.

  usage: python3 train_twin.py natural|structured [steps]
"""
import json, math, sys, time, torch
import torch.nn as nn
import torch.nn.functional as F

ARM = sys.argv[1] if len(sys.argv) > 1 else "natural"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- config (kept in a dict so eval can rebuild the model) ----
CFG = dict(vocab=8192, dim=384, n_layer=8, n_head=8, ctx=256, mlp=1024)
BATCH, LR, WARMUP = 32, 3e-4, 200

ids = torch.tensor(json.load(open(f"twin_{ARM}.json")), dtype=torch.long)
n_val = 100_000
train_ids, val_ids = ids[:-n_val], ids[-n_val:]
print(f"[{ARM}] {len(train_ids):,} train / {len(val_ids):,} val tokens, {STEPS} steps")

def rope(x, pos):
    d = x.shape[-1]
    freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos[:, None].float() * freq[None, :]
    cos, sin = ang.cos(), ang.sin()
    cos = torch.cat([cos, cos], -1)[None, None]; sin = torch.cat([sin, sin], -1)[None, None]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h, s.dh = c["n_head"], c["dim"] // c["n_head"]
        s.ln1 = nn.RMSNorm(c["dim"]); s.ln2 = nn.RMSNorm(c["dim"])
        s.qkv = nn.Linear(c["dim"], 3*c["dim"], bias=False)
        s.proj = nn.Linear(c["dim"], c["dim"], bias=False)
        s.w1 = nn.Linear(c["dim"], c["mlp"], bias=False)
        s.w2 = nn.Linear(c["dim"], c["mlp"], bias=False)
        s.w3 = nn.Linear(c["mlp"], c["dim"], bias=False)
    def forward(s, x, pos):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, -1)
        q = q.view(B, T, s.h, s.dh).transpose(1, 2)
        k = k.view(B, T, s.h, s.dh).transpose(1, 2)
        v = v.view(B, T, s.h, s.dh).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(o.transpose(1, 2).reshape(B, T, D))
        h = s.ln2(x)
        x = x + s.w3(F.silu(s.w1(h)) * s.w2(h))
        return x

class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.emb = nn.Embedding(c["vocab"], c["dim"])
        s.blocks = nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        s.lnf = nn.RMSNorm(c["dim"])
        s.head = nn.Linear(c["dim"], c["vocab"], bias=False)
        s.head.weight = s.emb.weight
    def forward(s, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = s.emb(idx)
        for b in s.blocks: x = b(x, pos)
        return s.head(s.lnf(x))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

def batch(src):
    i = torch.randint(0, len(src) - CFG["ctx"] - 1, (BATCH,))
    x = torch.stack([src[j:j+CFG["ctx"]] for j in i])
    y = torch.stack([src[j+1:j+1+CFG["ctx"]] for j in i])
    return x.to(DEV), y.to(DEV)

if __name__ == "__main__":
    torch.manual_seed(0)
    model = GPT(CFG)
    model.apply(init_weights)
    model.head.weight = model.emb.weight        # re-tie after init
    model = model.to(DEV)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model: {nparam/1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    sched = lambda t: min(t/WARMUP, 1.0) * 0.5 * (1 + math.cos(math.pi * max(0, t-WARMUP)/(STEPS-WARMUP)))
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups: g["lr"] = LR * sched(step)
        x, y = batch(train_ids)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, CFG["vocab"]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == STEPS-1:
            model.eval()
            with torch.no_grad():
                vl = torch.stack([F.cross_entropy(
                    model(bx).view(-1, CFG["vocab"]), by.view(-1))
                    for bx, by in [batch(val_ids) for _ in range(8)]]).mean().item()
            model.train()
            print(f"  step {step:>5}  train {loss.item():.3f}  val {vl:.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    torch.save({"cfg": CFG, "state": model.state_dict(), "val": vl}, f"twin_model_{ARM}.pt")
    print(f"saved twin_model_{ARM}.pt")
