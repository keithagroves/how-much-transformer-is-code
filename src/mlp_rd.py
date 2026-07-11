"""MLP rate-distortion: loss vs parameter count for the token table, so 'lookup
table' becomes a quantified statement about how program-like the layer is.
Sweep SVD rank r and top-k truncation, report MLP-only held-out damage AND the
parameter count of each surrogate, and mark the knee. No heals (forward only).
"""
import collections, csv, torch
import replace_all as RA
model, DEV = RA.model, RA.DEV; tokz = RA.tokz

mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]
D = model.config.hidden_size
raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw)-10000,1000000),40000) if not 185000<=o<=215000][:24]
train_chunks = [tokz.encode(raw[o:o+8000])[:600] for o in train_starts]
eval_starts = [o for o in range(40000, min(len(raw)-10000,1000000),80000) if o not in set(train_starts) and not 185000<=o<=215000][:8]
eval_chunks = [tokz.encode(raw[o:o+8000])[:300] for o in eval_starts]

SUM={l:{} for l in MLPS}; tot={l:None for l in MLPS}; cap={}; freq=collections.Counter()
hk=[model.model.layers[l].mlp.register_forward_hook((lambda m,i,o,l=l: cap.__setitem__(l,o[0].detach().float().cpu()))) for l in MLPS]
cnt=0
with torch.no_grad():
    for sq in train_chunks:
        model(torch.tensor([sq]).to(DEV)); freq.update(sq)
        for l in MLPS:
            o=cap[l]; tot[l]=o.sum(0) if tot[l] is None else tot[l]+o.sum(0)
            for i,t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1]+=1
                else: SUM[l][t]=[o[i].clone(),1]
        cnt+=len(sq)
for h in hk: h.remove()
MEAN={l:tot[l]/cnt for l in MLPS}; LUT={l:{t:v/n for t,(v,n) in SUM[l].items()} for l in MLPS}
ntok=len(set().union(*[set(LUT[l]) for l in MLPS]))

HOLD={"lut":None}; SEQ={"s":None}; hooks=[]
def lut_mat(seq,l):
    tab=HOLD["lut"][l]; return torch.stack([tab.get(t,MEAN[l]) for t in seq]).to(DEV)
for l in MLPS:
    def mh(m,i,o,l=l):
        if HOLD["lut"] is None: return None
        return lut_mat(SEQ["s"],l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mh))
def loss(seq):
    SEQ["s"]=seq
    with torch.no_grad(): lp=torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(),-1)
    return -lp.gather(-1,torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()
HOLD["lut"]=None
intact=sum(loss(sq) for sq in eval_chunks)/len(eval_chunks)
print(f"intact {intact:.4f}; distinct token entries ~{ntok} across {len(MLPS)} layers")

def topk(k):
    keep=set(t for t,_ in freq.most_common(k))
    tab={l:{t:v for t,v in LUT[l].items() if t in keep} for l in MLPS}
    params=sum(len(tab[l]) for l in MLPS)*D                 # kept vectors x d
    return tab, params
# A full SVD is the expensive part.  Factor every layer once, then reuse the
# factors for the whole rate-distortion sweep.
SVD = {}
for l in MLPS:
    toks = list(LUT[l]); M = torch.stack([LUT[l][t] for t in toks]); mu = M.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(M - mu, full_matrices=False)
    SVD[l] = (toks, mu, U, S, Vh)

def rankr(r):
    tab={}; params=0
    for l in MLPS:
        toks,mu,U,S,Vh = SVD[l]
        Mr=(U[:,:r]*S[:r])@Vh[:r]+mu
        tab[l]={t:Mr[i] for i,t in enumerate(toks)}
        params += len(toks)*r + r*D + D                     # U_r + V_r + mean
    return tab, params
def hybrid(r, k):
    """Rank-r global table plus exact residuals for the k frequent tokens."""
    keep=set(t for t,_ in freq.most_common(k)); tab={}; params=0
    for l in MLPS:
        toks,mu,U,S,Vh = SVD[l]
        Mr=(U[:,:r]*S[:r])@Vh[:r]+mu
        approx={t:Mr[i] for i,t in enumerate(toks)}
        # Storing the correction is equivalent at runtime to storing the exact
        # row, but the parameter accounting includes the low-rank base once.
        tab[l]={t:(LUT[l][t] if t in keep else approx[t]) for t in toks}
        params += len(toks)*r + r*D + D + sum(t in keep for t in toks)*D
    return tab, params
def dmg(tab):
    HOLD["lut"]=tab; d=sum(loss(sq) for sq in eval_chunks)/len(eval_chunks)-intact; HOLD["lut"]=None; return d

full_tab_params=sum(len(LUT[l]) for l in MLPS)*D
rows=[]
print(f"\n{'surrogate':>14}{'params':>12}{'damage':>10}")
full_damage=dmg(LUT); rows.append(("full_table", full_tab_params, full_damage))
print(f"{'full table':>14}{full_tab_params:>12}{full_damage:>+10.3f}")
print("-- top-k frequent --")
for k in [50,100,200,500,1000,2000]:
    tab,p=topk(k); d=dmg(tab); rows.append((f"top_{k}",p,d)); print(f"{'top-'+str(k):>14}{p:>12}{d:>+10.3f}",flush=True)
print("-- low-rank SVD --")
for r in [2,4,8,16,32,64,128]:
    tab,p=rankr(r); d=dmg(tab); rows.append((f"rank_{r}",p,d)); print(f"{'rank-'+str(r):>14}{p:>12}{d:>+10.3f}",flush=True)
print("-- low-rank + exact frequent-token residuals --")
for r,k in [(16,100),(16,500),(32,100),(32,500),(32,1000),(64,100),(64,500)]:
    tab,p=hybrid(r,k); d=dmg(tab); rows.append((f"rank_{r}_top_{k}",p,d)); print(f"{('r'+str(r)+'+t'+str(k)):>14}{p:>12}{d:>+10.3f}",flush=True)
for h in hooks: h.remove()
with open("mlp_rd_results.csv", "w", newline="") as f:
    w=csv.writer(f); w.writerow(["surrogate","parameters","damage_nats"]); w.writerows(rows)
print("wrote mlp_rd_results.csv")
print("\nread: the knee (where damage stops dropping as params grow) is the layer's effective")
print("      description length; a low knee => genuinely compressible, closer to 'code'.")
