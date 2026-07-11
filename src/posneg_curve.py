"""Where does content start earning its keep? Sweep the budget and compare the
FITTED code (all 25 template columns) against POSITIONAL-ONLY (drop the two
content columns, 'dup' and 'rule' — duplicate-token and induction-follower), both
installed the same way. If the curves never separate below the induction tail,
the substitutable attention is purely positional and content buys nothing there;
where they separate is exactly where copy-content begins to matter.
Unhealed fine grid (cheap) + one healed point (k=160, capped 10-epoch) for both.
"""
import gc, csv, sys, torch
import replace_rich as RR
import replace_all as RA

model, DEV, DH, GROUP = RA.model, RA.DEV, RA.DH, RA.GROUP
tokz = RA.tokz
LR, EPOCHS = 3e-4, 10
SKIP_HEAL = "--skip-heal" in sys.argv
CONTENT = {"dup", "rule"}

RR.W.update(torch.load("rich_templates.pt"))
nat_c = torch.load("rich_solo_costs.pt"); rnd_c = torch.load("rich_solo_rnd.pt")
nr = {k: i for i, k in enumerate(sorted(nat_c, key=lambda k: nat_c[k]))}
rr = {k: i for i, k in enumerate(sorted(rnd_c, key=lambda k: rnd_c[k]))}
HEADS_ALL = sorted(nat_c, key=lambda k: nr[k] + rr[k])
mcosts = torch.load("mlp_solo_costs.pt")["costs"]
MLPS = sorted(mcosts, key=lambda l: mcosts[l])[:6]

raw = open("ministral_corpus.txt").read()
train_starts = [o for o in range(20000, min(len(raw)-10000,1000000),40000) if not 185000<=o<=215000][:24]
train_chunks = [tokz.encode(raw[o:o+8000])[:600] for o in train_starts]
torch.manual_seed(11)
mkrnd = lambda: (lambda r: r+r)(torch.randint(1000,RA.V-1000,(50,)).tolist())*3
train_chunks += [mkrnd(), mkrnd()]
eval_starts = [o for o in range(40000, min(len(raw)-10000,1000000),80000) if o not in set(train_starts) and not 185000<=o<=215000][:8]
eval_chunks = [tokz.encode(raw[o:o+8000])[:300] for o in eval_starts]

SUM={l:{} for l in MLPS}; tot={l:None for l in MLPS}; cap={}
hk=[model.model.layers[l].mlp.register_forward_hook((lambda m,i,o,l=l: cap.__setitem__(l,o[0].detach().float().cpu()))) for l in MLPS]
cnt=0
with torch.no_grad():
    for sq in train_chunks:
        model(torch.tensor([sq]).to(DEV))
        for l in MLPS:
            o=cap[l]; tot[l]=o.sum(0) if tot[l] is None else tot[l]+o.sum(0)
            for i,t in enumerate(sq):
                if t in SUM[l]: SUM[l][t][0].add_(o[i]); SUM[l][t][1]+=1
                else: SUM[l][t]=[o[i].clone(),1]
        cnt+=len(sq)
for h in hk: h.remove()
MEAN={l:tot[l]/cnt for l in MLPS}; LUT={l:{t:v/n for t,(v,n) in SUM[l].items()} for l in MLPS}

def head_A(seq, heads, drop):
    base=RR.code_attn(seq); n=len(seq); byl={}
    for l,h in heads: byl.setdefault(l,[]).append(h)
    out={}
    for l,hs in byl.items():
        mats=[]
        for h in hs:
            M=torch.zeros(n,n)
            for k,wk in RR.W[(l,h)].items():
                if wk>1e-4 and k not in drop: M+=wk*base[k]
            s=M.sum(-1,keepdim=True)
            mats.append((h, M/s.clamp(min=1e-9)))
        out[l]=mats
    return out
def lut_mat(seq,l): return torch.stack([LUT[l].get(t,MEAN[l]) for t in seq]).to(DEV)

HOLD={"A":None,"mlp":False}; vcache=[]; hooks=[]; vc={}
for l in sorted({l for l,_ in HEADS_ALL}):
    attn=model.model.layers[l].self_attn
    hooks.append(attn.v_proj.register_forward_hook(lambda m,i,o,l=l: vc.__setitem__(l,o[0])))
    def oh(m,inp,l=l):
        if HOLD["A"] is None or l not in HOLD["A"]: return None
        x=inp[0].clone()
        for h,A in HOLD["A"][l]:
            g=h//GROUP; x[0,:,h*DH:(h+1)*DH]=A.to(DEV)@vc[l][:,g*DH:(g+1)*DH]
        return (x,)+inp[1:]
    hooks.append(attn.o_proj.register_forward_pre_hook(oh))
for l in MLPS:
    def mh(m,i,o,l=l):
        if not HOLD["mlp"]: return None
        return lut_mat(SEQ["s"],l).unsqueeze(0)
    hooks.append(model.model.layers[l].mlp.register_forward_hook(mh))

SEQ={"s":None}
def loss(seq,A):
    SEQ["s"]=seq; HOLD["A"]=A; HOLD["mlp"]=True
    with torch.no_grad(): lp=torch.log_softmax(model(torch.tensor([seq]).to(DEV)).logits[0,:-1].float(),-1)
    HOLD["A"]=None; HOLD["mlp"]=False
    return -lp.gather(-1,torch.tensor(seq[1:]).to(DEV).unsqueeze(-1)).mean().item()

norm=[p for n_,p in model.named_parameters() if "norm" in n_.lower()]
orig=[p.detach().clone() for p in norm]
# Only normalization gains are trainable.  Freezing the weight matrices avoids
# allocating their gradients while retaining gradients through them to norms.
for p in model.parameters(): p.requires_grad_(False)
for p in norm: p.requires_grad_(True)
HOLD["A"]=None
intact=sum(-torch.log_softmax(model(torch.tensor([sq]).to(DEV)).logits[0,:-1].float(),-1).gather(-1,torch.tensor(sq[1:]).to(DEV).unsqueeze(-1)).mean().item() for sq in eval_chunks)/len(eval_chunks)
print(f"intact {intact:.4f}")

print("\nUNHEALED damage vs #heads (+ fixed 6 MLP LUTs): fitted vs positional-only")
print(f"{'k':>6}{'fitted':>10}{'positional':>12}{'gap':>9}")
curve=[]
for k in [40,80,120,160,200,256]:
    heads=HEADS_ALL[:k]
    Af=[head_A(sq,heads,set()) for sq in eval_chunks]
    Ap=[head_A(sq,heads,CONTENT) for sq in eval_chunks]
    df=sum(loss(sq,A) for sq,A in zip(eval_chunks,Af))/len(eval_chunks)-intact
    dp=sum(loss(sq,A) for sq,A in zip(eval_chunks,Ap))/len(eval_chunks)-intact
    curve.append((k,df,dp,dp-df))
    print(f"{k:>6}{df:>+10.3f}{dp:>+12.3f}{dp-df:>+9.3f}",flush=True); gc.collect()

with open("posneg_curve_results.csv", "w", newline="") as f:
    w=csv.writer(f); w.writerow(["heads","fitted_damage_nats","positional_damage_nats","content_gain_nats"]); w.writerows(curve)
print("wrote posneg_curve_results.csv", flush=True)

if SKIP_HEAL:
    for h in hooks: h.remove()
    print("skipping healed confirmation (--skip-heal)")
    raise SystemExit(0)

def heal_eval(k,drop):
    heads=HEADS_ALL[:k]; A_tr=[head_A(sq,heads,drop) for sq in train_chunks]
    for p,o in zip(norm,orig): p.data.copy_(o)
    opt=torch.optim.Adam(norm,lr=LR); model.train()
    for ep in range(EPOCHS):
        for sq,A in zip(train_chunks,A_tr):
            SEQ["s"]=sq; HOLD["A"]=A; HOLD["mlp"]=True
            ids=torch.tensor([sq]).to(DEV); out=model(ids,labels=ids)
            opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step(); HOLD["A"]=None; HOLD["mlp"]=False
    model.eval()
    A_ev=[head_A(sq,heads,drop) for sq in eval_chunks]
    d=sum(loss(sq,A) for sq,A in zip(eval_chunks,A_ev))/len(eval_chunks)-intact
    for p,o in zip(norm,orig): p.data.copy_(o)
    return d
print("\nHEALED at k=160 (10-epoch, + fixed 6 MLP LUTs):")
print(f"  fitted:      {heal_eval(160,set()):+.3f}")
print(f"  positional:  {heal_eval(160,CONTENT):+.3f}")
for h in hooks: h.remove()
print("read: gap ~ 0 => content buys nothing in the substitutable set; gap grows where copying matters.")
