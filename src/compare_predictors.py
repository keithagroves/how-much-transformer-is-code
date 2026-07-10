"""Which representation actually predicts ministral's next token?
Same train/test split, several predictors head-to-head.
"""
import json, numpy as np
from collections import Counter, defaultdict

recs = json.load(open("store.json")); emb = np.load("store_emb.npy")
pref = [r["prefix"] for r in recs]; tok = np.array([r["token"] for r in recs])
n = len(recs); rng = np.random.default_rng(1); perm = rng.permutation(n)
cut = int(n*0.8); tr, te = perm[:cut], perm[cut:]
last = lambda s, k: " ".join(s.split()[-k:]).lower()

# build suffix tables from TRAIN only
uni, bi = defaultdict(Counter), defaultdict(Counter)
freq = Counter()
for i in tr:
    freq[tok[i]] += 1
    uni[last(pref[i],1)][tok[i]] += 1
    bi[last(pref[i],2)][tok[i]] += 1
top_freq = freq.most_common(1)[0][0]

def pred_freq(i, topn):   return [t for t,_ in freq.most_common(topn)]
def pred_uni(i, topn):
    c = uni.get(last(pref[i],1))
    return [t for t,_ in (c or freq).most_common(topn)]
def pred_bi(i, topn):
    c = bi.get(last(pref[i],2)) or uni.get(last(pref[i],1)) or freq   # backoff
    return [t for t,_ in c.most_common(topn)]
def pred_knn(i, topn, k=8):
    sims = emb[tr] @ emb[i]
    idx = np.argpartition(-sims, min(k,len(sims)-1))[:k]
    v = Counter()
    for j in idx: v[tok[tr[j]]] += sims[j]
    return [t for t,_ in v.most_common(topn)]
def pred_hybrid(i, topn):
    # trust an exact bigram if we've seen it, else fall back to embedding kNN
    c = bi.get(last(pref[i],2))
    return [t for t,_ in c.most_common(topn)] if c else pred_knn(i, topn)

preds = {"most-freq":pred_freq, "embedding-kNN":pred_knn,
         "last-1-word":pred_uni, "last-2-word(backoff)":pred_bi,
         "bigram-or-kNN":pred_hybrid}
print(f"store={n} train={len(tr)} test={len(te)} |vocab|={len(set(tok))}")
print(f"  {'predictor':<24}{'top1':>7}{'top3':>7}")
for name, fn in preds.items():
    c1 = sum(fn(i,3)[0]==tok[i] for i in te)
    c3 = sum(tok[i] in fn(i,3) for i in te)
    print(f"  {name:<24}{c1/len(te):>7.3f}{c3/len(te):>7.3f}")

# how often is the test bigram context even present in train? (ceiling for exact match)
seen_bi = sum(1 for i in te if last(pref[i],2) in bi)/len(te)
seen_uni = sum(1 for i in te if last(pref[i],1) in uni)/len(te)
print(f"\n  test contexts seen in train:  last-2word={seen_bi:.2f}  last-1word={seen_uni:.2f}")
