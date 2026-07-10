"""Does qwen PRE-COMPOSE negation (linearly readable) or BAG the words
(needs an interaction)? Compare a linear probe to nonlinear ceilings, LOO.
"""
import json, numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, LeaveOneOut

rows = json.load(open("compose_labeled.json"))
E = np.load("compose_emb.npy")
y = np.array([r["label"] for r in rows])
lex = np.array([r["lex"] for r in rows]); neg = np.array([r["neg"] for r in rows])
loo = LeaveOneOut()
def foo(clf): return (cross_val_predict(clf, E, y, cv=loo) == y).mean()

print("=== LOO fidelity to ministral on the compositional task (3-class) ===")
print(f"  majority-class baseline     {max(Counter(y).values())/len(y):.3f}")
lin = LogisticRegression(max_iter=3000, C=1.0)
print(f"  LINEAR (logistic regression) {foo(lin):.3f}")
print(f"  decision tree (depth<=4)     {foo(DecisionTreeClassifier(max_depth=4, random_state=0)):.3f}")
print(f"  MLP (64,)                    {foo(MLPClassifier((64,), max_iter=3000, random_state=0)):.3f}")
print(f"  RBF-SVM                      {foo(SVC(kernel='rbf', C=4, gamma='scale')):.3f}")

# per-cell accuracy of the LINEAR probe -- where does it break?
linpred = cross_val_predict(lin, E, y, cv=loo)
names = {(1,0):'great',(1,1):'not great',(-1,0):'terrible',(-1,1):'not terrible'}
print("\n=== linear probe accuracy per cell (does it handle negation?) ===")
for l in (1,-1):
    for n in (0,1):
        m = (lex==l)&(neg==n)
        print(f"  {names[(l,n)]:<14} acc={ (linpred[m]==y[m]).mean():.2f}  (n={m.sum()})")

# is 'negation' itself a linear direction in qwen space? is 'lex'?
def decode(target):
    return (cross_val_predict(LogisticRegression(max_iter=2000), E, target, cv=loo)==target).mean()
print("\n=== linear decodability of the underlying factors ===")
print(f"  decode NEGATED (0/1) from embedding: {decode(neg):.3f}")
print(f"  decode LEX polarity (+/-)          : {decode(lex):.3f}")

# the tell: correlation of our old single polarity axis with the TRUE label on negated items
unit=lambda v:v/np.linalg.norm(v)
mu=lambda c:E[y==c].mean(0)
axis=unit(unit(mu('positive'))-unit(mu('negative')))
proj=E@axis
print("\n=== single polarity axis: mean projection per cell ===")
for l in (1,-1):
    for n in (0,1):
        m=(lex==l)&(neg==n)
        print(f"  {names[(l,n)]:<14} proj={proj[m].mean():+.3f}  ministral={Counter(y[m]).most_common(1)[0][0]}")
