"""
Accident high-risk hotspot classification (parallels withdrawn patent KR20170065898A,
which used a binary logit). Compares Logistic Regression (= patent baseline) vs
XGBoost vs GraphSAGE GNN on real data.go.kr accident-hotspot data.
Temporal split: train year<2022, test year==2022.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
import torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

df = pd.read_csv('data/kaccident_hotspots.csv')
df = df.dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd'] = df['sido_cd'].astype(int)

# ── engineered spatial features (no target leakage) ─────────────────────────
coords = df[['lat','lon']].values
nn = NearestNeighbors(n_neighbors=9).fit(coords)
dist, idx = nn.kneighbors(coords)            # incl self at col 0
df['knn_dist_mean'] = dist[:,1:].mean(1)     # mean dist to 8 neighbors (density proxy)
df['knn_dist_min']  = dist[:,1].copy()
# local density: neighbors within ~0.05 deg (~5km)
from scipy.spatial import cKDTree
tree = cKDTree(coords)
df['density_5km'] = [len(tree.query_ball_point(c, 0.05))-1 for c in coords]

FEATS = ['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km']
# label: high-risk = accident count >= train 75th percentile
train_mask = (df['year'] < 2022).values
test_mask  = (df['year'] == 2022).values
thr = np.percentile(df.loc[train_mask,'occrrnc_cnt'], 75)
y = (df['occrrnc_cnt'] >= thr).astype(int).values
print(f'high-risk threshold (75pct count) = {thr:.0f} | positives: train {y[train_mask].mean():.2f} test {y[test_mask].mean():.2f}')

X = df[FEATS].values.astype(float)
sc = StandardScaler().fit(X[train_mask]); Xs = sc.transform(X)

def report(name, ytrue, prob):
    pred = (prob>=0.5).astype(int)
    print(f'{name:14s} AUC {roc_auc_score(ytrue,prob):.3f} | F1 {f1_score(ytrue,pred):.3f} | Acc {accuracy_score(ytrue,pred):.3f}')
    return roc_auc_score(ytrue,prob)

res={}
# 1) Logistic Regression (patent-class baseline)
lr = LogisticRegression(max_iter=2000).fit(Xs[train_mask], y[train_mask])
res['Logit'] = report('Logit', y[test_mask], lr.predict_proba(Xs[test_mask])[:,1])
# 2) XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                    eval_metric='logloss', verbosity=0).fit(X[train_mask], y[train_mask])
res['XGBoost'] = report('XGBoost', y[test_mask], xgb.predict_proba(X[test_mask])[:,1])

# 3) GraphSAGE GNN over k-NN spatial graph
ei = []
for i in range(len(df)):
    for j in idx[i,1:]:
        ei.append([i,j]); ei.append([j,i])
edge_index = torch.tensor(np.array(ei).T, dtype=torch.long)
xt = torch.tensor(Xs, dtype=torch.float)
yt = torch.tensor(y, dtype=torch.long)
data = Data(x=xt, edge_index=edge_index, y=yt)
trm = torch.tensor(train_mask); tem = torch.tensor(test_mask)

class SAGE(torch.nn.Module):
    def __init__(self, d):
        super().__init__(); self.c1=SAGEConv(d,32); self.c2=SAGEConv(32,16); self.lin=torch.nn.Linear(16,2)
    def forward(self,x,ei):
        x=F.relu(self.c1(x,ei)); x=F.dropout(x,0.3,self.training); x=F.relu(self.c2(x,ei)); return self.lin(x)

torch.manual_seed(0)
m=SAGE(Xs.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=0.01,weight_decay=5e-4)
w=torch.tensor([1.0, (y[train_mask]==0).sum()/(y[train_mask]==1).sum()],dtype=torch.float)
for ep in range(200):
    m.train(); opt.zero_grad(); out=m(data.x,data.edge_index)
    loss=F.cross_entropy(out[trm], yt[trm], weight=w); loss.backward(); opt.step()
m.eval()
with torch.no_grad():
    prob=F.softmax(m(data.x,data.edge_index),1)[:,1].numpy()
res['GraphSAGE'] = report('GraphSAGE', y[test_mask], prob[test_mask])

print('\nSUMMARY (test-year 2022 AUC):', {k:round(v,3) for k,v in res.items()})
import json; json.dump({k:round(v,3) for k,v in res.items()}, open('model_results.json','w'))
