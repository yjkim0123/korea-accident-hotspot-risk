"""Re-train high-risk hotspot classification WITH OSM road features added,
and report the delta vs the spatial-only baseline. This directly tests whether
the patent-style road variables (class, lanes, speed, junction complexity)
improve modern ML prediction. Temporal hold-out (train<2022, test 2022) + 5-fold CV.
Outputs road_model_results.json. Honest: prints baseline vs +road for each model."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
nn=NearestNeighbors(n_neighbors=9).fit(coords); dist,idx=nn.kneighbors(coords)
df['knn_dist_mean']=dist[:,1:].mean(1); df['knn_dist_min']=dist[:,1].copy()
tree=cKDTree(coords); df['density_5km']=[len(tree.query_ball_point(c,0.05))-1 for c in coords]

BASE=['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km']
ROAD=['road_rank','n_roads','lanes','maxspeed_v']
train=(df['year']<2022).values; test=(df['year']==2022).values
thr=np.percentile(df.loc[train,'occrrnc_cnt'],75)
y=(df['occrrnc_cnt']>=thr).astype(int).values
print(f'thr={thr:.0f} pos: train {y[train].mean():.2f} test {y[test].mean():.2f}')

def run(feats,tag):
    X=df[feats].values.astype(float)
    sc=StandardScaler().fit(X[train]); Xs=sc.transform(X)
    out={}
    lr=LogisticRegression(max_iter=2000).fit(Xs[train],y[train])
    p=lr.predict_proba(Xs[test])[:,1]
    out['Logit']={'AUC':round(roc_auc_score(y[test],p),3),'F1':round(f1_score(y[test],(p>=.5)),3)}
    xg=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,
                     eval_metric='logloss',verbosity=0).fit(X[train],y[train])
    p=xg.predict_proba(X[test])[:,1]
    out['XGBoost']={'AUC':round(roc_auc_score(y[test],p),3),'F1':round(f1_score(y[test],(p>=.5)),3)}
    # 5-fold CV AUC (XGB + Logit)
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for name,mk in [('Logit_cv',lambda:LogisticRegression(max_iter=2000)),
                    ('XGB_cv',lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0))]:
        a=[]
        for tr,te in skf.split(X,y):
            Xtr,Xte=X[tr],X[te]
            if 'Logit' in name:
                s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
            mm=mk(); mm.fit(Xtr,y[tr]); a.append(roc_auc_score(y[te],mm.predict_proba(Xte)[:,1]))
        out[name]={'AUC':round(float(np.mean(a)),3),'std':round(float(np.std(a)),3)}
    # GraphSAGE
    ei=[]
    for i in range(len(df)):
        for j in idx[i,1:]: ei+=[[i,j],[j,i]]
    data=Data(x=torch.tensor(Xs,dtype=torch.float),edge_index=torch.tensor(np.array(ei).T,dtype=torch.long),y=torch.tensor(y))
    trm=torch.tensor(train); tem=torch.tensor(test)
    class SAGE(torch.nn.Module):
        def __init__(s,d): super().__init__(); s.c1=SAGEConv(d,32); s.c2=SAGEConv(32,16); s.l=torch.nn.Linear(16,2)
        def forward(s,x,e): x=F.relu(s.c1(x,e)); x=F.dropout(x,0.3,s.training); x=F.relu(s.c2(x,e)); return s.l(x)
    torch.manual_seed(0); mm=SAGE(Xs.shape[1]); opt=torch.optim.Adam(mm.parameters(),lr=0.01,weight_decay=5e-4)
    w=torch.tensor([1.0,(y[train]==0).sum()/(y[train]==1).sum()],dtype=torch.float)
    for ep in range(200):
        mm.train(); opt.zero_grad(); o=mm(data.x,data.edge_index)
        F.cross_entropy(o[trm],data.y[trm],weight=w).backward(); opt.step()
    mm.eval()
    with torch.no_grad(): p=F.softmax(mm(data.x,data.edge_index),1)[:,1].numpy()
    out['GraphSAGE']={'AUC':round(float(roc_auc_score(y[test],p[test])),3)}
    print(tag,out); return out

res={'baseline':run(BASE,'BASELINE (spatial only)'),
     'with_road':run(BASE+ROAD,'WITH ROAD FEATURES')}
json.dump(res,open('road_model_results.json','w'),indent=2)
print('\nSaved road_model_results.json')
print('Delta XGB AUC (test):',res['with_road']['XGBoost']['AUC']-res['baseline']['XGBoost']['AUC'])
