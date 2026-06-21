"""Location-level accident-risk classification with pseudo-absence sampling
(parallels the patent's per-location accident-probability framing).
Positives = real hotspots; negatives = pseudo-absence points sampled within each
province's extent, >=3km from any hotspot but <=30km (plausible road locations)."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch, torch.nn.functional as F
from torch_geometric.nn import SAGEConv; from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

rng=np.random.default_rng(0)
pos=pd.read_csv('data/kaccident_hotspots.csv').dropna(subset=['lat','lon']).reset_index(drop=True)
postree=cKDTree(pos[['lat','lon']].values)
# pseudo-absence per province
negs=[]
for sido,g in pos.groupby('sido_cd'):
    la0,la1=g['lat'].min(),g['lat'].max(); lo0,lo1=g['lon'].min(),g['lon'].max()
    yrs=g['year'].values
    need=len(g); tries=0
    while need>0 and tries<need*200:
        tries+=1
        la=rng.uniform(la0,la1); lo=rng.uniform(lo0,lo1)
        d,_=postree.query([la,lo])
        if 0.03<=d<=0.30:   # ~3km..~33km from nearest hotspot
            negs.append({'lat':la,'lon':lo,'sido_cd':sido,'year':int(rng.choice(yrs))}); need-=1
neg=pd.DataFrame(negs)
pos2=pos[['lat','lon','sido_cd','year']].copy(); pos2['label']=1; neg['label']=0
df=pd.concat([pos2,neg],ignore_index=True).reset_index(drop=True)
print('positives',int((df.label==1).sum()),'negatives',int((df.label==0).sum()))

# spatial feature: # hotspots within 5km (exclude self-distance 0 for positives)
allc=df[['lat','lon']].values
hot=pos[['lat','lon']].values; ht=cKDTree(hot)
def dens(c,is_pos):
    n=len(ht.query_ball_point(c,0.05))
    return n-1 if is_pos else n
df['density_5km']=[dens(c,l==1) for c,l in zip(allc,df['label'].values)]
nn=NearestNeighbors(n_neighbors=9).fit(allc); dist,idx=nn.kneighbors(allc)
df['knn_dist_mean']=dist[:,1:].mean(1)
FEATS=['lat','lon','sido_cd','year','density_5km','knn_dist_mean']
X=df[FEATS].values.astype(float); y=df['label'].values
out={'pos':int((y==1).sum()),'neg':int((y==0).sum())}

def cv(model_fn,scale=False):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); a=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]
        if scale: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
        m=model_fn(); m.fit(Xtr,y[tr]); a.append(roc_auc_score(y[te],m.predict_proba(Xte)[:,1]))
    return float(np.mean(a)),float(np.std(a))
lr=cv(lambda:LogisticRegression(max_iter=2000),True)
xg=cv(lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0))
out['Logit_AUC']=f'{lr[0]:.3f}±{lr[1]:.3f}'; out['XGB_AUC']=f'{xg[0]:.3f}±{xg[1]:.3f}'

# GraphSAGE (single split for speed)
sc=StandardScaler().fit(X); Xs=sc.transform(X)
ei=[];
for i in range(len(df)):
    for j in idx[i,1:]: ei+=[[i,j],[j,i]]
data=Data(x=torch.tensor(Xs,dtype=torch.float),edge_index=torch.tensor(np.array(ei).T,dtype=torch.long),y=torch.tensor(y))
from sklearn.model_selection import train_test_split
tri,tei=train_test_split(np.arange(len(df)),test_size=0.3,stratify=y,random_state=0)
trm=torch.zeros(len(df),dtype=torch.bool); trm[tri]=True; tem=torch.zeros(len(df),dtype=torch.bool); tem[tei]=True
class SAGE(torch.nn.Module):
    def __init__(s,d): super().__init__(); s.c1=SAGEConv(d,32); s.c2=SAGEConv(32,16); s.l=torch.nn.Linear(16,2)
    def forward(s,x,e): x=F.relu(s.c1(x,e)); x=F.dropout(x,0.3,s.training); x=F.relu(s.c2(x,e)); return s.l(x)
torch.manual_seed(0); m=SAGE(X.shape[1]); opt=torch.optim.Adam(m.parameters(),lr=0.01,weight_decay=5e-4)
for ep in range(150):
    m.train(); opt.zero_grad(); o=m(data.x,data.edge_index); F.cross_entropy(o[trm],data.y[trm]).backward(); opt.step()
m.eval()
with torch.no_grad(): p=F.softmax(m(data.x,data.edge_index),1)[:,1].numpy()
out['GraphSAGE_AUC']=round(float(roc_auc_score(y[tei],p[tei])),3)
print(out); json.dump(out,open('location_results.json','w'))
