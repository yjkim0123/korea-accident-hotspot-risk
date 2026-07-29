"""
Spatial-extrapolation gap: diagnose + close it. NEW methods paper (IEEE Access).
All transferable features are FOLD-SAFE: computed from TRAIN points only each
leave-one-province-out (LOPO) fold (no test-region labels, no global leakage).

Variants (LOPO mean-region AUC):
  V0 baseline (region-identity): lat,lon,sido_cd + road + GLOBAL density  (the naive ~0.75)
  V1 transferable-only         : drop sido_cd & absolute coords; fold-safe density/knn/exposure + road
  V2 = V1 + CORAL              : 2nd-order feature alignment train->target
  V3 = V1 + importance weight  : density-ratio covariate-shift correction
Also random 5-fold reference (the inflated ~0.95) to quantify the gap.
Reports numbers AS-IS.
"""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os
# run from the repository root (where data/ lives)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
ROAD=['road_rank','n_roads','lanes','maxspeed_v']
thr=np.percentile(df['occrrnc_cnt'],75)
y=(df['occrrnc_cnt']>=thr).astype(int).values
N=len(df)
def mkXGB(): return XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',verbosity=0)
out={'thr':float(thr),'n':N,'n_sido':int(df['sido_cd'].nunique())}

# ---- fold-safe transferable feature builders (fit on train idx, apply to all) ----
def transfer_feats(tr_idx):
    """Return (N x d) transferable features built ONLY from training points/labels."""
    ctr=coords[tr_idx]; occ_tr=df['occrrnc_cnt'].values[tr_idx]
    # (a) fold-safe local density: # train points within ~5km (0.05 deg)
    tree=cKDTree(ctr)
    dens=np.array([len(tree.query_ball_point(c,0.05)) for c in coords],dtype=float)
    # (b) fold-safe knn distance to train points (mean of 5 nearest train)
    k=min(6,len(tr_idx)); nn=NearestNeighbors(n_neighbors=k).fit(ctr); dknn,_=nn.kneighbors(coords)
    knn_mean=dknn[:,1:].mean(1); knn_min=dknn[:,0]
    # (c) fold-safe spatial EXPOSURE prior: KNN-regression of occ count from train labels
    kk=min(8,len(tr_idx)); nn2=NearestNeighbors(n_neighbors=kk).fit(ctr); dE,iE=nn2.kneighbors(coords)
    expo=occ_tr[iE].mean(1)                 # expected count from nearest train analogs
    expo_w=(occ_tr[iE]/(1+dE)).sum(1)/(1/(1+dE)).sum(1)  # distance-weighted
    F=np.column_stack([dens,knn_mean,knn_min,expo,expo_w]+[df[c].values for c in ROAD])
    return F  # 5 transferable spatial + 4 road = 9 dims

def coral(Xs,Xt,Xeval):
    """Align source feature 2nd-order stats to target. Returns transformed Xs and Xeval."""
    eps=1e-4; d=Xs.shape[1]
    Cs=np.cov(Xs,rowvar=False)+eps*np.eye(d); Ct=np.cov(Xt,rowvar=False)+eps*np.eye(d)
    def msqrt(C):
        w,V=np.linalg.eigh(C); w=np.clip(w,1e-8,None); return V@np.diag(np.sqrt(w))@V.T
    def minvsqrt(C):
        w,V=np.linalg.eigh(C); w=np.clip(w,1e-8,None); return V@np.diag(1/np.sqrt(w))@V.T
    A=minvsqrt(Cs)@msqrt(Ct)
    mu_s=Xs.mean(0); mu_t=Xt.mean(0)
    f=lambda X:(X-mu_s)@A+mu_t
    return f(Xs),f(Xeval)

def imp_weights(Xs,Xt):
    """density-ratio P(target)/P(source) via a domain classifier -> sample weights for source."""
    Xd=np.vstack([Xs,Xt]); yd=np.r_[np.zeros(len(Xs)),np.ones(len(Xt))]
    mu=Xd.mean(0); sd=Xd.std(0)+1e-8; Xn=(Xd-mu)/sd
    clf=LogisticRegression(max_iter=1000,C=1.0).fit(Xn,yd)
    ps=clf.predict_proba((Xs-mu)/sd)[:,1]; ps=np.clip(ps,1e-3,1-1e-3)
    w=ps/(1-ps); w=w/ w.mean()
    return np.clip(w,0.1,10)

# ---- V0 baseline: region-identity features, GLOBAL density (the naive pipeline) ----
treeG=cKDTree(coords); densG=np.array([len(treeG.query_ball_point(c,0.05))-1 for c in coords],float)
nnG=NearestNeighbors(n_neighbors=9).fit(coords); dG,_=nnG.kneighbors(coords)
X0=np.column_stack([df['lat'],df['lon'],df['sido_cd'],densG,dG[:,1:].mean(1)]+[df[c] for c in ROAD])

def lopo(score_fn):
    aucs=[]
    for s in sorted(df['sido_cd'].unique()):
        tem=(df['sido_cd']==s).values; trm=~tem
        if y[tem].sum()<5 or y[tem].sum()==tem.sum(): continue
        p=score_fn(np.where(trm)[0],np.where(tem)[0])
        aucs.append(roc_auc_score(y[tem],p))
    return aucs

def sc_v0(tr,te):
    m=mkXGB(); m.fit(X0[tr],y[tr]); return m.predict_proba(X0[te])[:,1]
def sc_v1(tr,te):
    F=transfer_feats(tr); m=mkXGB(); m.fit(F[tr],y[tr]); return m.predict_proba(F[te])[:,1]
def sc_v2(tr,te):
    F=transfer_feats(tr); Xs,Xe=coral(F[tr],F[te],F[te]); m=mkXGB(); m.fit(Xs,y[tr]); return m.predict_proba(Xe)[:,1]
def sc_v3(tr,te):
    F=transfer_feats(tr); w=imp_weights(F[tr],F[te]); m=mkXGB(); m.fit(F[tr],y[tr],sample_weight=w); return m.predict_proba(F[te])[:,1]

for name,fn in [('V0_baseline',sc_v0),('V1_transferable',sc_v1),('V2_CORAL',sc_v2),('V3_impweight',sc_v3)]:
    a=lopo(fn); out[name]={'LOPO_mean_AUC':round(float(np.mean(a)),3),'std':round(float(np.std(a)),3),'n_regions':len(a)}
    print(name,out[name],flush=True)

# random 5-fold reference (inflated) using V1 features built per-fold from train
sk=StratifiedKFold(5,shuffle=True,random_state=0); rand=[]
for tr,te in sk.split(np.zeros(N),y):
    F=transfer_feats(tr); m=mkXGB(); m.fit(F[tr],y[tr]); rand.append(roc_auc_score(y[te],m.predict_proba(F[te])[:,1]))
out['RANDOM_5fold_ref']={'AUC':round(float(np.mean(rand)),3),'std':round(float(np.std(rand)),3)}
print('RANDOM ref',out['RANDOM_5fold_ref'],flush=True)

json.dump(out,open('spatial_results.json','w'),indent=2)
print('=== DONE ===\n',json.dumps(out,indent=2))
