"""Round 2: try variants that might actually BEAT the V0 baseline (LOPO 0.721).
Key insight from round 1: target-region OBSERVABLE GEOMETRY (label-free) transfers;
standard DA does not. Test label-safe transductive features + ensembles + smoothing."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int); coords=df[['lat','lon']].values
ROAD=['road_rank','n_roads','lanes','maxspeed_v']
thr=np.percentile(df['occrrnc_cnt'],75); y=(df['occrrnc_cnt']>=thr).astype(int).values; N=len(df)
mk=lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,eval_metric='logloss',verbosity=0)

# label-free transductive geometry (uses ALL locations, NO labels) -- legit at test time
treeG=cKDTree(coords)
densG=np.array([len(treeG.query_ball_point(c,0.05))-1 for c in coords],float)
nnG=NearestNeighbors(n_neighbors=9).fit(coords); dG,_=nnG.kneighbors(coords); knnG=dG[:,1:].mean(1)
X0=np.column_stack([df['lat'],df['lon'],df['sido_cd'],densG,knnG]+[df[c] for c in ROAD])  # baseline

def train_exposure(tr_idx):
    ctr=coords[tr_idx]; occ=df['occrrnc_cnt'].values[tr_idx]
    kk=min(8,len(tr_idx)); nn=NearestNeighbors(n_neighbors=kk).fit(ctr); dE,iE=nn.kneighbors(coords)
    return occ[iE].mean(1), (occ[iE]/(1+dE)).sum(1)/(1/(1+dE)).sum(1)

def feats_transd(tr_idx):
    e1,e2=train_exposure(tr_idx)
    # label-free geometry (densG,knnG) + train-exposure + road  (NO absolute coords, NO sido)
    return np.column_stack([densG,knnG,e1,e2]+[df[c].values for c in ROAD])

def feats_aug(tr_idx):  # baseline + train-exposure (augmented region-identity)
    e1,e2=train_exposure(tr_idx)
    return np.column_stack([X0,e1,e2])

def lopo(scorer):
    a=[]
    for s in sorted(df['sido_cd'].unique()):
        tem=(df['sido_cd']==s).values; trm=~tem
        if y[tem].sum()<5 or y[tem].sum()==tem.sum(): continue
        a.append(roc_auc_score(y[tem],scorer(np.where(trm)[0],np.where(tem)[0],s)))
    return a
out={}
def Xfit(X,tr,te): m=mk(); m.fit(X[tr],y[tr]); return m.predict_proba(X[te])[:,1]

# V0 baseline
out['V0_baseline']=lopo(lambda tr,te,s: Xfit(X0,tr,te))
# V4: baseline + train-exposure augment
out['V4_aug_exposure']=lopo(lambda tr,te,s:(lambda F:Xfit(F,tr,te))(feats_aug(tr)))
# V5: transductive-geometry (label-free) only
out['V5_transductive']=lopo(lambda tr,te,s:(lambda F:Xfit(F,tr,te))(feats_transd(tr)))
# V6: ensemble of V0 and V5 (mean prob)
def ens(tr,te,s):
    p0=Xfit(X0,tr,te); F=feats_transd(tr); p5=Xfit(F,tr,te); return 0.5*(p0+p5)
out['V6_ensemble']=lopo(ens)
# V7: spatial smoothing of V0 preds within target region (transductive, target geometry)
def smooth(tr,te,s):
    p0=Xfit(X0,tr,te); ct=coords[te]
    if len(te)<6: return p0
    nn=NearestNeighbors(n_neighbors=min(6,len(te))).fit(ct); d,i=nn.kneighbors(ct)
    return 0.5*p0+0.5*p0[i].mean(1)
out['V7_smooth']=lopo(smooth)

res={k:{'LOPO_mean_AUC':round(float(np.mean(v)),3),'std':round(float(np.std(v)),3),'n':len(v)} for k,v in out.items()}
json.dump(res,open('spatial_results2.json','w'),indent=2)
print(json.dumps(res,indent=2))
