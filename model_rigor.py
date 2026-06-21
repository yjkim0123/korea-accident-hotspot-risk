"""Rigour / robustness experiments answering a strict reviewer (existing data,
no fabrication). Reports numbers AS-IS even if they weaken the headline.
  (A) Spatial block CV: leave-one-province-out (no same-region info at test).
  (B) Novel-location test: 2022 sites NOT near any train hotspot (quantifies
      spatial memorisation / leakage across the temporal split).
  (C) Persistence baseline: predict 2022 risk from the nearest past-year count.
  (D) Density-dependence ablation: drop knn/density features.
  (E) Bootstrap 95% CIs for XGBoost & RF test AUC and their difference.
"""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
nn=NearestNeighbors(n_neighbors=9).fit(coords); dist,idx=nn.kneighbors(coords)
df['knn_dist_mean']=dist[:,1:].mean(1); df['knn_dist_min']=dist[:,1].copy()
tree=cKDTree(coords); df['density_5km']=[len(tree.query_ball_point(c,0.05))-1 for c in coords]

SPATIAL=['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km']
ROAD=['road_rank','n_roads','lanes','maxspeed_v']
FULL=SPATIAL+ROAD
NODENS=['lat','lon','sido_cd','year']+ROAD   # drop knn/density
train=(df['year']<2022).values; test=(df['year']==2022).values
thr=np.percentile(df.loc[train,'occrrnc_cnt'],75)
y=(df['occrrnc_cnt']>=thr).astype(int).values
mkXGB=lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0)
mkLogit=lambda:LogisticRegression(max_iter=2000)
mkRF=lambda:RandomForestClassifier(n_estimators=400,random_state=0)
out={'thr':float(thr)}

def auc_fit(feats,mk,trm,tem,scale=False):
    X=df[feats].values.astype(float); Xtr,Xte=X[trm],X[tem]
    if scale: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
    m=mk(); m.fit(Xtr,y[trm]); p=m.predict_proba(Xte)[:,1]
    return roc_auc_score(y[tem],p),p

# (A) Leave-one-province-out spatial CV (XGB full)
aucs=[]; pooled_y=[]; pooled_p=[]
for s,g in df.groupby('sido_cd'):
    tem=(df['sido_cd']==s).values; trm=~tem
    if y[tem].sum()<5 or y[tem].sum()==tem.sum(): continue
    a,p=auc_fit(FULL,mkXGB,trm,tem); aucs.append(a)
    pooled_y+=list(y[tem]); pooled_p+=list(p)
out['A_LOPO']={'mean_region_AUC':round(float(np.mean(aucs)),3),'std':round(float(np.std(aucs)),3),
               'pooled_AUC':round(float(roc_auc_score(pooled_y,pooled_p)),3),'n_regions':len(aucs)}
print('A LOPO',out['A_LOPO'])

# (B) Novel-location temporal test (XGB full): 2022 sites far from any train hotspot
trtree=cKDTree(coords[train])
d2022,_=trtree.query(coords[test]); novel=d2022>0.0015   # ~150 m
a_full,pte=auc_fit(FULL,mkXGB,train,test)
ti=np.where(test)[0]; yte=y[ti]
def sub_auc(mask):
    if mask.sum()>10 and 0<yte[mask].sum()<mask.sum():
        return round(float(roc_auc_score(yte[mask],pte[mask])),3),int(mask.sum()),int(yte[mask].sum())
    return None
out['B_novel']={'full_test_AUC':round(float(a_full),3),'novel_AUC':sub_auc(novel),
                'seen_AUC':sub_auc(~novel),'novel_frac':round(float(novel.mean()),3)}
print('B novel',out['B_novel'])

# (C) Persistence baseline: nearest past (<=2021) hotspot's count predicts 2022 risk
past=train; ptree=cKDTree(coords[past]); pcnt=df['occrrnc_cnt'].values[past]
dd,ii=ptree.query(coords[test]); score=np.where(dd<=0.0015,pcnt[ii],0.0)
out['C_persistence']={'AUC':round(float(roc_auc_score(yte,score)),3),
                      'matched_frac':round(float((dd<=0.0015).mean()),3)}
print('C persistence',out['C_persistence'])

# (D) Density-dependence ablation (test 2022)
dd_xgb,_=auc_fit(NODENS,mkXGB,train,test); dd_lr,_=auc_fit(NODENS,mkLogit,train,test,scale=True)
full_xgb,_=auc_fit(FULL,mkXGB,train,test); full_lr,_=auc_fit(FULL,mkLogit,train,test,scale=True)
out['D_density']={'XGB_full':round(float(full_xgb),3),'XGB_no_density':round(float(dd_xgb),3),
                  'Logit_full':round(float(full_lr),3),'Logit_no_density':round(float(dd_lr),3)}
print('D density',out['D_density'])

# (E) Bootstrap 95% CI for XGB & RF test AUC + difference
_,pX=auc_fit(FULL,mkXGB,train,test); _,pR=auc_fit(FULL,mkRF,train,test)
rng=np.random.default_rng(0); bx=[]; br=[]; bd=[]
n=len(yte)
for _ in range(2000):
    bi=rng.integers(0,n,n)
    if yte[bi].sum()==0 or yte[bi].sum()==n: continue
    ax=roc_auc_score(yte[bi],pX[bi]); ar=roc_auc_score(yte[bi],pR[bi])
    bx.append(ax); br.append(ar); bd.append(ar-ax)
ci=lambda v:[round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)]
out['E_bootstrap']={'XGB_AUC_CI':ci(bx),'RF_AUC_CI':ci(br),'RF_minus_XGB_CI':ci(bd),
                    'diff_includes_0':bool(np.percentile(bd,2.5)<=0<=np.percentile(bd,97.5))}
print('E bootstrap',out['E_bootstrap'])

json.dump(out,open('rigor_results.json','w'),indent=2)
print('DONE_RIGOR')
