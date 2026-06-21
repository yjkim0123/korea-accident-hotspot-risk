"""Expansion experiments for the accident paper (existing data only, no new
collection, no fabrication):
  (1) Ablation: spatial-only vs road-only vs full feature groups.
  (2) Extended baselines: Logit, kNN, MLP, RandomForest, XGBoost on full features.
  (3) Per-province (si-do) AUC on the 2022 hold-out, to show national generality.
  (4) Figures: road-class vs high-risk rate, PR curves, calibration, SHAP beeswarm.
Outputs expand_results.json + figures fig_*.png."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
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
train=(df['year']<2022).values; test=(df['year']==2022).values
thr=np.percentile(df.loc[train,'occrrnc_cnt'],75)
y=(df['occrrnc_cnt']>=thr).astype(int).values
out={'threshold':float(thr)}

def cv_auc(feats,mk,scale=False):
    X=df[feats].values.astype(float); skf=StratifiedKFold(5,shuffle=True,random_state=0); a=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]
        if scale: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
        m=mk(); m.fit(Xtr,y[tr]); a.append(roc_auc_score(y[te],m.predict_proba(Xte)[:,1]))
    return float(np.mean(a)),float(np.std(a))

def test_auc(feats,mk,scale=False):
    X=df[feats].values.astype(float); Xtr,Xte=X[train],X[test]
    if scale: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
    m=mk(); m.fit(Xtr,y[train]); p=m.predict_proba(Xte)[:,1]
    return float(roc_auc_score(y[test],p)),float(f1_score(y[test],(p>=.5))),float(accuracy_score(y[test],(p>=.5))),p

mkLogit=lambda:LogisticRegression(max_iter=2000)
mkXGB=lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0)

# (1) Ablation
abl={}
for name,feats in [('spatial',SPATIAL),('road',ROAD),('full',FULL)]:
    lr=cv_auc(feats,mkLogit,True); xg=cv_auc(feats,mkXGB)
    abl[name]={'Logit_cv':round(lr[0],3),'Logit_sd':round(lr[1],3),'XGB_cv':round(xg[0],3),'XGB_sd':round(xg[1],3),
               'Logit_test':round(test_auc(feats,mkLogit,True)[0],3),'XGB_test':round(test_auc(feats,mkXGB)[0],3)}
out['ablation']=abl
print('ABLATION',abl)

# (2) Extended baselines on FULL (test 2022)
base={}
for name,mk,sc in [('Logistic',mkLogit,True),('kNN',lambda:KNeighborsClassifier(n_neighbors=15),True),
                   ('MLP',lambda:MLPClassifier(hidden_layer_sizes=(100,50),max_iter=500,random_state=0),True),
                   ('RandomForest',lambda:RandomForestClassifier(n_estimators=400,random_state=0),False),
                   ('XGBoost',mkXGB,False)]:
    a,f,acc,_=test_auc(FULL,mk,sc); base[name]={'AUC':round(a,3),'F1':round(f,3),'Acc':round(acc,3)}
out['baselines_full']=base; print('BASELINES',base)

# (3) Per-province AUC (test 2022), XGB full
Xf=df[FULL].values.astype(float)
mx=mkXGB(); mx.fit(Xf[train],y[train]); pfull=mx.predict_proba(Xf)[:,1]
prov={}
SIDO={11:'Seoul',26:'Busan',27:'Daegu',28:'Incheon',29:'Gwangju',30:'Daejeon',31:'Ulsan',36:'Sejong',
      41:'Gyeonggi',42:'Gangwon',43:'Chungbuk',44:'Chungnam',45:'Jeonbuk',46:'Jeonnam',47:'Gyeongbuk',48:'Gyeongnam',50:'Jeju'}
for s,g in df[test].groupby('sido_cd'):
    ti=g.index.values; yy=y[ti]; pp=pfull[ti]
    if len(ti)>=20 and yy.min()!=yy.max():
        prov[SIDO.get(s,str(s))]={'n':int(len(ti)),'AUC':round(float(roc_auc_score(yy,pp)),3),'pos':int(yy.sum())}
out['per_province']=prov; print('PROVINCE',prov)

# (4a) road-class vs high-risk rate
df['is_high']=y
rc_order=['service','residential','unclassified','tertiary','secondary','primary','trunk','motorway']
present=[c for c in rc_order if c in set(df['road_rank'].map({1:'service',3:'residential',4:'unclassified',5:'tertiary',6:'secondary',7:'primary',8:'trunk',9:'motorway'}).dropna())]
rmap={1:'service',2:'living',3:'residential',4:'unclass',5:'tertiary',6:'secondary',7:'primary',8:'trunk',9:'motorway'}
df['rc']=df['road_rank'].map(rmap).fillna('none')
rates=df.groupby('rc')['is_high'].agg(['mean','count'])
order=[c for c in ['residential','tertiary','secondary','primary','trunk','motorway'] if c in rates.index]
rates=rates.loc[order]
plt.figure(figsize=(5,3))
plt.bar(range(len(rates)),rates['mean'],color='#c0392b')
plt.xticks(range(len(rates)),rates.index,rotation=30,ha='right'); plt.ylabel('High-risk rate')
plt.title('High-risk fraction by road class'); plt.tight_layout(); plt.savefig('paper/fig_roadclass.png',dpi=150); plt.close()
out['roadclass_rates']={k:round(float(v),3) for k,v in rates['mean'].items()}

# (4b) PR curves Logit vs XGB (full, test)
_,_,_,pL=test_auc(FULL,mkLogit,True); _,_,_,pX=test_auc(FULL,mkXGB)
plt.figure(figsize=(5,3))
for p,lab,c in [(pL,'Logistic','#2980b9'),(pX,'XGBoost','#c0392b')]:
    pr,rc_,_=precision_recall_curve(y[test],p); ap=average_precision_score(y[test],p)
    plt.plot(rc_,pr,label=f'{lab} (AP={ap:.2f})',color=c)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title('Precision–Recall (test 2022)')
plt.tight_layout(); plt.savefig('paper/fig_pr.png',dpi=150); plt.close()

# (4c) calibration XGB
plt.figure(figsize=(5,3))
frac,mean=calibration_curve(y[test],pX,n_bins=10)
plt.plot(mean,frac,'o-',color='#c0392b',label='XGBoost'); plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel('Mean predicted'); plt.ylabel('Observed fraction'); plt.legend(); plt.title('Calibration (test 2022)')
plt.tight_layout(); plt.savefig('paper/fig_calib.png',dpi=150); plt.close()

# (4d) SHAP beeswarm (best-effort)
try:
    import shap
    mxx=mkXGB(); mxx.fit(Xf[train],y[train])
    expl=shap.TreeExplainer(mxx); sv=expl.shap_values(Xf[test][:800])
    shap.summary_plot(sv,df[FULL].values[test][:800],feature_names=FULL,show=False,max_display=11)
    plt.tight_layout(); plt.savefig('paper/fig_shap_bee.png',dpi=150,bbox_inches='tight'); plt.close()
    out['shap']='ok'
except Exception as e:
    out['shap']=f'skip: {e}'

json.dump(out,open('expand_results.json','w'),indent=2,ensure_ascii=False)
print('SAVED expand_results.json'); print('DONE_EXPAND')
