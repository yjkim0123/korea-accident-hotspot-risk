"""Second expansion batch (existing data, no fabrication):
 (1) Multi-year rolling evaluation (test 2020, 2021, 2022).
 (2) Top-K precision (decision-focused metric) + curve figure.
 (3) Descriptive statistics (per-year counts/casualties) + figure.
 (4) Severity task across all tabular models.
 (5) Threshold sensitivity (50/75/90th percentile).
 (6) Hyperparameter sensitivity (XGBoost grid).
Outputs expand2_results.json + fig_topk.png, fig_descr.png."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
nn=NearestNeighbors(n_neighbors=9).fit(coords); dist,idx=nn.kneighbors(coords)
df['knn_dist_mean']=dist[:,1:].mean(1); df['knn_dist_min']=dist[:,1].copy()
tree=cKDTree(coords); df['density_5km']=[len(tree.query_ball_point(c,0.05))-1 for c in coords]
FULL=['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km','road_rank','n_roads','lanes','maxspeed_v']
X=df[FULL].values.astype(float)
mkXGB=lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0)
mkLogit=lambda:LogisticRegression(max_iter=2000)
out={}

def label_count(trm,pct=75):
    thr=np.percentile(df.loc[trm,'occrrnc_cnt'],pct)
    return (df['occrrnc_cnt']>=thr).astype(int).values

# (1) Rolling multi-year
roll={}
for ty in [2020,2021,2022]:
    trm=(df['year']<ty).values; tem=(df['year']==ty).values
    y=label_count(trm)
    s=StandardScaler().fit(X[trm]);
    lr=mkLogit().fit(s.transform(X[trm]),y[trm]); al=roc_auc_score(y[tem],lr.predict_proba(s.transform(X[tem]))[:,1])
    xg=mkXGB().fit(X[trm],y[trm]); ax=roc_auc_score(y[tem],xg.predict_proba(X[tem])[:,1])
    roll[str(ty)]={'n_test':int(tem.sum()),'Logit':round(float(al),3),'XGB':round(float(ax),3)}
out['rolling']=roll; print('1 ROLLING',roll)

# (2) Top-K precision (test 2022, XGB full)
trm=(df['year']<2022).values; tem=(df['year']==2022).values; y=label_count(trm)
xg=mkXGB().fit(X[trm],y[trm]); p=xg.predict_proba(X[tem])[:,1]; yte=y[tem]
order=np.argsort(-p); topk={}
for K in [10,20,50,100,150]:
    sel=order[:K]; topk[str(K)]=round(float(yte[sel].mean()),3)
out['topk']=topk; print('2 TOPK',topk)
Ks=np.arange(1,len(yte)+1); prec=np.cumsum(yte[order])/Ks
plt.figure(figsize=(5,3)); plt.plot(Ks,prec,color='#c0392b')
plt.axhline(yte.mean(),ls='--',color='gray',label=f'base rate {yte.mean():.2f}')
plt.xlabel('K (sites inspected)'); plt.ylabel('Precision@K'); plt.legend(); plt.title('Decision-focused precision (test 2022)')
plt.tight_layout(); plt.savefig('paper/fig_topk.png',dpi=150); plt.close()

# (3) Descriptive statistics
desc={}
g=df.groupby('year').agg(sites=('occrrnc_cnt','size'),accidents=('occrrnc_cnt','sum'),
    deaths=('dth_cnt','sum'),serious=('se_cnt','sum'),casualties=('caslt_cnt','sum'))
desc['per_year']={str(k):{kk:int(vv) for kk,vv in v.items()} for k,v in g.to_dict('index').items()}
out['descriptive']=desc; print('3 DESCR',desc['per_year'])
yrs=g.index.astype(str)
fig,ax1=plt.subplots(figsize=(5.2,3))
ax1.bar(yrs,g['accidents'],color='#2980b9',alpha=0.8,label='Accidents')
ax1.set_ylabel('Annual accidents',color='#2980b9'); ax1.set_xlabel('Year')
ax2=ax1.twinx(); ax2.plot(yrs,g['deaths'],'o-',color='#c0392b',label='Deaths')
ax2.set_ylabel('Deaths',color='#c0392b')
plt.title('Hotspot accidents and fatalities by year'); fig.tight_layout()
plt.savefig('paper/fig_descr.png',dpi=150); plt.close()

# (4) Severity task across models
se_thr=np.percentile(df.loc[trm,'se_cnt'],75)
ysev=((df['dth_cnt']>=1)|(df['se_cnt']>=se_thr)).astype(int).values
sev={}
for name,mk,sc in [('Logistic',mkLogit,True),('kNN',lambda:KNeighborsClassifier(15),True),
                   ('MLP',lambda:MLPClassifier((100,50),max_iter=500,random_state=0),True),
                   ('RandomForest',lambda:RandomForestClassifier(400,random_state=0),False),
                   ('XGBoost',mkXGB,False)]:
    Xtr,Xte=X[trm],X[tem]
    if sc: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
    m=mk().fit(Xtr,ysev[trm]); pp=m.predict_proba(Xte)[:,1]
    sev[name]={'AUC':round(float(roc_auc_score(ysev[tem],pp)),3),'F1':round(float(f1_score(ysev[tem],(pp>=.5))),3)}
out['severity_models']=sev; print('4 SEVERITY',sev)

# (5) Threshold sensitivity
thr_sens={}
for pct in [50,75,90]:
    y=label_count(trm,pct)
    s=StandardScaler().fit(X[trm]); al=roc_auc_score(y[tem],mkLogit().fit(s.transform(X[trm]),y[trm]).predict_proba(s.transform(X[tem]))[:,1])
    ax=roc_auc_score(y[tem],mkXGB().fit(X[trm],y[trm]).predict_proba(X[tem])[:,1])
    thr_sens[f'p{pct}']={'pos_rate':round(float(y[tem].mean()),3),'Logit':round(float(al),3),'XGB':round(float(ax),3)}
out['threshold_sensitivity']=thr_sens; print('5 THRESH',thr_sens)

# (6) Hyperparameter sensitivity (XGB grid)
y=label_count(trm); hp={}
for depth in [3,4,6]:
    for ne in [100,300,500]:
        m=XGBClassifier(n_estimators=ne,max_depth=depth,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0).fit(X[trm],y[trm])
        hp[f'd{depth}_n{ne}']=round(float(roc_auc_score(y[tem],m.predict_proba(X[tem])[:,1])),3)
out['hyperparam']=hp; print('6 HP',hp)

json.dump(out,open('expand2_results.json','w'),indent=2)
print('DONE_EXPAND2')
