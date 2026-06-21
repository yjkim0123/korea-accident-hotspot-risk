"""Enhanced experiments: severity task, 5-fold CV robustness, SHAP, national risk map."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
from xgboost import XGBClassifier
import shap

df = pd.read_csv('data/kaccident_hotspots.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
nn=NearestNeighbors(n_neighbors=9).fit(coords); dist,idx=nn.kneighbors(coords)
df['knn_dist_mean']=dist[:,1:].mean(1); df['knn_dist_min']=dist[:,1]
tree=cKDTree(coords); df['density_5km']=[len(tree.query_ball_point(c,0.05))-1 for c in coords]
FEATS=['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km']
X=df[FEATS].values.astype(float)
out={}

def cv_auc(y, model_fn, scale=False):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); aucs=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]
        if scale:
            sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        m=model_fn(); m.fit(Xtr,y[tr]); aucs.append(roc_auc_score(y[te],m.predict_proba(Xte)[:,1]))
    return float(np.mean(aucs)), float(np.std(aucs))

# ── Task 1: high accident-count (>=75pct) ──
y1=(df['occrrnc_cnt']>=np.percentile(df['occrrnc_cnt'],75)).astype(int).values
# ── Task 2: severity (any death or serious injury heavy) ──
y2=((df['dth_cnt']>0)|(df['se_cnt']>=df['se_cnt'].quantile(.75))).astype(int).values
for name,y in [('count_high',y1),('severity',y2)]:
    lr=cv_auc(y, lambda:LogisticRegression(max_iter=2000), scale=True)
    xg=cv_auc(y, lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0))
    out[name]={'Logit_AUC':f'{lr[0]:.3f}±{lr[1]:.3f}','XGB_AUC':f'{xg[0]:.3f}±{xg[1]:.3f}','pos_rate':round(float(y.mean()),3)}
    print(name, out[name])

# ── SHAP on XGB (count task) ──
xgb=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0).fit(X,y1)
expl=shap.TreeExplainer(xgb); sv=expl.shap_values(X)
imp=np.abs(sv).mean(0); order=np.argsort(imp)[::-1]
out['shap_importance']={FEATS[i]:round(float(imp[i]),4) for i in order}
print('SHAP importance:', out['shap_importance'])
plt.figure(figsize=(5,3)); plt.barh([FEATS[i] for i in order][::-1],[imp[i] for i in order][::-1],color='#3b6ea5')
plt.xlabel('mean |SHAP|'); plt.title('Feature importance (accident-count risk)'); plt.tight_layout(); plt.savefig('fig_shap.png',dpi=200); plt.close()

# ── National risk map ──
plt.figure(figsize=(5.2,6)); sc=plt.scatter(df['lon'],df['lat'],c=df['occrrnc_cnt'],s=10,cmap='YlOrRd',alpha=0.7)
plt.colorbar(sc,label='accident count'); plt.xlabel('Longitude'); plt.ylabel('Latitude')
plt.title('Korea accident hotspots (n=%d, 2017-2022)'%len(df)); plt.gca().set_aspect(1.2); plt.tight_layout(); plt.savefig('fig_riskmap.png',dpi=200); plt.close()
print('figures: fig_shap.png, fig_riskmap.png')
json.dump(out,open('enhanced_results.json','w'),ensure_ascii=False,indent=2)
print('done')
