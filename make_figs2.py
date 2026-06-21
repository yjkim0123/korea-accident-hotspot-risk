"""Generate additional figures + fairness analysis (existing data, no fabrication):
 fig_roc_all (ROC all tabular models), fig_confusion (Logit vs XGB),
 fig_shap_dep (SHAP dependence: density & road_rank), fig_corr (feature corr),
 fig_errormap (spatial test errors). Plus urban/rural fairness AUCs."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

df=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon','occrrnc_cnt']).reset_index(drop=True)
df['sido_cd']=df['sido_cd'].astype(int)
coords=df[['lat','lon']].values
nn=NearestNeighbors(n_neighbors=9).fit(coords); dist,idx=nn.kneighbors(coords)
df['knn_dist_mean']=dist[:,1:].mean(1); df['knn_dist_min']=dist[:,1].copy()
tree=cKDTree(coords); df['density_5km']=[len(tree.query_ball_point(c,0.05))-1 for c in coords]
FULL=['lat','lon','sido_cd','year','knn_dist_mean','knn_dist_min','density_5km','road_rank','n_roads','lanes','maxspeed_v']
LBL=['lat','lon','sido','year','knn_mean','knn_min','density','road_rank','n_roads','lanes','maxspeed']
X=df[FULL].values.astype(float)
trm=(df['year']<2022).values; tem=(df['year']==2022).values
thr=np.percentile(df.loc[trm,'occrrnc_cnt'],75); y=(df['occrrnc_cnt']>=thr).astype(int).values
mkXGB=lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,eval_metric='logloss',verbosity=0)
out={}

# ROC all models
models={'Logistic':(LogisticRegression(max_iter=2000),True),'kNN':(KNeighborsClassifier(15),True),
        'MLP':(MLPClassifier((100,50),max_iter=500,random_state=0),True),
        'RandomForest':(RandomForestClassifier(400,random_state=0),False),'XGBoost':(mkXGB(),False)}
plt.figure(figsize=(5,4)); probs={}
for name,(m,sc) in models.items():
    Xtr,Xte=X[trm],X[tem]
    if sc: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
    m.fit(Xtr,y[trm]); p=m.predict_proba(Xte)[:,1]; probs[name]=p
    fpr,tpr,_=roc_curve(y[tem],p); plt.plot(fpr,tpr,label=f'{name} ({roc_auc_score(y[tem],p):.3f})')
plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
plt.legend(fontsize=8); plt.title('ROC curves (test 2022)'); plt.tight_layout(); plt.savefig('paper/fig_roc_all.png',dpi=150); plt.close()

# Confusion matrices Logit vs XGB
fig,axs=plt.subplots(1,2,figsize=(6,3))
for ax,name in zip(axs,['Logistic','XGBoost']):
    cm=confusion_matrix(y[tem],(probs[name]>=0.5).astype(int))
    im=ax.imshow(cm,cmap='Blues'); ax.set_title(name,fontsize=10)
    for (r,c),v in np.ndenumerate(cm): ax.text(c,r,int(v),ha='center',va='center',fontsize=10)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['Low','High']); ax.set_yticklabels(['Low','High'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig('paper/fig_confusion.png',dpi=150); plt.close()

# SHAP dependence
try:
    import shap
    m=mkXGB().fit(X[trm],y[trm]); expl=shap.TreeExplainer(m); sv=expl.shap_values(X[tem])
    fig,axs=plt.subplots(1,2,figsize=(7,3))
    for ax,feat in zip(axs,['density_5km','road_rank']):
        fi=FULL.index(feat); ax.scatter(X[tem][:,fi],sv[:,fi],s=6,alpha=0.4,c='#c0392b')
        ax.axhline(0,color='gray',lw=0.6); ax.set_xlabel(feat); ax.set_ylabel('SHAP value')
    plt.tight_layout(); plt.savefig('paper/fig_shap_dep.png',dpi=150); plt.close(); out['shap_dep']='ok'
except Exception as e: out['shap_dep']=str(e)

# Feature correlation heatmap
corr=np.corrcoef(X.T)
plt.figure(figsize=(5.2,4.6)); im=plt.imshow(corr,cmap='coolwarm',vmin=-1,vmax=1)
plt.colorbar(im,fraction=0.046); plt.xticks(range(len(LBL)),LBL,rotation=90,fontsize=7); plt.yticks(range(len(LBL)),LBL,fontsize=7)
plt.title('Feature correlation'); plt.tight_layout(); plt.savefig('paper/fig_corr.png',dpi=150); plt.close()

# Spatial error map (XGB test)
pX=probs['XGBoost']; pred=(pX>=0.5).astype(int); yt=y[tem]; cc=coords[tem]
plt.figure(figsize=(5,5.2))
correct=(pred==yt); plt.scatter(cc[correct,1],cc[correct,0],s=5,c='#bbbbbb',label='Correct')
fp=(pred==1)&(yt==0); fn=(pred==0)&(yt==1)
plt.scatter(cc[fp,1],cc[fp,0],s=14,c='#2980b9',label='False positive')
plt.scatter(cc[fn,1],cc[fn,0],s=14,c='#c0392b',label='False negative')
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.legend(fontsize=8); plt.title('XGBoost test-2022 errors')
plt.tight_layout(); plt.savefig('paper/fig_errormap.png',dpi=150); plt.close()

# Fairness: metropolitan vs provincial
metro={11,26,27,28,29,30,31,36}
ti=np.where(tem)[0]; ismetro=df.loc[ti,'sido_cd'].isin(metro).values
for grp,mask in [('metropolitan',ismetro),('provincial',~ismetro)]:
    yy=yt[mask]; pp=pX[mask]
    if 0<yy.sum()<len(yy):
        out[f'fair_{grp}']={'n':int(mask.sum()),'pos':int(yy.sum()),'AUC':round(float(roc_auc_score(yy,pp)),3)}
json.dump(out,open('figs2_results.json','w'),indent=2)
print(out); print('DONE_FIGS2')
