"""#7 Segment-level task with REAL on-road negatives (honest upgrade over random
pseudo-absence): positives = accident hotspots, negatives = non-hotspot points
that actually lie on a road, both described by OSM road features + location.
We deliberately EXCLUDE hotspot-derived density features (which would trivially
separate the classes), using only road attributes and coordinates. 5-fold CV."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

RANK={'motorway':9,'trunk':8,'primary':7,'secondary':6,'tertiary':5,'unclassified':4,'residential':3,'living_street':2,'service':1}
# positives: unique hotspot locations
pos=pd.read_csv('data/kaccident_hotspots_road.csv').dropna(subset=['lat','lon'])
pos['k']=pos['lat'].round(5).astype(str)+pos['lon'].round(5).astype(str)
pos=pos.drop_duplicates('k')
P=pd.DataFrame({'lat':pos['lat'],'lon':pos['lon'],'sido_cd':pos['sido_cd'].astype(int),
                'road_rank':pos['road_rank'],'n_roads':pos['n_roads'],
                'lanes':pos['lanes'],'maxspeed_v':pos['maxspeed_v'],'label':1})
# negatives: on-road non-hotspots (key column itself holds "lat,lon" with a comma,
# so the stored header is misaligned; re-parse with explicit names)
neg=pd.read_csv('data/roadneg_features.csv',header=None,skiprows=1,
                names=['lat','lon','sido_cd','road_class','n_roads','max_lanes','maxspeed'])
neg['lat']=neg['lat'].astype(float); neg['lon']=neg['lon'].astype(float)
neg['road_rank']=neg['road_class'].map(RANK).fillna(0).astype(int)
neg['n_roads']=pd.to_numeric(neg['n_roads'],errors='coerce').fillna(0)
N=pd.DataFrame({'lat':neg['lat'],'lon':neg['lon'],'sido_cd':neg['sido_cd'].astype(int),
                'road_rank':neg['road_rank'],'n_roads':neg['n_roads'],
                'lanes':pd.to_numeric(neg['max_lanes'],errors='coerce'),
                'maxspeed_v':pd.to_numeric(neg['maxspeed'],errors='coerce'),'label':0})
df=pd.concat([P,N],ignore_index=True)
for c in ['lanes','maxspeed_v']:
    med=df.groupby('road_rank')[c].transform('median'); df[c]=df[c].fillna(med); df[c]=df[c].fillna(df[c].median())
FEATS=['lat','lon','sido_cd','road_rank','n_roads','lanes','maxspeed_v']
X=df[FEATS].values.astype(float); y=df['label'].values
out={'n_pos':int((y==1).sum()),'n_neg':int((y==0).sum())}
print('pos',out['n_pos'],'neg',out['n_neg'])

def cv(mk,scale=False):
    skf=StratifiedKFold(5,shuffle=True,random_state=0); a=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]
        if scale: s=StandardScaler().fit(Xtr); Xtr,Xte=s.transform(Xtr),s.transform(Xte)
        m=mk(); m.fit(Xtr,y[tr]); a.append(roc_auc_score(y[te],m.predict_proba(Xte)[:,1]))
    return round(float(np.mean(a)),3),round(float(np.std(a)),3)
out['Logit']=cv(lambda:LogisticRegression(max_iter=2000,class_weight='balanced'),True)
out['RandomForest']=cv(lambda:RandomForestClassifier(400,random_state=0,class_weight='balanced'))
out['XGBoost']=cv(lambda:XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,
                  scale_pos_weight=out['n_neg']/out['n_pos'],eval_metric='logloss',verbosity=0))
print(out); json.dump(out,open('roadneg_results.json','w'),indent=2); print('DONE_RN')
