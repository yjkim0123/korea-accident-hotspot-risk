"""#7 real road-network negatives: sample non-hotspot points, query OSM, and
keep those that actually sit on a road (highway within 120 m) WITH their road
features. These become genuine on-road negatives for a segment-level hotspot-
vs-not task (an honest upgrade over random pseudo-absence). Resumable CSV."""
import numpy as np, pandas as pd, requests, time, os
from scipy.spatial import cKDTree
URL='https://overpass-api.de/api/interpreter'
HDR={'User-Agent':'kaccident-research/1.0 (academic; yjkim0123@ajou.ac.kr)'}
RANK={'motorway':9,'trunk':8,'primary':7,'secondary':6,'tertiary':5,'unclassified':4,'residential':3,'living_street':2,'service':1}
OUT='data/roadneg_features.csv'
pos=pd.read_csv('data/kaccident_hotspots.csv').dropna(subset=['lat','lon'])
postree=cKDTree(pos[['lat','lon']].values)
rng=np.random.default_rng(1)
# candidate non-hotspot points within province extents, 3-33 km from any hotspot
cands=[]
for sido,g in pos.groupby('sido_cd'):
    la0,la1=g['lat'].min(),g['lat'].max(); lo0,lo1=g['lon'].min(),g['lon'].max()
    need=len(g); tries=0
    while need>0 and tries<need*200:
        tries+=1; la=rng.uniform(la0,la1); lo=rng.uniform(lo0,lo1)
        d,_=postree.query([la,lo])
        if 0.03<=d<=0.30:
            cands.append((round(la,5),round(lo,5),int(sido))); need-=1
cand=pd.DataFrame(cands,columns=['lat','lon','sido_cd']).drop_duplicates(['lat','lon'])
cand['key']=cand['lat'].astype(str)+','+cand['lon'].astype(str)
done=set(pd.read_csv(OUT)['key'].astype(str)) if os.path.exists(OUT) else set()
if not os.path.exists(OUT):
    open(OUT,'w').write('key,sido_cd,road_class,n_roads,max_lanes,maxspeed\n')
print(f'candidates {len(cand)}, done {len(done)}',flush=True)
def to_int(v):
    try: return int(str(v).split(';')[0].split()[0])
    except: return None
f=open(OUT,'a'); n=0
for _,r in cand.iterrows():
    if r['key'] in done: continue
    q=f'[out:json][timeout:25];way(around:120,{r.lat},{r.lon})[highway];out tags;'
    cls=''; nr=0; lanes=[]; spd=[]
    for at in range(3):
        try:
            rr=requests.get(URL,params={'data':q},headers=HDR,timeout=40)
            if rr.status_code==200:
                for e in rr.json().get('elements',[]):
                    t=e.get('tags',{}); hw=t.get('highway','')
                    if hw not in RANK: continue
                    nr+=1
                    if RANK.get(hw,0)>RANK.get(cls,0): cls=hw
                    l=to_int(t.get('lanes'));  lanes.append(l) if l else None
                    s=to_int(t.get('maxspeed')); spd.append(s) if s else None
                break
            elif rr.status_code in (429,504): time.sleep(8*(at+1))
            else: time.sleep(3)
        except Exception: time.sleep(5)
    if nr>0:  # only keep on-road negatives
        f.write(f"{r.key},{int(r.sido_cd)},{cls},{nr},{max(lanes) if lanes else ''},{max(spd) if spd else ''}\n"); f.flush()
    n+=1
    if n%100==0: print(f'{n} queried',flush=True)
    time.sleep(0.8)
f.close(); print('DONE_ROADNEG',flush=True)
