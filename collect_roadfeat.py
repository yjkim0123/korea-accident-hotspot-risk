"""Collect road features (highway class, lanes, maxspeed) for each accident
hotspot from the OpenStreetMap Overpass API. Free, no key. Resumable: writes
incrementally to data/road_features.csv and skips coords already done.

We query the nearest roads within ~120 m of each hotspot and reduce to:
  road_class  : OSM highway tag of the highest-ranked nearby road
  n_roads     : number of distinct nearby highway ways (junction complexity proxy)
  max_lanes   : max 'lanes' tag among nearby roads (NaN if none tagged)
  maxspeed    : max 'maxspeed' among nearby roads (NaN if none tagged)
Honest note: Korea OSM tags road class densely but lanes/maxspeed sparsely,
so max_lanes/maxspeed will be partially missing -- reported as coverage %.
"""
import pandas as pd, requests, time, os, sys

SRC='data/kaccident_hotspots.csv'
OUT='data/road_features.csv'
URL='https://overpass-api.de/api/interpreter'
HDR={'User-Agent':'kaccident-research/1.0 (academic; contact yjkim0123@ajou.ac.kr)'}
R=120  # metres

# highway class rank (higher = more major road)
RANK={'motorway':9,'trunk':8,'primary':7,'secondary':6,'tertiary':5,
      'unclassified':4,'residential':3,'living_street':2,'service':1}

df=pd.read_csv(SRC).dropna(subset=['lat','lon'])
# unique coords (rounded) to avoid re-querying repeated sites across years
df['key']=df['lat'].round(5).astype(str)+','+df['lon'].round(5).astype(str)
uniq=df.drop_duplicates('key')[['key','lat','lon']].reset_index(drop=True)

done=set()
if os.path.exists(OUT):
    done=set(pd.read_csv(OUT)['key'].astype(str))
print(f'unique coords {len(uniq)}, already done {len(done)}',flush=True)

if not done and not os.path.exists(OUT):
    pd.DataFrame(columns=['key','road_class','n_roads','max_lanes','maxspeed']).to_csv(OUT,index=False)

def to_int(v):
    try: return int(str(v).split(';')[0].split()[0])
    except: return None

f=open(OUT,'a')
n_done=0
for i,row in uniq.iterrows():
    if row['key'] in done: continue
    q=f'[out:json][timeout:25];way(around:{R},{row.lat},{row.lon})[highway];out tags;'
    cls=''; nr=0; lanes=[]; spd=[]
    for attempt in range(3):
        try:
            r=requests.get(URL,params={'data':q},headers=HDR,timeout=40)
            if r.status_code==200:
                els=r.json().get('elements',[])
                for e in els:
                    t=e.get('tags',{})
                    hw=t.get('highway','')
                    if hw not in RANK: continue
                    nr+=1
                    if RANK.get(hw,0)>RANK.get(cls,0): cls=hw
                    l=to_int(t.get('lanes'));  lanes.append(l) if l else None
                    s=to_int(t.get('maxspeed')); spd.append(s) if s else None
                break
            elif r.status_code in (429,504):
                time.sleep(8*(attempt+1)); continue
            else:
                time.sleep(3); continue
        except Exception as ex:
            time.sleep(5); continue
    ml=max(lanes) if lanes else ''
    ms=max(spd) if spd else ''
    f.write(f'{row.key},{cls},{nr},{ml},{ms}\n'); f.flush()
    n_done+=1
    if n_done%50==0:
        print(f'{n_done} new done (i={i}/{len(uniq)})',flush=True)
    time.sleep(0.8)  # be polite to the free server
f.close()
print('DONE collect',flush=True)
