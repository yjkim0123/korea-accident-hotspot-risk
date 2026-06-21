"""Attach OSM road features to each accident hotspot (spatial join by rounded
coordinate key, matching collect_roadfeat.py). Produces an augmented dataset
with: road_rank (ordinal road class), n_roads (junction complexity),
lanes, maxspeed (imputed by road class where missing), plus honesty flags
has_lanes/has_maxspeed. Writes data/kaccident_hotspots_road.csv and prints
coverage statistics (so missing-tag rates are disclosed, never hidden)."""
import numpy as np, pandas as pd, json

RANK={'motorway':9,'trunk':8,'primary':7,'secondary':6,'tertiary':5,
      'unclassified':4,'residential':3,'living_street':2,'service':1}

df=pd.read_csv('data/kaccident_hotspots.csv').dropna(subset=['lat','lon']).reset_index(drop=True)
df['key']=df['lat'].round(5).astype(str)+','+df['lon'].round(5).astype(str)
# NOTE: road_features.csv key column itself contains a comma ("lat,lon"), so the
# stored header (5 cols) is misaligned vs each data row (6 fields). Re-parse with
# explicit names, skip the bad header, and rebuild the key from the first two cols.
rf=pd.read_csv('data/road_features.csv',header=None,skiprows=1,
               names=['kl','klon','road_class','n_roads','max_lanes','maxspeed'],
               dtype={'kl':str,'klon':str})
rf['key']=rf['kl'].str.strip()+','+rf['klon'].str.strip()
rf=rf.drop_duplicates('key')

m=df.merge(rf,on='key',how='left')
n=len(m)
matched=m['road_class'].notna().sum()
print(f'hotspots {n} | road match {matched} ({matched/n*100:.1f}%)')

# ordinal road class rank
m['road_rank']=m['road_class'].map(RANK).fillna(0).astype(int)
m['n_roads']=pd.to_numeric(m['n_roads'],errors='coerce').fillna(0).astype(int)

# coverage of raw tags BEFORE imputation (honesty)
lanes_cov=pd.to_numeric(m['max_lanes'],errors='coerce').notna().mean()
spd_cov  =pd.to_numeric(m['maxspeed'], errors='coerce').notna().mean()
print(f'raw tag coverage -> lanes {lanes_cov*100:.1f}% | maxspeed {spd_cov*100:.1f}%')

m['has_lanes']=pd.to_numeric(m['max_lanes'],errors='coerce').notna().astype(int)
m['has_maxspeed']=pd.to_numeric(m['maxspeed'],errors='coerce').notna().astype(int)

# impute lanes/maxspeed by road_class median, then global median
def impute(col):
    s=pd.to_numeric(m[col],errors='coerce')
    med_by=s.groupby(m['road_class']).transform('median')
    s=s.fillna(med_by)
    return s.fillna(s.median())
m['lanes']=impute('max_lanes')
m['maxspeed_v']=impute('maxspeed')

cov={'road_match_pct':round(matched/n*100,1),
     'lanes_cov_pct':round(lanes_cov*100,1),
     'maxspeed_cov_pct':round(spd_cov*100,1),
     'road_class_dist':m['road_class'].value_counts().to_dict()}
json.dump(cov,open('road_coverage.json','w'),ensure_ascii=False,indent=2)
print('road_class distribution:',cov['road_class_dist'])

keep=[c for c in df.columns if c!='key']+['road_rank','n_roads','lanes','maxspeed_v','has_lanes','has_maxspeed']
m[keep].to_csv('data/kaccident_hotspots_road.csv',index=False)
print('wrote data/kaccident_hotspots_road.csv')
