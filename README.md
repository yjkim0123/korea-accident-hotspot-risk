# Machine Learning Prediction of Road-Accident Hotspot Risk in Korea

Reproducible pipeline for the paper *"Machine Learning Prediction of Road-Accident
Hotspot Risk in Korea."* We revisit a withdrawn Korean patent application
(KR 10-2017-0065898), which proposed a binary-logistic accident-probability model,
and ask whether modern machine learning improves it on nationwide open data.

## Data

- **Accident hotspots** — Korea Road Traffic Authority "accident frequent-zone"
  service via the public [data.go.kr](https://www.data.go.kr) API
  (dataset 15057467). 3,956 hotspot-years, 2017–2022. Provided here as
  `data/kaccident_hotspots.csv` (re-collection needs your own free data.go.kr key).
- **Road features** — OpenStreetMap [Overpass API](https://overpass-api.de) (no key),
  road class / lanes / speed within 120 m of each hotspot
  (`data/road_features.csv`, merged into `data/kaccident_hotspots_road.csv`).
- **On-road negatives** — non-hotspot points that lie on an OSM road
  (`data/roadneg_features.csv`).

All data derive from public sources; no personal information is included.

## Pipeline

| Step | Script |
|------|--------|
| Collect road features (OSM) | `collect_roadfeat.py` |
| Spatial join hotspots + roads | `merge_roadfeat.py` |
| Core models (Logit / XGBoost / GraphSAGE) | `model_risk.py` |
| SHAP, severity, risk map | `model_enhanced.py` |
| Pseudo-absence location task | `model_location.py` |
| Road-feature ablation | `model_roadfeat.py` |
| Ablation / baselines / per-province | `model_expand.py` |
| Rolling / top-K / threshold / hyperparams | `model_expand2.py` |
| Robustness (LOPO, persistence, bootstrap) | `model_rigor.py` |
| On-road-negative segment task | `collect_roadneg.py`, `model_roadneg.py` |
| Figures | `make_figs2.py`, `make_pipeline_fig.py` |

## Usage

```bash
pip install -r requirements.txt
# road features are already collected in data/; to re-run modelling:
OMP_NUM_THREADS=1 python3 model_risk.py
OMP_NUM_THREADS=1 python3 model_roadfeat.py
OMP_NUM_THREADS=1 python3 model_rigor.py
```

> **Note:** set `OMP_NUM_THREADS=1` for the GraphSAGE (PyTorch Geometric) runs to
> avoid a thread deadlock.

## Honest evaluation

The standard temporal hold-out AUC (~0.95) is optimistic. Under
leave-one-province-out (LOPO) evaluation it falls to **0.72-0.75**, and a
persistence baseline already reaches **0.88**. We report both the optimistic
in-distribution and the conservative cross-region figures; see `model_rigor.py`.

## Can the spatial gap be closed?

`build_spatial.py` and `build_spatial2.py` re-run LOPO under seven remediation
strategies (fold-safe transferable features, CORAL alignment, importance
weighting, KNN exposure priors, semi-supervised transductive geometry,
ensembling, target-region spatial smoothing). **None beats the naive baseline
(0.72).** Raw outputs are in `results/`.

## Citation

Y. Kim, "Spatial Generalization of Machine-Learned Road-Accident Hotspot Risk:
An Honest Benchmark and the Limits of Domain Adaptation." (under review)

## License

MIT (see `LICENSE`).
