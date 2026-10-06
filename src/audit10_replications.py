"""Audit 10 - Framingham replications under other master seeds, and the environment check.

  python src/audit10_replications.py

Collects the headline quantities from every complete Framingham run:
  main (seed 42, laptop), seed 42 re-run on Kaggle, and seeds 7, 11, 23, 101 on Kaggle.
Writes results/replications/framingham_replications.txt and .csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUT

S = OUT / "second_degree"
RUNS = {
    "42 (laptop, main)": OUT / "framingham",
    "42 (Kaggle)": S / "rep-fram-b" / "x" / "results_s42" / "framingham",
    "7": S / "rep-framingham" / "res" / "framingham",
    "11": S / "rep-fram-a" / "x" / "results_s11" / "framingham",
    "23": S / "rep-fram-a" / "x" / "results_s23" / "framingham",
    "101": S / "rep-fram-b" / "x" / "results_s101" / "framingham",
}
NAT = [("lr", "linear_shap"), ("rf", "tree_shap"), ("xgb", "tree_shap"), ("mlp", "ig"), ("mlp", "eg")]
DEST = OUT / "replications"
DEST.mkdir(exist_ok=True)


def faith(p):
    a = (p / "base_axes_report.txt").read_text(encoding="utf-8")
    blk = a[a.rfind("--- Axis F"):].splitlines()[2:12]
    out = {}
    for line in blk:
        w = line.split()
        if len(w) == 7 and w[0] in ("lr", "rf", "xgb", "mlp") and w[1] != "kernel_shap":
            out[f"{w[0]} {w[1]}"] = float(w[5])
    return out


rows = {}
for name, p in RUNS.items():
    r = {}
    hp = json.loads((p / "hyperparameters.json").read_text())
    r["tuned: RF leaf / XGB depth, trees / MLP epochs"] = (f"{hp['rf']['min_samples_leaf']} / {hp['xgb']['max_depth']}, "
                                                             f"{hp['xgb']['n_estimators']} / {hp['mlp']['epochs']}")
    perf = pd.read_csv(p / "performance.csv").set_index("model")
    r["test AUC, LR"] = f"{perf.loc['lr', 'auc']:.3f}"
    r["test AUC, range of four models"] = f"{perf.auc.min():.3f} to {perf.auc.max():.3f}"
    r["calibration slope, range"] = f"{perf.cal_slope.min():.2f} to {perf.cal_slope.max():.2f}"
    f = faith(p)
    r["faithfulness ratio, range"] = f"{min(f.values()):.2f} to {max(f.values()):.2f}"
    pl = pd.read_csv(p / "plausibility.csv")
    pl = pl[pl.explainer != "kernel_shap"]
    r["cosine with Ref-A, range"] = f"{pl['cos_Ref-A'].min():.3f} to {pl['cos_Ref-A'].max():.3f}"
    r["null mean cosine, range"] = f"{pl['null_mean_Ref-A'].min():.3f} to {pl['null_mean_Ref-A'].max():.3f}"
    mu = pd.read_csv(p / "multiplicity.csv").query("metric == 'jaccard3'").set_index(["source", "model", "explainer"]).per_patient
    for m, e in NAT:
        r[f"bootstrap top-3, {m} {e}"] = f"{mu[('boot', m, e)]:.3f}"
    for m, e in NAT[1:]:
        r[f"seed top-3, {m} {e}"] = f"{mu[('seed', m, e)]:.3f}"
    r["smallest seed minus bootstrap gap"] = f"{min(mu[('seed', m, e)] - mu[('boot', m, e)] for m, e in NAT[1:]):.3f}"
    b = np.load(p / "base_axes_per_patient.npz")
    nz = [b[f"noise_high_{m}_{e}_top3"].mean() for m, e in NAT]
    r["noise top-3, range"] = f"{min(nz):.3f} to {max(nz):.3f}"
    ms = [b[k].mean() for k in b.keys() if k.startswith("model_")]
    es = [b[k].mean() for k in b.keys() if k.startswith("explainer_")]
    r["model swap top-3, range"] = f"{min(ms):.3f} to {max(ms):.3f}"
    r["explainer swap top-3, range"] = f"{min(es):.3f} to {max(es):.3f}"
    r["explainer swap minus model swap (means)"] = f"{np.mean(es) - np.mean(ms):.3f}"
    v = pd.read_csv(p / "variance_decomposition.csv")
    g = lambda s: v[v.source.str.startswith(s)].icc
    r["ICC model class"] = f"{g('model class').iloc[0]:.3f}"
    for s, lab in (("training-data", "bootstrap"), ("measurement", "noise"), ("background", "background"), ("seed", "seed")):
        r[f"ICC {lab}, range"] = f"{g(s).min():.3f} to {g(s).max():.3f}"
    sm = (p / "smote_report.txt").read_text(encoding="utf-8")
    L = [l.split() for l in sm[sm.find("model  auc_none"):].splitlines()[1:5]]
    ratio = [float(l[4]) / float(l[5]) for l in L]
    r["SMOTE over-prediction, range"] = f"{min(ratio):.2f} to {max(ratio):.2f}"
    rows[name] = r
t = pd.DataFrame(rows)
t.to_csv(DEST / "framingham_replications.csv")
pd.set_option("display.width", 300)
pd.set_option("display.max_colwidth", 40)
txt = ["=" * 78, "FRAMINGHAM: MAIN RUN AND REPLICATIONS UNDER OTHER MASTER SEEDS", "=" * 78,
       "The master seed sets the train/test split, the 500 explained patients, the background rows,",
       "the tuning folds and all model seeds. Hyperparameters are re-tuned in every run.", "", t.to_string(), ""]

# environment check: same seed, laptop (torch 2.9.1) vs Kaggle (torch 2.11.0)
a = np.load(RUNS["42 (laptop, main)"] / "variants" / "base.npz")
bfile = RUNS["42 (Kaggle)"] / "variants" / "base.npz"
txt.append("Environment check, master seed 42: laptop against Kaggle")
if bfile.exists():
    b = np.load(bfile)
    for m, e in NAT:
        k = f"phi_{m}_{e}"
        d = np.abs(a[k] - b[k])
        ra, rb = np.argsort(-np.abs(a[k]), 1)[:, :3], np.argsort(-np.abs(b[k]), 1)[:, :3]
        same = np.mean([set(x) == set(y) for x, y in zip(ra, rb)])
        txt.append(f"  {m:<4}{e:<12} largest absolute difference in any attribution {d.max():.2e} "
                   f"(mean |attribution| {np.abs(a[k]).mean():.3f}); same top-3 set for {same:.1%} of patients; "
                   f"test AUC {float(a[f'auc_{m}']):.4f} vs {float(b[f'auc_{m}']):.4f}")
else:
    txt.append("  per-variant arrays were not kept in the Kaggle archive; compare the two seed-42 columns above")
(DEST / "framingham_replications.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
