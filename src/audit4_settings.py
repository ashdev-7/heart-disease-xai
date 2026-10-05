"""Audit 4 - checks on settings that were fixed by choice rather than taken from the literature.

  python src/audit4_settings.py

H1 number of random-forest trees: 300 (used) vs 1,000
H2 XGBoost learning rate and subsampling: used (0.05, 0.8) vs (0.1, 1.0) with trees re-chosen by CV
H3 MLP width: 64-32 (used) vs 128-64
H4 number of seeds: first 5 vs all 10
For H1-H3: test AUC and, per patient, top-3 agreement between the explanation from the
used setting and from the alternative, set beside seed-to-seed agreement for the same model.
Appends to results/audit_report.txt
"""
import json
import warnings

import numpy as np
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import common
from common import MLP, NATIVE, OUT, SEED, Prep, SkWrap, load, sigmoid
from metrics import boot_ci, jaccard_topk, ranks_desc

warnings.filterwarnings("ignore")
R = open(OUT / "audit_report.txt", "a", encoding="utf-8")


def log(s=""):
    print(s, flush=True)
    R.write(s + "\n")


log("\n" + "=" * 78)
log("AUDIT 4 - SETTINGS FIXED BY CHOICE (Framingham base models, 500 patients)")
log("=" * 78)
d = load("framingham")
hps = json.loads((OUT / "framingham" / "hyperparameters.json").read_text())
prep = Prep().fit(d["Xtr"])
Xtr, Xte, Xex, Xbg = (prep.transform(d[k]) for k in ("Xtr", "Xte", "Xex", "Xbg"))
ytr, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()
base = np.load(OUT / "framingham" / "variants" / "base.npz")
mp = np.load(OUT / "framingham" / "multiplicity_per_patient.npz")


def report(label, model, m, e, seed_key):
    phi = NATIVE[m][e](model, Xex, Xbg)
    j = jaccard_topk(ranks_desc(phi), ranks_desc(base[f"phi_{m}_{e}"]), 3)
    lo, hi = boot_ci(j)
    log(f"  {label}: test AUC {roc_auc_score(yte, model.margin(Xte)):.3f} "
        f"(used setting {float(base[f'auc_{m}']):.3f}); top-3 agreement with the used setting "
        f"{j.mean():.3f} (95% CI {lo:.3f}-{hi:.3f}); seed-to-seed agreement for this model {mp[seed_key].mean():.3f}")


log("\nH1 Random forest with 1,000 trees instead of 300 (same leaf size and depth)")
rf = SkWrap(RandomForestClassifier(n_estimators=1000, min_samples_leaf=hps["rf"]["min_samples_leaf"],
                                   max_depth=hps["rf"]["max_depth"], max_features="sqrt", n_jobs=-1,
                                   random_state=SEED), "rf").fit(Xtr, ytr)
report("1,000 trees", rf, "rf", "tree_shap", "seed_rf_tree_shap_jaccard3")

log("\nH2 XGBoost with learning rate 0.1 and no row/column subsampling (depth and trees re-chosen by 5-fold CV)")
best, best_ll = None, 9
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for dp in (2, 3, 4):
    for n in (50, 100, 300):
        ls = []
        for tr, va in skf.split(d["Xtr"], ytr):
            pr = Prep().fit(d["Xtr"].iloc[tr])
            mdl = XGBClassifier(n_estimators=n, max_depth=dp, learning_rate=0.1, tree_method="hist",
                                n_jobs=-1, random_state=SEED, eval_metric="logloss").fit(
                pr.transform(d["Xtr"].iloc[tr]), ytr[tr])
            ls.append(log_loss(ytr[va], mdl.predict_proba(pr.transform(d["Xtr"].iloc[va]))[:, 1]))
        if np.mean(ls) < best_ll:
            best, best_ll = (dp, n), np.mean(ls)
log(f"  chosen: depth {best[0]}, {best[1]} trees")
xgb = SkWrap(XGBClassifier(n_estimators=best[1], max_depth=best[0], learning_rate=0.1, tree_method="hist",
                           n_jobs=-1, random_state=SEED, eval_metric="logloss"), "xgb").fit(Xtr, ytr)
report("learning rate 0.1, no subsampling", xgb, "xgb", "tree_shap", "seed_xgb_tree_shap_jaccard3")

log("\nH3 MLP with hidden layers 128-64 instead of 64-32 (same optimiser settings and epochs)")


class WideNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


orig = common.Net
common.Net = WideNet
wide = MLP(epochs=hps["mlp"]["epochs"], batch=hps["mlp"]["batch"], seed=SEED).fit(Xtr, ytr)
common.Net = orig
report("128-64, Expected Gradients", wide, "mlp", "eg", "seed_mlp_eg_jaccard3")
report("128-64, Integrated Gradients", wide, "mlp", "ig", "seed_mlp_ig_jaccard3")

log("\nH4 Number of seeds: per-patient seed agreement from the first 5 seeds vs all 10")
files = sorted((OUT / "framingham" / "variants").glob("seed_*.npz"))
for m, e in (("rf", "tree_shap"), ("xgb", "tree_shap"), ("mlp", "ig"), ("mlp", "eg")):
    phi = np.stack([np.load(f)[f"phi_{m}_{e}"] for f in files])
    Rk = ranks_desc(phi)

    def pair(Rs):
        iu = np.triu_indices(len(Rs), 1)
        M = (Rs <= 3).astype(np.float32)
        inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
        return (inter / (6 - inter)).mean()

    log(f"  {m} {e}: 5 seeds {pair(Rk[:5]):.3f}, 10 seeds {pair(Rk):.3f}")
R.close()
