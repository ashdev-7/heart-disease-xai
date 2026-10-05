"""Audit 4 - second-degree checks on settings that were fixed by our own choice.

  python src/audit4_settings.py <dataset>

For each alternative setting: test AUC, and per-patient top-3 agreement between the
explanation from the used setting and from the alternative, set beside the seed-to-seed
agreement of the same model (the yardstick for "no more than a seed changes it").

H1 random-forest size: 1,000 trees instead of 300
H2 XGBoost learning rate 0.1 and no subsampling (depth, trees re-chosen by CV)
H3 MLP width 128-64 instead of 64-32
H3b MLP optimiser: learning rate 0.001, weight decay 0.0001, dropout 0.1 (epochs re-chosen)
H4 number of seeds: first 5 vs all 10
H5 wider search grids: was the chosen setting at the edge of the grid, and does a wider
   grid change the model or its explanations?
H9 (BRFSS) a different banding of the day-count items in the survey-noise model
Appends to results/audit_report.txt (or results/<dataset>/audit4_report.txt on a fresh run).
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import common
from common import MLP, NATIVE, OUT, SEED, Prep, SkWrap, load, sigmoid
from metrics import boot_ci, expected_jaccard, jaccard_topk, ranks_desc

warnings.filterwarnings("ignore")
DS = sys.argv[1]
BIG = DS == "brfss"
RES = OUT / DS
R = open(RES / "audit4_report.txt", "a", encoding="utf-8")
T0 = time.time()


def log(s=""):
    print(s, flush=True)
    R.write(s + "\n")
    R.flush()


d = load(DS)
hps = json.loads((RES / "hyperparameters.json").read_text())
prep = Prep().fit(d["Xtr"])
Xtr, Xte, Xex, Xbg = (prep.transform(d[k]) for k in ("Xtr", "Xte", "Xex", "Xbg"))
ytr, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()
base = np.load(RES / "variants" / "base.npz")
mp = np.load(RES / "multiplicity_per_patient.npz")
F = Xtr.shape[1]

log("\n" + "=" * 78)
log(f"AUDIT 4 - SETTINGS FIXED BY OUR OWN CHOICE   dataset: {DS}   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)
log(f"500 explained patients; chance level of top-3 agreement {expected_jaccard(F, 3):.3f}")


def report(label, model, m, e):
    phi = NATIVE[m][e](model, Xex, Xbg)
    j = jaccard_topk(ranks_desc(phi), ranks_desc(base[f"phi_{m}_{e}"]), 3)
    lo, hi = boot_ci(j)
    key = f"seed_{m}_{e}_jaccard3"
    seed = f"{mp[key].mean():.3f}" if key in mp else "n/a (deterministic)"
    log(f"  {label}: test AUC {roc_auc_score(yte, model.margin(Xte)):.4f} (used {float(base[f'auc_{m}']):.4f}); "
        f"top-3 agreement with the used setting {j.mean():.3f} (95% CI {lo:.3f}-{hi:.3f}); "
        f"seed-to-seed for this model {seed}   [{(time.time() - T0) / 60:.0f} min]")


def cv_loss(make, n_splits=5):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    ls = []
    for tr, va in skf.split(d["Xtr"], ytr):
        pr = Prep().fit(d["Xtr"].iloc[tr])
        mdl = make().fit(pr.transform(d["Xtr"].iloc[tr]), ytr[tr])
        ls.append(log_loss(ytr[va], sigmoid(mdl.margin(pr.transform(d["Xtr"].iloc[va])))))
    return float(np.mean(ls))


def rf(leaf, depth, trees=300):
    return SkWrap(RandomForestClassifier(n_estimators=trees, min_samples_leaf=leaf, max_depth=depth,
                                         max_features="sqrt", n_jobs=-1, random_state=SEED), "rf")


def xgb(depth, trees, lr=0.05, sub=0.8):
    return SkWrap(XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=lr, subsample=sub,
                                colsample_bytree=sub, tree_method="hist", n_jobs=-1, random_state=SEED,
                                eval_metric="logloss"), "xgb")


# ---------------------------------------------------------------- H1
log("\nH1 Random forest with 1,000 trees instead of 300")
report("1,000 trees", rf(hps["rf"]["min_samples_leaf"], hps["rf"]["max_depth"], 1000).fit(Xtr, ytr), "rf", "tree_shap")

# ---------------------------------------------------------------- H2
log("\nH2 XGBoost with learning rate 0.1 and no subsampling (depth and trees re-chosen by 5-fold CV)")
grid = [(dp, n) for dp in (2, 3, 4) for n in (50, 100, 300)]
sc = [cv_loss(lambda dp=dp, n=n: xgb(dp, n, 0.1, 1.0)) for dp, n in grid]
dp, n = grid[int(np.argmin(sc))]
log(f"  chosen: depth {dp}, {n} trees")
report("learning rate 0.1, no subsampling", xgb(dp, n, 0.1, 1.0).fit(Xtr, ytr), "xgb", "tree_shap")

# ---------------------------------------------------------------- H3
log("\nH3 MLP with hidden layers 128-64 instead of 64-32 (same optimiser settings and epochs)")


def net_class(w1, w2, drop):
    class Alt(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, w1), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(w1, w2), nn.ReLU(), nn.Dropout(drop), nn.Linear(w2, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)
    return Alt


orig = common.Net
common.Net = net_class(128, 64, 0.2)
wide = MLP(epochs=hps["mlp"]["epochs"], batch=hps["mlp"]["batch"], seed=SEED).fit(Xtr, ytr)
common.Net = orig
report("128-64, Expected Gradients", wide, "mlp", "eg")
report("128-64, Integrated Gradients", wide, "mlp", "ig")

log("\nH3b MLP with learning rate 0.001, weight decay 0.0001, dropout 0.1 (epochs re-chosen by early stopping)")
common.Net = net_class(64, 32, 0.1)
tr, va = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=SEED)
pr = Prep().fit(d["Xtr"].iloc[tr])
probe = MLP(epochs=100, batch=hps["mlp"]["batch"], seed=SEED, lr=1e-3, weight_decay=1e-4).fit(
    pr.transform(d["Xtr"].iloc[tr]), ytr[tr], pr.transform(d["Xtr"].iloc[va]), ytr[va])
alt = MLP(epochs=int(probe.best_epoch), batch=hps["mlp"]["batch"], seed=SEED, lr=1e-3, weight_decay=1e-4).fit(Xtr, ytr)
common.Net = orig
log(f"  epochs chosen: {probe.best_epoch} (used setting: {hps['mlp']['epochs']})")
report("alternative optimiser, Expected Gradients", alt, "mlp", "eg")
report("alternative optimiser, Integrated Gradients", alt, "mlp", "ig")

# ---------------------------------------------------------------- H4
log("\nH4 Number of seeds: per-patient seed agreement from the first 5 seeds vs all 10")
files = sorted((RES / "variants").glob("seed_*.npz"))
if len(files) >= 10:
    for m, e in (("rf", "tree_shap"), ("xgb", "tree_shap"), ("mlp", "ig"), ("mlp", "eg")):
        Rk = ranks_desc(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in files]))

        def pair(Rs):
            iu = np.triu_indices(len(Rs), 1)
            M = (Rs <= 3).astype(np.float32)
            inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
            return (inter / (6 - inter)).mean()

        log(f"  {m} {e}: 5 seeds {pair(Rk[:5]):.3f}, 10 seeds {pair(Rk):.3f}")
else:
    log("  seed variants not available in this run")

# ---------------------------------------------------------------- H5
log("\nH5 Wider search grids. 'at edge' = the tuned value was the smallest or largest offered.")
log("   CV log loss: used setting vs best in the wider grid; then the wider-grid model is explained.")
# logistic regression
c_used = hps["lr"]["C"]
c_grid = (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4)
lr_make = lambda c: SkWrap(LogisticRegression(C=c, max_iter=5000), "lr")
sc = [cv_loss(lambda c=c: lr_make(c)) for c in c_grid]
c_best = c_grid[int(np.argmin(sc))]
log(f"  LR : used C={c_used} (at edge of original grid: {c_used in (0.001, 100)}); CV loss used "
    f"{sc[c_grid.index(c_used)]:.5f}, best in wider grid {min(sc):.5f} at C={c_best}")
report(f"LR with C={c_best}", lr_make(c_best).fit(Xtr, ytr), "lr", "linear_shap")

# XGBoost
u = hps["xgb"]
x_grid = ([(dp, n) for dp in (1, 2, 3) for n in (25, 50, 100, 200)] if not BIG
          else [(dp, n) for dp in (4, 5, 6) for n in (300, 600, 1000)])
used_loss = cv_loss(lambda: xgb(u["max_depth"], u["n_estimators"]))
sc = [cv_loss(lambda dp=dp, n=n: xgb(dp, n)) for dp, n in x_grid]
bdp, bn = x_grid[int(np.argmin(sc))]
log(f"  XGB: used depth {u['max_depth']}, {u['n_estimators']} trees (depth at edge: {u['max_depth'] in (2, 4)}; "
    f"trees at edge: {u['n_estimators'] in (100, 600)}); CV loss used {used_loss:.5f}, best in wider grid "
    f"{min(sc):.5f} at depth {bdp}, {bn} trees")
report(f"XGB depth {bdp}, {bn} trees", xgb(bdp, bn).fit(Xtr, ytr), "xgb", "tree_shap")

# random forest
u = hps["rf"]
r_grid = ([(l, dp) for l in (1, 2, 5) for dp in (12, 20, None)] if not BIG
          else [(l, dp) for l in (10, 25, 50) for dp in (12, 16)])
used_loss = cv_loss(lambda: rf(u["min_samples_leaf"], u["max_depth"]))
sc = [cv_loss(lambda l=l, dp=dp: rf(l, dp)) for l, dp in r_grid]
bl, bd = r_grid[int(np.argmin(sc))]
log(f"  RF : used leaf {u['min_samples_leaf']}, depth {u['max_depth']} (leaf at edge: "
    f"{u['min_samples_leaf'] in (5, 25, 50, 200)}; depth at edge: True); CV loss used {used_loss:.5f}, "
    f"best in wider grid {min(sc):.5f} at leaf {bl}, depth {bd}")
m_rf = rf(bl, bd).fit(Xtr, ytr)
log(f"  RF leaf {bl}, depth {bd}: test AUC {roc_auc_score(yte, m_rf.margin(Xte)):.4f} "
    f"(used {float(base['auc_rf']):.4f}); mean leaves per tree {np.mean([t.get_n_leaves() for t in m_rf.est.estimators_]):.0f}")
if (bl, bd) != (u["min_samples_leaf"], u["max_depth"]):
    report(f"RF leaf {bl}, depth {bd}", m_rf, "rf", "tree_shap")

# ---------------------------------------------------------------- H9
if BIG:
    log("\nH9 Survey-noise model with a different banding of the day-count items")
    full = pd.concat([d["Xtr"], d["Xte"]])
    B = {"used (0, 1-13, 14+)": [1, 14], "alternative (0, 1-7, 8+)": [1, 8]}
    kap = {"GenHlth": 0.75, "PhysHlth": 0.71, "MentHlth": 0.67}

    def kappa(a, b):
        cats = np.union1d(a, b)
        po = (a == b).mean()
        pe = sum((a == c).mean() * (b == c).mean() for c in cats)
        return (po - pe) / (1 - pe)

    def perturb(col, v, rate, rng):
        v = v.copy()
        hit = rng.random(len(v)) < rate
        if col == "GenHlth":
            v[hit] = np.clip(v[hit] + rng.choice([-1, 1], hit.sum()), 1, 5)
        else:
            v[hit] = rng.choice(full[col].to_numpy(), hit.sum())
        return v

    models = {"lr": SkWrap(LogisticRegression(C=hps["lr"]["C"], max_iter=5000), "lr").fit(Xtr, ytr),
              "xgb": xgb(hps["xgb"]["max_depth"], hps["xgb"]["n_estimators"]).fit(Xtr, ytr)}
    for label, edges in B.items():
        rates = {}
        for col, k in kap.items():
            v = full[col].to_numpy()
            cat = (lambda z: z) if col == "GenHlth" else (lambda z, e=edges: np.digitize(z, e))
            lo, hi = 0.0, 1.0
            for _ in range(25):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if kappa(cat(v), cat(perturb(col, v, mid, np.random.default_rng(7)))) > k else (lo, mid)
            rates[col] = (lo + hi) / 2
        rng = np.random.default_rng(SEED)
        out = {m: [] for m in models}
        for _ in range(20):
            Xn = prep.transform(d["Xex"].assign(**{c: perturb(c, d["Xex"][c].to_numpy(), r, rng) for c, r in rates.items()}))
            for m, mdl in models.items():
                e = "linear_shap" if m == "lr" else "tree_shap"
                out[m].append(jaccard_topk(ranks_desc(NATIVE[m][e](mdl, Xn, Xbg)), ranks_desc(base[f"phi_{m}_{e}"]), 3).mean())
        log(f"  {label}: change rates " + ", ".join(f"{c} {r:.3f}" for c, r in rates.items())
            + "; top-3 robustness " + ", ".join(f"{m} {np.mean(v):.3f}" for m, v in out.items()))
log(f"\nAudit 4 complete for {DS} in {(time.time() - T0) / 60:.0f} min")
R.close()
