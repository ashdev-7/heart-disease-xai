"""Audit 6 - hyperparameter choice as a source of explanation variability.

  python src/audit6_hyperparameters.py <dataset>

For each model class a candidate set of configurations is scored by 5-fold CV log loss
(same folds and criterion as the main tuning). The near-equivalent set is every
configuration whose mean CV loss lies within one standard error of the best (the
one-standard-error rule of Breiman et al. 1984; Hastie, Tibshirani & Friedman 2009,
sec. 7.10), capped at the MAX_SET best. Each near-equivalent configuration is refitted on
the full training set with the base seed and explained for the first 200 patients.

Reported per model and explainer:
  - pairwise top-3 agreement between near-equivalent configurations (same definition as
    the seed and bootstrap figures of the main analysis)
  - ICC(1,1) of attribution shares with configuration as the source (first 100 patients,
    same estimator as src/phase6_variance.py)
For the MLP all eight cells of a 2x2x2 factorial (learning rate, weight decay, dropout)
are also fitted, and the effect of each factor is the mean agreement between the four
pairs of configurations that differ in that factor only.
Writes results/<dataset>/audit6_report.txt and audit6.npz
"""
import itertools
import json
import sys
import time
import warnings

import numpy as np
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import common
from common import MLP, NATIVE, OUT, SEED, Prep, SkWrap, load, sigmoid
from metrics import expected_jaccard, ranks_desc

warnings.filterwarnings("ignore")
DS = sys.argv[1]
BIG = DS == "brfss"
RES = OUT / DS
RES.mkdir(parents=True, exist_ok=True)
N_PAT, N_ICC, MAX_SET = 200, 100, 6
BATCH = 1024 if BIG else 64
T0 = time.time()
R = open(RES / "audit6_report.txt", "w", encoding="utf-8")


def log(s=""):
    print(s, flush=True)
    R.write(s + "\n")
    R.flush()


d = load(DS)
prep = Prep().fit(d["Xtr"])
Xtr, Xte, Xbg = (prep.transform(d[k]) for k in ("Xtr", "Xte", "Xbg"))
Xex = prep.transform(d["Xex"].iloc[:N_PAT])
ytr, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()
F = Xtr.shape[1]
ORIG_NET = common.Net


def net_class(drop):
    class Alt(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(64, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)
    return Alt


def build(m, hp):
    if m == "lr":
        return SkWrap(LogisticRegression(C=hp["C"], max_iter=5000), "lr")
    if m == "rf":
        return SkWrap(RandomForestClassifier(n_estimators=300, min_samples_leaf=hp["leaf"], max_depth=hp["depth"],
                                             max_features="sqrt", n_jobs=-1, random_state=SEED), "rf")
    if m == "xgb":
        return SkWrap(XGBClassifier(n_estimators=hp["trees"], max_depth=hp["depth"], learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                    random_state=SEED, eval_metric="logloss"), "xgb")
    common.Net = net_class(hp["dropout"])
    return MLP(epochs=hp["epochs"], batch=BATCH, seed=SEED, lr=hp["lr"], weight_decay=hp["wd"])


def fit(m, hp, X, y):
    mdl = build(m, hp).fit(X, y)
    common.Net = ORIG_NET
    return mdl


def mlp_epochs(hp):
    """One early-stopping run on a 15% validation split, as in the main analysis."""
    tr, va = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=SEED)
    pr = Prep().fit(d["Xtr"].iloc[tr])
    common.Net = net_class(hp["dropout"])
    probe = MLP(epochs=100, batch=BATCH, seed=SEED, lr=hp["lr"], weight_decay=hp["wd"]).fit(
        pr.transform(d["Xtr"].iloc[tr]), ytr[tr], pr.transform(d["Xtr"].iloc[va]), ytr[va])
    common.Net = ORIG_NET
    return int(probe.best_epoch)


def cv(m, hp):
    ls = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(d["Xtr"], ytr):
        pr = Prep().fit(d["Xtr"].iloc[tr])
        mdl = fit(m, hp, pr.transform(d["Xtr"].iloc[tr]), ytr[tr])
        ls.append(log_loss(ytr[va], sigmoid(mdl.margin(pr.transform(d["Xtr"].iloc[va])))))
    return float(np.mean(ls)), float(np.std(ls, ddof=1) / np.sqrt(len(ls)))


def pair_top3(phi):
    Rk = ranks_desc(phi)
    iu = np.triu_indices(len(Rk), 1)
    M = (Rk <= 3).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return inter / (6 - inter)                                   # patients x pairs


def icc(phi):
    a = np.abs(phi[:, :N_ICC])
    S = a / a.sum(-1, keepdims=True)
    k = S.shape[0]
    msw = S.var(0, ddof=1).mean(0)
    msb = k * S.mean(0).var(0, ddof=1)
    between = np.clip((msb - msw) / k, 0, None).sum()
    return between / (between + msw.sum())


GRIDS = {
    "lr": [dict(C=c) for c in (1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3)],
    "rf": [dict(leaf=l, depth=dp) for l in ((25, 50, 100, 200) if BIG else (2, 5, 10, 25))
           for dp in ((6, 12, 16) if BIG else (6, 12, None))],
    "xgb": [dict(depth=dp, trees=n) for dp in ((3, 4, 5, 6) if BIG else (1, 2, 3, 4))
            for n in ((100, 300, 600, 1000) if BIG else (50, 100, 200, 300, 600))],
    "mlp": [dict(lr=a, wd=w, dropout=p) for a in (5e-4, 1e-3) for w in (5e-3, 1e-4) for p in (0.2, 0.1)],
}

log("=" * 78)
log(f"AUDIT 6 - HYPERPARAMETER CHOICE AS A SOURCE OF VARIATION   dataset: {DS}   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)
log(f"master seed {SEED}; {N_PAT} explained patients; chance level of top-3 agreement {expected_jaccard(F, 3):.3f}")
log("Near-equivalent set: mean 5-fold CV log loss within one standard error of the best; at most "
    f"{MAX_SET} configurations (the best ones) are refitted.")
save = {}
for m in ("lr", "rf", "xgb", "mlp"):
    grid = GRIDS[m]
    if m == "mlp":
        for hp in grid:
            hp["epochs"] = mlp_epochs(hp)
    scored = [(hp,) + cv(m, hp) for hp in grid]
    scored.sort(key=lambda t: t[1])
    best_loss, best_se = scored[0][1], scored[0][2]
    near = [t for t in scored if t[1] <= best_loss + best_se]
    log(f"\n--- {m}: {len(grid)} candidate configurations; best CV loss {best_loss:.5f} (SE {best_se:.5f}); "
        f"{len(near)} within one SE   [{(time.time() - T0) / 60:.0f} min]")
    for hp, l, se in scored:
        log(f"    {'*' if l <= best_loss + best_se else ' '} loss {l:.5f}  {json.dumps(hp)}")
    use = near[:MAX_SET]
    todo = [t[0] for t in use]
    if m == "mlp":                                               # the factorial needs all eight cells
        todo = [t[0] for t in scored]
    fitted = {}
    for hp in todo:
        mdl = fit(m, hp, Xtr, ytr)
        key = json.dumps(hp, sort_keys=True)
        fitted[key] = dict(auc=roc_auc_score(yte, mdl.margin(Xte)),
                           phi={e: fn(mdl, Xex, Xbg).astype(np.float32) for e, fn in NATIVE[m].items()})
        log(f"    fitted {key}: test AUC {fitted[key]['auc']:.4f}   [{(time.time() - T0) / 60:.0f} min]")
    keys = [json.dumps(t[0], sort_keys=True) for t in use]
    aucs = [fitted[k]["auc"] for k in keys]
    log(f"  near-equivalent set used: {len(keys)} configurations; test AUC {min(aucs):.4f} to {max(aucs):.4f}")
    for e in NATIVE[m]:
        phi = np.stack([fitted[k]["phi"][e] for k in keys])
        save[f"{m}_{e}"] = phi
        if len(keys) < 2:
            log(f"  {m} {e}: only one configuration within one SE - hyperparameter choice is not a source here")
            continue
        j = pair_top3(phi)
        log(f"  {m} {e}: pairwise top-3 agreement between near-equivalent configurations {j.mean():.3f} "
            f"(least similar pair {j.mean(0).min():.3f}, most similar {j.mean(0).max():.3f}); "
            f"ICC with configuration as the source {icc(phi):.3f}")
    if m == "mlp":
        log("  2x2x2 factorial (all eight configurations, whether or not within one SE):")
        cells = {(hp["lr"], hp["wd"], hp["dropout"]): json.dumps(hp, sort_keys=True) for hp in todo}
        names = ("learning rate", "weight decay", "dropout")
        for e in NATIVE[m]:
            allphi = np.stack([fitted[k]["phi"][e] for k in cells.values()])
            save[f"mlp_factorial_{e}"] = allphi
            log(f"    {e}: all eight, pairwise top-3 agreement {pair_top3(allphi).mean():.3f}; ICC {icc(allphi):.3f}")
            for f_i, name in enumerate(names):
                vals = []
                for a, b in itertools.combinations(cells, 2):
                    if all((a[i] == b[i]) != (i == f_i) for i in range(3)):
                        vals.append(pair_top3(np.stack([fitted[cells[a]]["phi"][e], fitted[cells[b]]["phi"][e]])).mean())
                log(f"      changing only {name}: mean top-3 agreement {np.mean(vals):.3f} over {len(vals)} pairs "
                    f"(range {min(vals):.3f} to {max(vals):.3f})")
        log("    test AUC of the eight: " + ", ".join(f"{fitted[k]['auc']:.4f}" for k in cells.values()))
        save["mlp_factorial_cells"] = np.array(json.dumps(list(cells.values())))
np.savez_compressed(RES / "audit6.npz", **save)
log(f"\nAudit 6 complete for {DS} in {(time.time() - T0) / 60:.0f} min")
R.close()
