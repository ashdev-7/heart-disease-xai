"""Phases 3-4 (native explainers): tuning, base models, training variants.

  python src/phase3_models.py <dataset> base       tune, fit base models, checks
  python src/phase3_models.py <dataset> variants   seeds + bootstraps (resumable)

Outputs under results/<dataset>/ :
  hyperparameters.json, performance.csv, explainer_checks.txt, base_models.joblib
  variants/<variant>.npz   margins, test AUCs and native attributions per model
"""
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import NATIVE, OUT, SEED, MLP, Prep, load, make_model, sigmoid

warnings.filterwarnings("ignore")

DS, STAGE = sys.argv[1], sys.argv[2]
BIG = DS == "brfss"
N_BOOT = 50 if BIG else 100
N_SEEDS = 10
RES = OUT / DS
(RES / "variants").mkdir(parents=True, exist_ok=True)

GRIDS = {
    "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1, 10, 100)],
    "rf": [dict(n_estimators=300, min_samples_leaf=l, max_depth=d)
           for l in ((50, 100, 200) if BIG else (5, 10, 25)) for d in (6, 12)],
    "xgb": [dict(n_estimators=n, max_depth=d, learning_rate=0.05)
            for d in (2, 3, 4) for n in (100, 300, 600)],
}
BATCH = 1024 if BIG else 64

d = load(DS)
ytr, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()


def log(msg, f=None):
    print(msg, flush=True)
    if f is not None:
        f.write(msg + "\n")
        f.flush()


def cv_logloss(name, hp):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    losses = []
    for tr, va in skf.split(d["Xtr"], ytr):
        prep = Prep().fit(d["Xtr"].iloc[tr])
        m = make_model(name, hp, SEED).fit(prep.transform(d["Xtr"].iloc[tr]), ytr[tr])
        losses.append(log_loss(ytr[va], sigmoid(m.margin(prep.transform(d["Xtr"].iloc[va])))))
    return float(np.mean(losses))


def auc_ci(y, p, n_boot=1000):
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    return np.percentile(b, [2.5, 97.5])


def calibration(y, margin):
    slope = sm.Logit(y, sm.add_constant(margin)).fit(disp=0).params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=margin).fit().params[0]
    return float(slope), float(citl)


def fit_variant(hps, rows, seed):
    """Refit preprocessing and all four models on the given training rows."""
    Xr, yr = d["Xtr"].iloc[rows], ytr[rows]
    prep = Prep().fit(Xr)
    Xs = prep.transform(Xr)
    models = {n: make_model(n, hps[n], seed).fit(Xs, yr) for n in ("lr", "rf", "xgb", "mlp")}
    return prep, models


def explain_all(prep, models, which=("lr", "rf", "xgb", "mlp")):
    Xex, Xbg, Xte = prep.transform(d["Xex"]), prep.transform(d["Xbg"]), prep.transform(d["Xte"])
    out = {}
    for n in which:
        m = models[n]
        out[f"margin_{n}"] = m.margin(Xex).astype(np.float32)
        out[f"auc_{n}"] = np.float32(roc_auc_score(yte, m.margin(Xte)))
        for ex, fn in NATIVE[n].items():
            out[f"phi_{n}_{ex}"] = fn(m, Xex, Xbg).astype(np.float32)
    return out, (Xex, Xbg, Xte)


# ============================================================================ base
if STAGE == "base":
    f = open(RES / "explainer_checks.txt", "w", encoding="utf-8")
    log(f"=== {DS}: tuning (5-fold CV log loss on base training set, n={len(ytr)}) ===", f)
    hps = {}
    for name, grid in GRIDS.items():
        scores = []
        for hp in grid:
            t = time.time()
            s = cv_logloss(name, hp)
            scores.append(s)
            log(f"  {name} {hp}  logloss {s:.5f}  ({time.time() - t:.0f}s)", f)
        hps[name] = grid[int(np.argmin(scores))]
        log(f"  -> {name} chosen {hps[name]}", f)

    # MLP: epoch count from early stopping on a 15% validation split of base training
    tr, va = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=SEED)
    prep = Prep().fit(d["Xtr"].iloc[tr])
    probe = MLP(epochs=100, batch=BATCH, seed=SEED).fit(
        prep.transform(d["Xtr"].iloc[tr]), ytr[tr], prep.transform(d["Xtr"].iloc[va]), ytr[va])
    hps["mlp"] = dict(epochs=int(probe.best_epoch), batch=BATCH)
    log(f"  -> mlp epochs {probe.best_epoch} (early stopping, patience 10)", f)
    (RES / "hyperparameters.json").write_text(json.dumps(hps, indent=2))

    log(f"\n=== {DS}: base models ===", f)
    prep, models = fit_variant(hps, np.arange(len(ytr)), SEED)
    out, (Xex, Xbg, Xte) = explain_all(prep, models)
    rows = []
    for n, m in models.items():
        mg = m.margin(Xte)
        p = sigmoid(mg)
        lo, hi = auc_ci(yte, p)
        slope, citl = calibration(yte, mg)
        rows.append(dict(model=n, auc=roc_auc_score(yte, p), auc_lo=lo, auc_hi=hi,
                         brier=brier_score_loss(yte, p), cal_slope=slope, cal_in_large=citl,
                         mean_pred=p.mean(), prevalence=yte.mean()))
    perf = pd.DataFrame(rows)
    perf.to_csv(RES / "performance.csv", index=False)
    log(perf.round(4).to_string(index=False), f)

    log(f"\n=== {DS}: explainer exactness checks on the base models (max abs error) ===", f)
    mb = {n: models[n].margin(Xbg) for n in models}
    e = np.abs(out["phi_lr_linear_shap"].sum(1) + mb["lr"].mean() - out["margin_lr"]).max()
    log(f"  lr  linear_shap additivity      {e:.2e}", f)
    e = np.abs(out["phi_xgb_tree_shap"].sum(1) + mb["xgb"].mean() - out["margin_xgb"]).max()
    log(f"  xgb tree_shap additivity (margin){e:.2e}", f)
    p_ex = models["rf"].est.predict_proba(Xex)[:, 1]
    p_bg = models["rf"].est.predict_proba(Xbg)[:, 1]
    e = np.abs(out["phi_rf_tree_shap"].sum(1) + p_bg.mean() - p_ex).max()
    log(f"  rf  tree_shap additivity (prob)  {e:.2e}", f)
    base_margin = models["mlp"].margin(Xbg.mean(axis=0, keepdims=True))[0]
    gap = out["margin_mlp"] - base_margin
    e = np.abs(out["phi_mlp_ig"].sum(1) - gap)
    log(f"  mlp IG completeness              max {e.max():.2e}, relative to mean |f(x)-f(b)| {e.max() / np.abs(gap).mean():.2e}", f)
    gap = out["margin_mlp"] - mb["mlp"].mean()
    e = np.abs(out["phi_mlp_eg"].sum(1) - gap)
    log(f"  mlp EG completeness              max {e.max():.2e}, relative to mean |f(x)-E f(b)| {e.max() / np.abs(gap).mean():.2e}", f)

    np.savez_compressed(RES / "variants" / "base.npz", **out)
    joblib.dump(dict(prep=prep, models=models, hps=hps), RES / "base_models.joblib")
    f.close()

# ======================================================================== variants
if STAGE == "variants":
    hps = json.loads((RES / "hyperparameters.json").read_text())
    n = len(ytr)
    plan = [(f"seed_{k:02d}", np.arange(n), k) for k in range(N_SEEDS)]
    plan += [(f"boot_{b:03d}", np.random.default_rng(1000 + b).integers(0, n, n), SEED)
             for b in range(N_BOOT)]
    t0 = time.time()
    for i, (tag, rows, seed) in enumerate(plan):
        path = RES / "variants" / f"{tag}.npz"
        if path.exists():
            continue
        t = time.time()
        prep, models = fit_variant(hps, rows, seed)
        out, _ = explain_all(prep, models)
        np.savez_compressed(path, **out)
        print(f"{DS} {tag} done in {time.time() - t:.0f}s  [{i + 1}/{len(plan)}]  "
              f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"{DS} variants complete")
