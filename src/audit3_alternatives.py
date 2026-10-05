"""Audit 3 - re-run the main design alternatives on Framingham and see if conclusions move.

  python src/audit3_alternatives.py <stage>      (each stage is resumable; stops after ~8.5 min)
  python src/audit3_alternatives.py report

Stages
  splits     A1  three other train/test splits (Decision D-01)
  retune     A2  hyperparameters re-tuned inside every bootstrap (D-07)
  impute     A3  regression (iterative) imputation instead of median (D-04)
  pathdep    A4  path-dependent TreeSHAP instead of interventional (D-13)
  subsample  A5  80% subsampling without replacement instead of the bootstrap (D-06)
  kernel     A7  KernelSHAP budget vs exact Shapley values (D-21)
  faith      A8  faithfulness with k = 1, 3, 5 and mean replacement (D-22)
  noisecorr  A11 correlated blood-pressure measurement error (D-23)
All use 30 resamples and the first 200 explained patients unless stated.
"""
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import DATA, NATIVE, OUT, SEED, Prep, load, make_model, sigmoid
from metrics import boot_ci, corrected, expected_jaccard, jaccard_topk, ranks_desc, spearman

warnings.filterwarnings("ignore")
STAGE = sys.argv[1]
import os

T0, BUDGET = time.time(), float(os.environ.get("XAI_BUDGET", 500))
AUD = OUT / "framingham" / "audit"
AUD.mkdir(exist_ok=True)
N_RES, N_PAT = 30, 200
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
HPS = json.loads((OUT / "framingham" / "hyperparameters.json").read_text())
CH = expected_jaccard(15, 3)


def out_of_time():
    if time.time() - T0 > BUDGET:
        print("time budget reached; run the same stage again to resume", flush=True)
        return True
    return False


def pair_top3(phi):
    R = ranks_desc(phi)
    V = len(R)
    iu = np.triu_indices(V, 1)
    M = (R <= 3).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return (inter / (6 - inter)).mean(1)


def fit_explain(Xtr, ytr, Xex, Xbg, hps, seed=SEED, which=("lr", "rf", "xgb", "mlp"), Xte=None, yte=None):
    prep = Prep().fit(Xtr)
    Xs, Xe, Xb = prep.transform(Xtr), prep.transform(Xex), prep.transform(Xbg)
    out = {}
    for m in which:
        mdl = make_model(m, hps[m], seed).fit(Xs, ytr)
        if Xte is not None:
            out[f"auc_{m}"] = np.float32(roc_auc_score(yte, mdl.margin(prep.transform(Xte))))
        for e, fn in NATIVE[m].items():
            out[f"phi_{m}_{e}"] = fn(mdl, Xe, Xb).astype(np.float32)
    return out


def split(seed):
    df = pd.read_csv(DATA / "framingham.csv")
    y, X = df.TenYearCHD.astype(int), df.drop(columns="TenYearCHD").astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    _, Xex, _, _ = train_test_split(Xte, yte, test_size=N_PAT, stratify=yte, random_state=seed)
    bg = Xtr.iloc[np.random.default_rng(seed).choice(len(Xtr), 100, replace=False)]
    return Xtr, ytr.to_numpy(), Xte, yte.to_numpy(), Xex, bg


d = load("framingham")
YTR = d["ytr"].to_numpy()
XEX = d["Xex"].iloc[:N_PAT]

# ----------------------------------------------------------------- A1 splits
if STAGE == "splits":
    for s in (1, 2, 3):
        Xtr, ytr, Xte, yte, Xex, bg = split(s)
        for b in range(-1, N_RES):
            path = AUD / f"split{s}_{'base' if b < 0 else f'boot{b:02d}'}.npz"
            if path.exists():
                continue
            if out_of_time():
                sys.exit(0)
            rows = np.arange(len(ytr)) if b < 0 else np.random.default_rng(1000 + b).integers(0, len(ytr), len(ytr))
            np.savez_compressed(path, **fit_explain(Xtr.iloc[rows], ytr[rows], Xex, bg, HPS, Xte=Xte, yte=yte))
            print(path.name, flush=True)
    print("splits COMPLETE")

# ----------------------------------------------------------------- A2 retune
if STAGE == "retune":
    grids = {
        "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1, 10, 100)],
        "rf": [dict(n_estimators=300, min_samples_leaf=l, max_depth=dp) for l in (5, 10, 25) for dp in (6, 12)],
        "xgb": [dict(n_estimators=k, max_depth=dp, learning_rate=0.05) for dp in (2, 3, 4) for k in (100, 300, 600)],
    }
    for b in range(N_RES):
        path = AUD / f"retune_boot{b:02d}.npz"
        if path.exists():
            continue
        if out_of_time():
            sys.exit(0)
        rows = np.random.default_rng(1000 + b).integers(0, len(YTR), len(YTR))
        Xb, yb = d["Xtr"].iloc[rows], YTR[rows]
        hps = {"mlp": HPS["mlp"]}
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for name, grid in grids.items():
            sc = []
            for hp in grid:
                ls = []
                for tr, va in skf.split(Xb, yb):
                    pr = Prep().fit(Xb.iloc[tr])
                    mdl = make_model(name, hp, SEED).fit(pr.transform(Xb.iloc[tr]), yb[tr])
                    ls.append(log_loss(yb[va], sigmoid(mdl.margin(pr.transform(Xb.iloc[va])))))
                sc.append(np.mean(ls))
            hps[name] = grid[int(np.argmin(sc))]
        out = fit_explain(Xb, yb, XEX, d["Xbg"], hps, which=("lr", "rf", "xgb"))
        out["hp"] = np.array(json.dumps(hps))
        np.savez_compressed(path, **out)
        print(path.name, {k: v for k, v in hps.items() if k != "mlp"}, flush=True)
    print("retune COMPLETE")

# ----------------------------------------------------------------- A2b leak-free re-tuning
if STAGE == "retune_grouped":
    # In a bootstrap sample the same original row appears several times. Ordinary K-fold
    # puts copies of one row in both the training and the validation fold, which rewards
    # over-complex models. Grouped folds keep all copies of a row together.
    from sklearn.model_selection import StratifiedGroupKFold
    which = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("lr", "rf", "xgb")
    grids = {
        "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1, 10, 100)],
        "rf": [dict(n_estimators=300, min_samples_leaf=l, max_depth=dp) for l in (5, 10, 25) for dp in (6, 12)],
        "xgb": [dict(n_estimators=k, max_depth=dp, learning_rate=0.05) for dp in (2, 3, 4) for k in (100, 300, 600)],
    }
    for b in range(N_RES):
        path = AUD / f"retunegrp_{'-'.join(which)}_boot{b:02d}.npz"
        if path.exists():
            continue
        if out_of_time():
            sys.exit(0)
        rows = np.random.default_rng(1000 + b).integers(0, len(YTR), len(YTR))
        Xb, yb = d["Xtr"].iloc[rows], YTR[rows]
        hps = dict(HPS)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        for name in which:
            sc = []
            for hp in grids[name]:
                ls = []
                for tr, va in cv.split(Xb, yb, groups=rows):
                    pr = Prep().fit(Xb.iloc[tr])
                    mdl = make_model(name, hp, SEED).fit(pr.transform(Xb.iloc[tr]), yb[tr])
                    ls.append(log_loss(yb[va], sigmoid(mdl.margin(pr.transform(Xb.iloc[va])))))
                sc.append(np.mean(ls))
            hps[name] = grids[name][int(np.argmin(sc))]
        out = fit_explain(Xb, yb, XEX, d["Xbg"], hps, which=which)
        out["hp"] = np.array(json.dumps({k: hps[k] for k in which}))
        np.savez_compressed(path, **out)
        print(path.name, {k: hps[k] for k in which}, flush=True)
    print("retune_grouped COMPLETE")

# ----------------------------------------------------------------- A2 reference (same environment)
if STAGE == "fixedref":
    for b in range(N_RES):
        path = AUD / f"fixedref_boot{b:02d}.npz"
        if path.exists():
            continue
        if out_of_time():
            sys.exit(0)
        rows = np.random.default_rng(1000 + b).integers(0, len(YTR), len(YTR))
        np.savez_compressed(path, **fit_explain(d["Xtr"].iloc[rows], YTR[rows], XEX, d["Xbg"], HPS,
                                                which=("lr", "rf", "xgb")))
        print(path.name, flush=True)
    print("fixedref COMPLETE")

# ----------------------------------------------------------------- A3 impute
if STAGE == "impute":
    class IterPrep(Prep):
        def fit(self, X):
            self.imp = IterativeImputer(random_state=SEED, max_iter=15, sample_posterior=False).fit(X)
            Xi = pd.DataFrame(self.imp.transform(X), columns=X.columns)
            self.mu, self.sd = Xi.mean(), Xi.std(ddof=0).replace(0, 1.0)
            return self

        def transform(self, X):
            Xi = pd.DataFrame(self.imp.transform(X), columns=X.columns)
            return ((Xi - self.mu) / self.sd).to_numpy(np.float32)

    base = np.load(OUT / "framingham" / "variants" / "base.npz")
    prep = IterPrep().fit(d["Xtr"])
    Xs, Xe, Xb, Xt = (prep.transform(d[k]) for k in ("Xtr", "Xex", "Xbg", "Xte"))
    miss = d["Xex"].isna().any(axis=1).to_numpy()
    rows = []
    for m in ("lr", "rf", "xgb", "mlp"):
        mdl = make_model(m, HPS[m], SEED).fit(Xs, YTR)
        auc = roc_auc_score(d["yte"], mdl.margin(Xt))
        for e, fn in NATIVE[m].items():
            R1, R0 = ranks_desc(fn(mdl, Xe, Xb)), ranks_desc(base[f"phi_{m}_{e}"])
            j = jaccard_topk(R1, R0, 3)
            rows.append(dict(model=m, explainer=e, auc_iterative=auc, top3_all=j.mean(),
                             top3_complete_cases=j[~miss].mean(), top3_imputed_cases=j[miss].mean(),
                             spearman=spearman(R1, R0).mean()))
    pd.DataFrame(rows).to_csv(AUD / "impute.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

# ----------------------------------------------------------------- A4 path-dependent
if STAGE == "pathdep":
    import shap

    def path_dep(model, X):
        sv = np.asarray(shap.TreeExplainer(model.est, feature_perturbation="tree_path_dependent").shap_values(
            X, check_additivity=False))
        return sv[:, :, 1] if sv.ndim == 3 else sv

    for b in range(N_RES):
        path = AUD / f"pathdep_boot{b:02d}.npz"
        if path.exists():
            continue
        if out_of_time():
            sys.exit(0)
        rows = np.random.default_rng(1000 + b).integers(0, len(YTR), len(YTR))
        prep = Prep().fit(d["Xtr"].iloc[rows])
        Xs, Xe, Xb = prep.transform(d["Xtr"].iloc[rows]), prep.transform(XEX), prep.transform(d["Xbg"])
        out = {}
        for m in ("rf", "xgb"):
            mdl = make_model(m, HPS[m], SEED).fit(Xs, YTR[rows])
            out[f"{m}_interventional"] = NATIVE[m]["tree_shap"](mdl, Xe, Xb).astype(np.float32)
            out[f"{m}_pathdep"] = path_dep(mdl, Xe).astype(np.float32)
        np.savez_compressed(path, **out)
        print(path.name, flush=True)
    print("pathdep COMPLETE")

# ----------------------------------------------------------------- A5 subsample
if STAGE == "subsample":
    for b in range(N_RES):
        path = AUD / f"subsample_{b:02d}.npz"
        if path.exists():
            continue
        if out_of_time():
            sys.exit(0)
        rows = np.random.default_rng(3000 + b).choice(len(YTR), int(0.8 * len(YTR)), replace=False)
        np.savez_compressed(path, **fit_explain(d["Xtr"].iloc[rows], YTR[rows], XEX, d["Xbg"], HPS))
        print(path.name, flush=True)
    print("subsample COMPLETE")

# ----------------------------------------------------------------- shared base objects
if STAGE in ("kernel", "faith", "noisecorr"):
    B = joblib.load(OUT / "framingham" / "base_models.joblib")
    prep, models = B["prep"], B["models"]
    Xex, Xbg = prep.transform(d["Xex"]), prep.transform(d["Xbg"])
    F = Xex.shape[1]

# ----------------------------------------------------------------- A7 kernel budget
if STAGE == "kernel":
    import shap
    from math import factorial
    masks = np.arange(2 ** F)
    bits = ((masks[:, None] >> np.arange(F)) & 1).astype(bool)
    size = bits.sum(1)
    w = np.array([factorial(s) * factorial(F - s - 1) / factorial(F) for s in range(F)])

    def exact(f, x):
        v = np.empty(2 ** F)
        for s in range(0, 2 ** F, 2048):
            bb = bits[s:s + 2048]
            Xc = np.where(bb[:, None, :], x[None, None, :], Xbg[None, :, :])
            v[s:s + 2048] = f(Xc.reshape(-1, F).astype(np.float32)).reshape(len(bb), len(Xbg)).mean(1)
        return np.array([(w[size[masks[~bits[:, j]]]] * (v[masks[~bits[:, j]] | (1 << j)] - v[masks[~bits[:, j]]])).sum()
                         for j in range(F)])

    rows = []
    for m in ("lr", "xgb", "mlp"):
        E = np.stack([exact(models[m].margin, Xex[i]) for i in range(20)])
        for ns in (500, 2078, 8000):
            np.random.seed(SEED)
            K = np.asarray(shap.KernelExplainer(models[m].margin, Xbg).shap_values(Xex[:20], nsamples=ns, silent=True))
            rows.append(dict(model=m, coalitions=ns, max_error_pct_of_total=100 * np.abs(K - E).max() / np.abs(E).sum(1).mean(),
                             top3_vs_exact=jaccard_topk(ranks_desc(K), ranks_desc(E), 3).mean(),
                             spearman_vs_exact=spearman(ranks_desc(K), ranks_desc(E)).mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(AUD / "kernel_budget.csv", index=False)

# ----------------------------------------------------------------- A8 faithfulness variants
if STAGE == "faith":
    rng = np.random.default_rng(SEED)
    P = len(Xex)
    repl_bg = Xbg[rng.integers(0, len(Xbg), (20, P))]
    repl_mean = np.broadcast_to(Xbg.mean(0), (20, P, F))
    rows = []
    for k in (1, 3, 5):
        rand = np.stack([np.stack([rng.choice(F, k, replace=False) for _ in range(P)]) for _ in range(20)])
        for label, repl in (("background row", repl_bg), ("background mean", repl_mean)):
            def gap(m, idx):
                idx = np.broadcast_to(idx, (20, P, k))
                Xr = np.broadcast_to(Xex, (20, P, F)).copy()
                a, b_ = np.arange(20)[:, None, None], np.arange(P)[None, :, None]
                Xr[a, b_, idx] = repl[a, b_, idx]
                return np.abs(models[m].margin(Xr.reshape(-1, F)).reshape(20, P) - models[m].margin(Xex)[None]).mean(0)
            for m, e in PAIRS:
                R = ranks_desc(NATIVE[m][e](models[m], Xex, Xbg))
                gt, gr = gap(m, np.argsort(R, axis=1)[:, :k][None]), gap(m, rand)
                rows.append(dict(k=k, replacement=label, model=m, explainer=e, ratio=gt.mean() / gr.mean(),
                                 pct_top_beats_random=(gt > gr).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(AUD / "faith_variants.csv", index=False)
    print(t.pivot_table(index=["model", "explainer"], columns=["replacement", "k"], values="ratio").round(2).to_string())

# ----------------------------------------------------------------- A11 correlated BP noise
if STAGE == "noisecorr":
    base_R = {(m, e): ranks_desc(NATIVE[m][e](models[m], Xex[:N_PAT], Xbg)) for m, e in PAIRS}
    rows = []
    for rho in (0.0, 0.7):
        rng = np.random.default_rng(SEED)
        acc = {p: [] for p in PAIRS}
        for _ in range(20):
            X = XEX.copy()
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], len(X))
            X["sysBP"] *= 1 + 0.086 * z[:, 0]
            X["diaBP"] *= 1 + 0.086 * z[:, 1]
            X["totChol"] *= 1 + rng.normal(0, 0.0595, len(X))
            X["glucose"] *= 1 + rng.normal(0, 0.056, len(X))
            X["heartRate"] += rng.normal(0, 3.0, len(X))
            Xn = prep.transform(X)
            for m, e in PAIRS:
                acc[(m, e)].append(jaccard_topk(ranks_desc(NATIVE[m][e](models[m], Xn, Xbg)), base_R[(m, e)], 3).mean())
        for (m, e), v in acc.items():
            rows.append(dict(bp_error_correlation=rho, model=m, explainer=e, top3=np.mean(v)))
    t = pd.DataFrame(rows)
    t.to_csv(AUD / "noise_corr.csv", index=False)
    print(t.pivot_table(index=["model", "explainer"], columns="bp_error_correlation", values="top3").round(3).to_string())

# ----------------------------------------------------------------- report
if STAGE == "report":
    R = open(OUT / "audit_report.txt", "a", encoding="utf-8")
    pd.set_option("display.width", 250)

    def log(s=""):
        print(s)
        R.write(s + "\n")

    def main_boot(m, e, n=N_RES):
        files = sorted((OUT / "framingham" / "variants").glob("boot_*.npz"))[:n]
        return np.stack([np.load(f)[f"phi_{m}_{e}"][:N_PAT] for f in files])

    log("\n" + "=" * 78)
    log(f"AUDIT 3 - DESIGN ALTERNATIVES RE-RUN (Framingham)   {time.strftime('%Y-%m-%d %H:%M')}")
    log("=" * 78)
    log(f"Per-patient top-3 agreement across 30 resamples, 200 patients (chance {CH:.3f}).")
    ref = {(m, e): pair_top3(main_boot(m, e)) for m, e in PAIRS}

    log("\nA1 Other train/test splits (same hyperparameters). 'main' = the protocol's split, same 30")
    log("   bootstrap seeds. Base-model test AUC in brackets.")
    rows = []
    for m, e in PAIRS:
        r = dict(model=m, explainer=e, main=ref[(m, e)].mean())
        for s in (1, 2, 3):
            fs = sorted(AUD.glob(f"split{s}_boot*.npz"))
            if len(fs) == N_RES:
                r[f"split{s}"] = pair_top3(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in fs])).mean()
                r[f"auc{s}"] = float(np.load(AUD / f"split{s}_base.npz")[f"auc_{m}"])
        rows.append(r)
    log(pd.DataFrame(rows).round(3).to_string(index=False))

    fs = sorted(AUD.glob("retune_boot*.npz"))
    if len(fs) == N_RES:
        log("\nA2 Hyperparameters re-tuned inside every bootstrap vs tuned once (same 30 bootstraps)")
        rows = []
        fr = sorted(AUD.glob("fixedref_boot*.npz"))
        same_env = len(fr) == N_RES
        log("   reference computed in the same environment as the re-tuned runs: " + ("yes" if same_env else "no (main run)"))
        for m, e in [p for p in PAIRS if p[0] != "mlp"]:
            v = pair_top3(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in fs]))
            r0 = pair_top3(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in fr])) if same_env else ref[(m, e)]
            lo, hi = boot_ci(v - r0)
            rows.append(dict(model=m, explainer=e, tuned_once=r0.mean(), retuned=v.mean(),
                             difference=(v - r0).mean(), diff_lo=lo, diff_hi=hi))
        log(pd.DataFrame(rows).round(3).to_string(index=False))
        hp = pd.DataFrame([json.loads(str(np.load(f)["hp"])) for f in fs])
        for m in ("lr", "rf", "xgb"):
            log(f"   {m} choices across bootstraps: " + "; ".join(
                f"{k} x{v}" for k, v in hp[m].astype(str).value_counts().items()))
        log("")
        log("A2b Leak-free re-tuning: grouped folds keep all copies of a bootstrapped row in one fold")
        rows = []
        for tag in sorted({f.name.split("_boot")[0] for f in AUD.glob("retunegrp_*_boot*.npz")}):
            fg = sorted(AUD.glob(f"{tag}_boot*.npz"))
            if len(fg) < N_RES:
                continue
            hpg = pd.DataFrame([json.loads(str(np.load(f)["hp"])) for f in fg])
            for m in tag.replace("retunegrp_", "").split("-"):
                e = list(NATIVE[m])[0]
                v = pair_top3(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in fg]))
                r0 = ref[(m, e)]
                lo, hi = boot_ci(v - r0)
                rows.append(dict(model=m, explainer=e, tuned_once=r0.mean(), retuned_grouped=v.mean(),
                                 difference=(v - r0).mean(), diff_lo=lo, diff_hi=hi,
                                 choices="; ".join(f"{k} x{c}" for k, c in hpg[m].astype(str).value_counts().items())))
        if rows:
            log(pd.DataFrame(rows).round(3).to_string(index=False))
            log("   (tuned_once here is the laptop main run; grouped re-tuning was also run on the laptop)")

    if (AUD / "impute.csv").exists():
        log("\nA3 Regression (iterative) imputation instead of median: agreement of each patient's")
        log("   explanation with the median-imputation explanation (500 patients; 56 have imputed values)")
        log(pd.read_csv(AUD / "impute.csv").round(3).to_string(index=False))

    fs = sorted(AUD.glob("pathdep_boot*.npz"))
    if len(fs) == N_RES:
        log("\nA4 Path-dependent TreeSHAP ('true to the data') instead of interventional")
        rows = []
        for m in ("rf", "xgb"):
            I = np.stack([np.load(f)[f"{m}_interventional"] for f in fs])
            Pd_ = np.stack([np.load(f)[f"{m}_pathdep"] for f in fs])
            rows.append(dict(model=m, bootstrap_top3_interventional=pair_top3(I).mean(),
                             bootstrap_top3_path_dependent=pair_top3(Pd_).mean(),
                             same_model_agreement=jaccard_topk(ranks_desc(I), ranks_desc(Pd_), 3).mean()))
        log(pd.DataFrame(rows).round(3).to_string(index=False))

    fs = sorted(AUD.glob("subsample_*.npz"))
    if len(fs) == N_RES:
        log("\nA5 80% subsampling without replacement instead of the bootstrap")
        log("   (two 80% subsamples share about 80% of their rows; two bootstraps share fewer distinct")
        log("   rows, so subsampling is expected to give higher agreement)")
        rows = [dict(model=m, explainer=e, bootstrap=ref[(m, e)].mean(),
                     subsample_80pct=pair_top3(np.stack([np.load(f)[f"phi_{m}_{e}"] for f in fs])).mean())
                for m, e in PAIRS]
        log(pd.DataFrame(rows).round(3).to_string(index=False))

    if (AUD / "kernel_budget.csv").exists():
        log("\nA7 KernelSHAP budget vs exact Shapley values (20 patients; default budget is 2,078)")
        log(pd.read_csv(AUD / "kernel_budget.csv").round(3).to_string(index=False))
    if (AUD / "faith_variants.csv").exists():
        log("\nA8 Faithfulness ratio (top-k gap / random-k gap) for k = 1, 3, 5 and two replacement rules")
        t = pd.read_csv(AUD / "faith_variants.csv")
        log(t.pivot_table(index=["model", "explainer"], columns=["replacement", "k"], values="ratio").round(2).to_string())
        log(f"   smallest share of patients where top-k beats random-k: {t.pct_top_beats_random.min():.3f}")
    if (AUD / "noise_corr.csv").exists():
        log("\nA11 Blood-pressure errors independent (0.0) vs correlated (0.7, an assumption)")
        t = pd.read_csv(AUD / "noise_corr.csv")
        log(t.pivot_table(index=["model", "explainer"], columns="bp_error_correlation", values="top3").round(3).to_string())
    R.close()
