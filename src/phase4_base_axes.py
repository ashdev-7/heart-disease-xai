"""Phases 4-5 on the base models: KernelSHAP, explainer noise floor, background
sensitivity, robustness to measurement noise, faithfulness.

  python src/phase4_base_axes.py <dataset> [kernel|floor|noise|faith|all]

Writes under results/<dataset>/ :
  kernel.npz                 KernelSHAP attributions (log-odds) for the 4 base models
  base_axes_report.txt       all tables
  base_axes_per_patient.npz  per-patient scores for the linking analysis
"""
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from common import NATIVE, OUT, SEED, load, sigmoid
from metrics import boot_ci, corrected, expected_jaccard, jaccard_topk, ranks_desc, spearman

warnings.filterwarnings("ignore")
DS = sys.argv[1]
STAGE = sys.argv[2] if len(sys.argv) > 2 else "all"
RES = OUT / DS
d = load(DS)
FEATS = d["features"]
F = len(FEATS)
B = joblib.load(RES / "base_models.joblib")
prep, models = B["prep"], B["models"]
Xex, Xbg = prep.transform(d["Xex"]), prep.transform(d["Xbg"])
CH3, CH5 = expected_jaccard(F, 3), expected_jaccard(F, 5)
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]

report = open(RES / "base_axes_report.txt", "a", encoding="utf-8")
pp_path = RES / "base_axes_per_patient.npz"
pp = dict(np.load(pp_path)) if pp_path.exists() else {}


def log(s=""):
    print(s, flush=True)
    report.write(s + "\n")
    report.flush()


def native(m, e, X, bg=None):
    return NATIVE[m][e](models[m], X, Xbg if bg is None else bg)


def kernel(m, X, seed, bg=None):
    np.random.seed(seed)
    ex = shap.KernelExplainer(models[m].margin, Xbg if bg is None else bg)
    # l1_reg=False: shap's default keeps only 10 features and sets the rest to zero (D-32)
    return np.asarray(ex.shap_values(X, silent=True, l1_reg=False))


def agree_row(label, Ra, Rb):
    j3, j5, sp = jaccard_topk(Ra, Rb, 3), jaccard_topk(Ra, Rb, 5), spearman(Ra, Rb)
    while j3.ndim > 1:                                   # average over repeats, keep patients
        j3, j5, sp = j3.mean(0), j5.mean(0), sp.mean(0)
    lo, hi = boot_ci(j3)
    return dict(item=label, top3=j3.mean(), top3_lo=lo, top3_hi=hi,
                top3_corrected=corrected(j3.mean(), CH3), top5=j5.mean(),
                spearman=sp.mean(), pct_top3_below_half=(j3 < 0.5).mean()), j3


log("\n" + "=" * 78)
log(f"BASE-MODEL AXES   dataset: {DS}   stage: {STAGE}   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)
log(f"Chance level of top-3 / top-5 Jaccard for {F} features: {CH3:.3f} / {CH5:.3f}")

# ------------------------------------------------------------------ KernelSHAP
if STAGE == "kernel1":                                      # one model at a time (memory / time limits)
    m = sys.argv[3]
    np.save(RES / f"kernel_part_{m}.npy", kernel(m, Xex, SEED).astype(np.float32))
    log(f"kernel part saved for {m}")

if STAGE in ("kernel", "all"):
    log("\n--- KernelSHAP on the four base models (log-odds, 100 background rows) ---")
    out = {}
    for m in ("lr", "xgb", "mlp", "rf"):
        t = time.time()
        part = RES / f"kernel_part_{m}.npy"                 # written by the kernel1 stage
        out[m] = np.load(part) if part.exists() else kernel(m, Xex, SEED).astype(np.float32)
        add = np.abs(out[m].sum(1) + models[m].margin(Xbg).mean() - models[m].margin(Xex)).max()
        log(f"  {m:<4} {time.time() - t:6.0f}s   additivity error {add:.1e}")
    np.savez_compressed(RES / "kernel.npz", **out)

    exact = native("lr", "linear_shap", Xex)
    err = np.abs(out["lr"] - exact)
    log("\nD-21 estimator error at the default budget, LR model (exact answer known):")
    log(f"  max |KernelSHAP - exact| {err.max():.2e}; relative to mean |attribution| {err.max() / np.abs(exact).mean():.2e}")
    r, _ = agree_row("lr kernel vs exact", ranks_desc(out["lr"]), ranks_desc(exact))
    log(f"  top-3 Jaccard {r['top3']:.4f}, Spearman {r['spearman']:.4f}")

    log("\nExplainer effect with the model fixed: KernelSHAP vs native explainer, per patient")
    rows = []
    for m, e in PAIRS:
        ref = native(m, e, Xex)
        r, j3 = agree_row(f"{m}: kernel vs {e}", ranks_desc(out[m]), ranks_desc(ref))
        rows.append(r)
        pp[f"explainer_{m}_{e}_top3"] = j3.astype(np.float32)
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("(rf tree_shap explains probability, KernelSHAP explains log-odds: part of that gap is scale.)")

    log("\nModel effect with the explainer fixed: KernelSHAP rankings, model vs model, per patient")
    rows = []
    names = ["lr", "rf", "xgb", "mlp"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r, j3 = agree_row(f"{a} vs {b}", ranks_desc(out[a]), ranks_desc(out[b]))
            rows.append(r)
            pp[f"model_{a}_{b}_top3"] = j3.astype(np.float32)
    log(pd.DataFrame(rows).round(3).to_string(index=False))

# ------------------------------------------------- noise floor and background
if STAGE in ("floor", "all"):
    sub = np.arange(100)
    log("\n--- Explainer noise floor: KernelSHAP re-run 5 times, nothing changed (100 patients) ---")
    rows = []
    for m in ("lr", "xgb", "mlp", "rf"):
        runs = np.stack([kernel(m, Xex[sub], 100 + s) for s in range(5)])
        R = ranks_desc(runs)
        r, _ = agree_row(m, R[1:], R[:1])
        rows.append(r)
    log(pd.DataFrame(rows).round(4).to_string(index=False))
    log("Native explainers (linear SHAP, TreeSHAP, IG, EG) are deterministic: floor = 1 by construction.")

    log("\n--- D-14 background sensitivity: native explainers, 5 fresh background draws per size ---")
    Xtr_s = prep.transform(d["Xtr"])
    rows = []
    for size in (50, 100, 200):
        for m, e in PAIRS:
            if m == "rf" and size == 200:
                continue                                   # cost; reported for 50 and 100
            base = ranks_desc(native(m, e, Xex[sub]))
            alt = np.stack([
                ranks_desc(native(m, e, Xex[sub], Xtr_s[np.random.default_rng(500 + s).choice(
                    len(Xtr_s), size, replace=False)])) for s in range(5)])
            r, _ = agree_row(f"{m} {e}, {size} rows", alt, base[None])
            rows.append(r)
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("Each row: agreement between the explanation with the fixed 100-row background and")
    log("explanations with a different random background of the stated size.")

# ------------------------------------------------------------ measurement noise
NOISE = {
    "framingham": {
        "high": dict(sysBP=("cv", 0.086), diaBP=("cv", 0.086), totChol=("cv", 0.0595),
                     glucose=("cv", 0.056), heartRate=("abs", 3.0)),
        "low": dict(sysBP=("cv", 0.043), diaBP=("cv", 0.043), totChol=("cv", 0.030),
                    glucose=("cv", 0.028), heartRate=("abs", 1.5)),
    }
}
THRESH = {"framingham": 0.20, "brfss": float(d["ytr"].mean())}


def cohen_kappa(a, b):
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)


def brfss_noiser():
    """Flip rates calibrated so that a simulated re-interview reproduces published
    test-retest kappa (Andresen et al. 2003 via Pierannunzi et al. 2013)."""
    full = pd.concat([d["Xtr"], d["Xte"]])
    days_cat = lambda v: np.digitize(v, [1, 14])          # 0 days, 1-13, 14+
    spec = {"GenHlth": (0.75, "shift", lambda v: v), "PhysHlth": (0.71, "resample", days_cat),
            "MentHlth": (0.67, "resample", days_cat)}

    def perturb(col, v, rate, rng):
        v = v.copy()
        hit = rng.random(len(v)) < rate
        if spec[col][1] == "shift":
            step = rng.choice([-1, 1], hit.sum())
            v[hit] = np.clip(v[hit] + step, 1, 5)
        else:
            v[hit] = rng.choice(full[col].to_numpy(), hit.sum())
        return v

    rates = {}
    for col, (kap, _, cat) in spec.items():
        v = full[col].to_numpy()
        lo, hi = 0.0, 1.0
        for _ in range(25):
            mid = (lo + hi) / 2
            k = cohen_kappa(cat(v), cat(perturb(col, v, mid, np.random.default_rng(7))))
            lo, hi = (mid, hi) if k > kap else (lo, mid)
        rates[col] = (lo + hi) / 2
        log(f"  {col:<9} published kappa {kap:.2f} -> calibrated change rate {rates[col]:.3f}")
    return lambda X, rng: X.assign(**{c: perturb(c, X[c].to_numpy(), r, rng) for c, r in rates.items()})


def framingham_noiser(level):
    spec = NOISE["framingham"][level]

    def f(X, rng):
        X = X.copy()
        for c, (kind, s) in spec.items():
            e = rng.normal(0, s, len(X))
            X[c] = X[c] * (1 + e) if kind == "cv" else X[c] + e
        return X
    return f


if STAGE in ("noise", "all"):
    log("\n--- Axis R: robustness to measurement noise (20 draws per patient, native explainers) ---")
    if DS == "framingham":
        noisers = {lv: framingham_noiser(lv) for lv in ("high", "low")}
        log("high = published within-person variation (BP 8.6%, cholesterol 5.95%, glucose 5.6%,")
        log("heart rate 3 bpm); low = half of that. Age, BMI, cigarettes and yes/no items held fixed.")
    else:
        log("Survey re-interview noise on GenHlth, PhysHlth, MentHlth only (other items held fixed):")
        noisers = {"retest": brfss_noiser()}
    thr = THRESH[DS]
    base_R = {(m, e): ranks_desc(native(m, e, Xex)) for m, e in PAIRS}
    base_cls = {m: sigmoid(models[m].margin(Xex)) >= thr for m in models}
    for level, noiser in noisers.items():
        rng = np.random.default_rng(SEED)
        draws = [prep.transform(noiser(d["Xex"], rng)) for _ in range(20)]
        rows = []
        for m, e in PAIRS:
            R = np.stack([ranks_desc(native(m, e, Xn)) for Xn in draws])           # [20, P, F]
            flip = np.stack([(sigmoid(models[m].margin(Xn)) >= thr) != base_cls[m] for Xn in draws])
            j3 = jaccard_topk(R, base_R[(m, e)][None], 3)
            keep = np.where(flip, np.nan, j3)
            r, per = agree_row(f"{m} {e}", R, base_R[(m, e)][None])
            r["top3_same_class"] = np.nanmean(keep)
            r["pct_draws_class_flip"] = flip.mean()
            r["risk_abs_change"] = np.mean([np.abs(sigmoid(models[m].margin(Xn)) -
                                                   sigmoid(models[m].margin(Xex))).mean() for Xn in draws])
            rows.append(r)
            pp[f"noise_{level}_{m}_{e}_top3"] = per.astype(np.float32)
        log(f"\n  noise level: {level}   (class threshold: predicted risk {thr:.3f})")
        log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("top3 = mean top-3 Jaccard between the clean and the re-measured explanation.")
    log("top3_same_class = same, using only draws where the predicted class did not change.")

# ----------------------------------------------------------------- faithfulness
if STAGE in ("faith", "all"):
    log("\n--- Axis F: faithfulness (prediction gap, k = 3, marginal replacement, 20 draws) ---")
    kern = np.load(RES / "kernel.npz") if (RES / "kernel.npz").exists() else {}
    rng = np.random.default_rng(SEED)
    P = len(Xex)
    repl = Xbg[rng.integers(0, len(Xbg), (20, P))]                                 # [20, P, F]
    rand_sets = np.stack([np.stack([rng.choice(F, 3, replace=False) for _ in range(P)]) for _ in range(20)])

    def gap(m, idx):
        """Mean |margin change| when the features in idx [.., P, 3] are replaced."""
        idx = np.broadcast_to(idx, (20, P, 3))
        Xr = np.broadcast_to(Xex, (20, P, F)).copy()
        rows_ = np.arange(P)[None, :, None]
        draws_ = np.arange(20)[:, None, None]
        Xr[draws_, rows_, idx] = repl[draws_, rows_, idx]
        mg = models[m].margin(Xr.reshape(-1, F)).reshape(20, P)
        return np.abs(mg - models[m].margin(Xex)[None]).mean(0)

    rows = []
    for m in ("lr", "rf", "xgb", "mlp"):
        g_rand = gap(m, rand_sets)
        expl = {e: native(m, e, Xex) for e in NATIVE[m]}
        if m in kern:
            expl["kernel_shap"] = kern[m]
        for e, phi in expl.items():
            R = ranks_desc(phi)
            top = np.argsort(R, axis=1)[:, :3]
            bottom = np.argsort(R, axis=1)[:, -3:]
            g_top, g_bot = gap(m, top[None]), gap(m, bottom[None])
            lo, hi = boot_ci(g_top / np.maximum(g_rand, 1e-9))
            rows.append(dict(model=m, explainer=e, gap_top3=g_top.mean(), gap_random3=g_rand.mean(),
                             gap_bottom3=g_bot.mean(), ratio_top_vs_random=g_top.mean() / g_rand.mean(),
                             pct_patients_top_beats_random=(g_top > g_rand).mean()))
            pp[f"faith_{m}_{e}_ratio"] = (g_top / np.maximum(g_rand, 1e-9)).astype(np.float32)
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    log("gap = mean absolute change in log-odds. A faithful explanation has gap_top3 well above")
    log("gap_random3 and gap_bottom3 near zero.")

np.savez_compressed(pp_path, **pp)
report.close()
