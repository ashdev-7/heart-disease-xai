"""Phase 6, A1: variance decomposition, one source at a time (Decision Register D-30).

  python src/phase6_variance.py <dataset> noise|background|report

For each source of variation the other sources are held at their base setting.
Quantity: normalised attribution share |phi_j| / sum_j |phi_j| per explanation.
Per feature, one-way random-effects ANOVA with patients as groups; components are
summed over features. ICC(1,1) = between-patient / (between-patient + within-patient).
Uses the first 100 explained patients for every source.
"""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from common import NATIVE, OUT, SEED, load

warnings.filterwarnings("ignore")
DS, STAGE = sys.argv[1], sys.argv[2]
RES = OUT / DS
N_PAT = 100
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
d = load(DS)
cache = RES / "variance_cache"
cache.mkdir(exist_ok=True)


def cohen_kappa(a, b):
    cats = np.union1d(a, b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)


def make_noiser():
    """Same noise models as src/phase4_base_axes.py (Framingham 'high' level; BRFSS re-interview)."""
    if DS == "framingham":
        spec = dict(sysBP=("cv", 0.086), diaBP=("cv", 0.086), totChol=("cv", 0.0595),
                    glucose=("cv", 0.056), heartRate=("abs", 3.0))

        def f(X, rng):
            X = X.copy()
            for c, (kind, s) in spec.items():
                e = rng.normal(0, s, len(X))
                X[c] = X[c] * (1 + e) if kind == "cv" else X[c] + e
            return X
        return f
    full = pd.concat([d["Xtr"], d["Xte"]])
    days_cat = lambda v: np.digitize(v, [1, 14])
    spec = {"GenHlth": (0.75, "shift", lambda v: v), "PhysHlth": (0.71, "resample", days_cat),
            "MentHlth": (0.67, "resample", days_cat)}

    def perturb(col, v, rate, rng):
        v = v.copy()
        hit = rng.random(len(v)) < rate
        if spec[col][1] == "shift":
            v[hit] = np.clip(v[hit] + rng.choice([-1, 1], hit.sum()), 1, 5)
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
    return lambda X, rng: X.assign(**{c: perturb(c, X[c].to_numpy(), r, rng) for c, r in rates.items()})


if STAGE in ("noise", "background"):
    B = joblib.load(RES / "base_models.joblib")
    prep, models = B["prep"], B["models"]
    Xraw = d["Xex"].iloc[:N_PAT]
    Xex, Xbg, Xtr = prep.transform(Xraw), prep.transform(d["Xbg"]), prep.transform(d["Xtr"])
    out = {}
    if STAGE == "noise":
        noiser, rng = make_noiser(), np.random.default_rng(SEED)
        draws = [prep.transform(noiser(Xraw, rng)) for _ in range(20)]
        for m, e in PAIRS:
            out[f"{m}_{e}"] = np.stack([NATIVE[m][e](models[m], Xn, Xbg) for Xn in draws]).astype(np.float32)
            print("noise", m, e, flush=True)
    else:
        bgs = [Xtr[np.random.default_rng(500 + s).choice(len(Xtr), 100, replace=False)] for s in range(10)]
        for m, e in PAIRS:
            out[f"{m}_{e}"] = np.stack([NATIVE[m][e](models[m], Xex, bg) for bg in bgs]).astype(np.float32)
            print("background", m, e, flush=True)
    np.savez_compressed(cache / f"{STAGE}.npz", **out)


def shares(phi):
    a = np.abs(phi)
    return a / a.sum(-1, keepdims=True)


def icc(S):
    """S: shares [levels, patients, features]. Returns ICC(1,1), between and within variance."""
    k, n, _ = S.shape
    msw = S.var(0, ddof=1).mean(0)                         # per feature: mean within-patient variance
    msb = k * S.mean(0).var(0, ddof=1)                     # per feature
    between = np.clip((msb - msw) / k, 0, None).sum()
    within = msw.sum()
    return between / (between + within), between, within


def icc_ci(S, n_boot=500):
    rng = np.random.default_rng(SEED)
    n = S.shape[1]
    v = [icc(S[:, rng.integers(0, n, n)])[0] for _ in range(n_boot)]
    return np.percentile(v, [2.5, 97.5])


if STAGE == "report":
    base = np.load(RES / "variants" / "base.npz")
    kern = np.load(RES / "kernel.npz")
    boots = [np.load(f) for f in sorted((RES / "variants").glob("boot_*.npz"))]
    seeds = [np.load(f) for f in sorted((RES / "variants").glob("seed_*.npz"))]
    noise, bgd = np.load(cache / "noise.npz"), np.load(cache / "background.npz")
    rows = []

    def add(source, model, explainer, S):
        r, b, w = icc(S)
        lo, hi = icc_ci(S)
        rows.append(dict(source=source, model=model, explainer=explainer, levels=len(S), icc=r,
                         icc_lo=lo, icc_hi=hi, within_share_of_total=w / (b + w), within_sd=np.sqrt(w)))

    for m, e in PAIRS:
        key = f"phi_{m}_{e}"
        add("training-data bootstrap", m, e, shares(np.stack([z[key][:N_PAT] for z in boots])))
        if m != "lr":
            add("seed", m, e, shares(np.stack([z[key][:N_PAT] for z in seeds])))
        add("measurement noise", m, e, shares(noise[f"{m}_{e}"]))
        add("background draw", m, e, shares(bgd[f"{m}_{e}"]))
    for m in ("lr", "rf", "xgb", "mlp"):
        lv = [base[f"phi_{m}_{e}"][:N_PAT] for e in NATIVE[m]] + [kern[m][:N_PAT]]
        add("explainer (native vs KernelSHAP)", m, "+".join(list(NATIVE[m]) + ["kernel"]), shares(np.stack(lv)))
    add("model class (KernelSHAP)", "all four", "kernel_shap",
        shares(np.stack([kern[m][:N_PAT] for m in ("lr", "rf", "xgb", "mlp")])))

    t = pd.DataFrame(rows)
    t.to_csv(RES / "variance_decomposition.csv", index=False)
    pd.set_option("display.width", 250)
    summ = t.groupby("source").agg(icc_min=("icc", "min"), icc_median=("icc", "median"),
                                   icc_max=("icc", "max")).sort_values("icc_median")
    txt = ["=" * 78, f"VARIANCE DECOMPOSITION, ONE SOURCE AT A TIME (D-30)   dataset: {DS}", "=" * 78,
           f"Quantity: normalised attribution share per feature. {N_PAT} patients.",
           "ICC = share of the variance in attribution shares that is due to real differences between",
           "patients; 1 - ICC = share due to the named source. Higher ICC = the source matters less.",
           "Conventional bands (Koo & Li 2016, from rater studies): <0.5 poor, 0.5-0.75 moderate,",
           "0.75-0.9 good, >0.9 excellent. Context only, not validated for explanations.", "",
           "Summary across model / explainer combinations, most disruptive source first:",
           summ.round(3).to_string(), "", "Detail:",
           t.round(3).to_string(index=False), "",
           "within_sd: root of the summed within-patient variance of the share vector, i.e. the typical",
           "size of the shift in a patient's attribution shares caused by that source."]
    (RES / "variance_report.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
