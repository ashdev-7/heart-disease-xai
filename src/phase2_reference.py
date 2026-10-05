"""Phase 2 - reference construction and validation (protocol v0.2, D4, R1, R2).

Run from the repository root:  python src/phase2_reference.py
Writes results/reference_validation_report.txt
       results/reference_importance.csv   (global, per group, per reference)
       results/refA_per_patient.csv       (Framingham, per patient, per group)
       results/group_map.csv
"""
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from common import DATA, OUT, SEED

OUT.mkdir(exist_ok=True)

lines = []


def log(s=""):
    print(s)
    lines.append(s)


# D'Agostino et al. 2008, Circulation 117:743-753. Values from the Framingham
# Heart Study risk-function page (retrieved 2026-10-04).
BMI_MODEL = {
    1: dict(ln_age=3.11296, ln_bmi=0.79277, ln_sbp_untreated=1.85508, ln_sbp_treated=1.92672,
            smoker=0.70953, diabetes=0.53160, s0=0.88431, mean_lp=23.9388),
    0: dict(ln_age=2.72107, ln_bmi=0.51125, ln_sbp_untreated=2.81291, ln_sbp_treated=2.88267,
            smoker=0.61868, diabetes=0.77763, s0=0.94833, mean_lp=26.0145),
}
LIPID_MODEL = {
    1: dict(ln_age=3.06117, ln_tc=1.12370, ln_hdl=-0.93263, ln_sbp_untreated=1.93303,
            ln_sbp_treated=1.99881, smoker=0.65451, diabetes=0.57367),
    0: dict(ln_age=2.32888, ln_tc=1.20904, ln_hdl=-0.70833, ln_sbp_untreated=2.76157,
            ln_sbp_treated=2.82263, smoker=0.52873, diabetes=0.69154),
}
# Yusuf et al. 2004, Lancet 364:937-952, Table 3, fully adjusted (read 2026-10-04).
INTERHEART_OR = {
    "ApoB/ApoA1 top vs lowest quintile": 3.25, "Psychosocial index": 2.67, "Diabetes": 2.37,
    "Smoking current vs never": 2.87, "Smoking current+former vs never": 2.04,
    "Hypertension": 1.91, "Fruit and vegetables daily": 0.70, "Exercise": 0.86,
}

FRAM_GROUPS = {
    "age": "Age", "male": "Sex",
    "sysBP": "Blood pressure", "diaBP": "Blood pressure", "prevalentHyp": "Blood pressure",
    "BPMeds": "Blood pressure",
    "currentSmoker": "Smoking", "cigsPerDay": "Smoking",
    "diabetes": "Glycaemia", "glucose": "Glycaemia",
    "BMI": "Adiposity", "totChol": "Lipids",
    "education": "Unreferenced", "heartRate": "Unreferenced", "prevalentStroke": "Unreferenced",
}
BRFSS_CLASSES = {
    "HighBP": "Referenced", "HighChol": "Referenced", "Smoker": "Referenced",
    "Diabetes": "Referenced", "Fruits": "Referenced", "Veggies": "Referenced",
    "PhysActivity": "Referenced", "MentHlth": "Referenced",
    "Age": "Referenced (Ref-E only)", "Sex": "Referenced (Ref-E only)", "BMI": "Referenced (Ref-E only)",
    "HvyAlcoholConsump": "Unreferenced",
    "GenHlth": "Downstream", "PhysHlth": "Downstream", "DiffWalk": "Downstream",
    "Stroke": "Prior disease",
    "Income": "Access / social", "Education": "Access / social", "AnyHealthcare": "Access / social",
    "NoDocbcCost": "Access / social", "CholCheck": "Access / social",
}

REF_FEATURES = ["age", "male", "BMI", "sysBP", "BPMeds", "currentSmoker", "diabetes"]
REF_GROUP = {"age": "Age", "male": "Sex", "BMI": "Adiposity", "sysBP": "Blood pressure",
             "BPMeds": "Blood pressure", "currentSmoker": "Smoking", "diabetes": "Glycaemia"}


def linear_predictor(X):
    """Centred log relative hazard of the BMI-based equation. X columns = REF_FEATURES."""
    X = np.asarray(X, dtype=float)
    age, male, bmi, sbp, treated, smoker, diab = X.T
    out = np.empty(len(X))
    for sex in (0, 1):
        c = BMI_MODEL[sex]
        m = male == sex
        b_sbp = np.where(treated[m] == 1, c["ln_sbp_treated"], c["ln_sbp_untreated"])
        out[m] = (c["ln_age"] * np.log(age[m]) + c["ln_bmi"] * np.log(bmi[m])
                  + b_sbp * np.log(sbp[m]) + c["smoker"] * smoker[m] + c["diabetes"] * diab[m]
                  - c["mean_lp"])
    return out


def risk_10y(X):
    X = np.asarray(X, dtype=float)
    s0 = np.where(X[:, 1] == 1, BMI_MODEL[1]["s0"], BMI_MODEL[0]["s0"])
    return 1 - s0 ** np.exp(linear_predictor(X))


def baseline_logodds_risk(X):
    """Explained output: logit of predicted 10-year risk (comparable with model margins)."""
    r = np.clip(risk_10y(X), 1e-9, 1 - 1e-9)
    return np.log(r / (1 - r))


def exact_shapley(f, X_explain, X_background):
    """Exact interventional Shapley values by enumerating all coalitions."""
    X_explain = np.asarray(X_explain, dtype=float)
    X_background = np.asarray(X_background, dtype=float)
    n, d = X_explain.shape
    nb = len(X_background)
    subsets = [s for k in range(d + 1) for s in combinations(range(d), k)]
    value = {}
    for s in subsets:
        # v(S): features in S take the patient's value, the rest come from background rows
        tiled = np.repeat(X_background[None, :, :], n, axis=0)        # n x nb x d
        if s:
            tiled[:, :, list(s)] = X_explain[:, None, list(s)]
        value[s] = f(tiled.reshape(-1, d)).reshape(n, nb).mean(axis=1)
    phi = np.zeros((n, d))
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for k in range(d):
            w = factorial(k) * factorial(d - k - 1) / factorial(d)
            for s in combinations(others, k):
                phi[:, j] += w * (value[tuple(sorted(s + (j,)))] - value[s])
    return phi, value[()], value[tuple(range(d))]


def auc_ci(y, p, n_boot=2000):
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    b = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), *np.percentile(b, [2.5, 97.5])


log("=" * 78)
log("PHASE 2 - REFERENCE CONSTRUCTION AND VALIDATION")
log("=" * 78)

fr = pd.read_csv(DATA / "framingham.csv")

# ---------------------------------------------------------------- D4
log("\n--- D4  Does the published equation work on this data? ------------------")
elig = fr[(fr.age.between(30, 74)) & (fr.prevalentStroke == 0)]
log(f"Eligible (age 30-74, no prior stroke): {len(elig)} of {len(fr)}")
cc = elig.dropna(subset=REF_FEATURES)
log(f"Complete on the 7 equation inputs     : {len(cc)} (dropped {len(elig) - len(cc)} with missing BMI or BPMeds)")
log(f"CHD events in this set                : {int(cc.TenYearCHD.sum())} ({100 * cc.TenYearCHD.mean():.1f}%)")

Xref = cc[REF_FEATURES].to_numpy(float)
risk = risk_10y(Xref)
y = cc.TenYearCHD.to_numpy()

auc, lo, hi = auc_ci(y, risk)
log(f"\nAUC of D'Agostino BMI equation vs TenYearCHD: {auc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
for sex, name in ((1, "men"), (0, "women")):
    m = cc.male.to_numpy() == sex
    a, l, h = auc_ci(y[m], risk[m])
    log(f"  {name:<5} n={m.sum():<5} AUC {a:.3f} (95% CI {l:.3f}-{h:.3f})   "
        f"[published C for general CVD: {'0.763' if sex else '0.793'} (lipid model)]")

log(f"\nMean predicted 10-year CVD risk {risk.mean():.3f} vs observed 10-year CHD rate {y.mean():.3f} "
    f"(ratio observed/predicted {y.mean() / risk.mean():.2f})")
log("Calibration by tenth of predicted risk (predicted = all CVD, observed = CHD only):")
dec = pd.qcut(risk, 10, labels=False)
cal = pd.DataFrame({"dec": dec, "pred": risk, "obs": y}).groupby("dec").agg(
    n=("obs", "size"), predicted=("pred", "mean"), observed=("obs", "mean"))
log(cal.round(3).to_string())
mono = kendalltau(cal.predicted, cal.observed)[0]
log(f"Rank agreement between predicted and observed across tenths: Kendall tau {mono:.2f}")

stop2 = auc >= 0.60
log(f"\nSTOP RULE 2 (AUC >= 0.60): {'PASS' if stop2 else 'FAIL'}")

# Refit the same functional form on this data and compare with published values.
log("\nRefit of the same model form on this data (logistic, sexes pooled with sex term):")
Z = pd.DataFrame({
    "ln_age": np.log(cc.age), "ln_bmi": np.log(cc.BMI), "ln_sbp": np.log(cc.sysBP),
    "treated": cc.BPMeds, "smoker": cc.currentSmoker, "diabetes": cc.diabetes, "male": cc.male,
})
fit = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
ci = fit.conf_int()
pub = {"ln_age": (3.11296 + 2.72107) / 2, "ln_bmi": (0.79277 + 0.51125) / 2,
       "ln_sbp": (1.85508 + 2.81291) / 2, "smoker": (0.70953 + 0.61868) / 2,
       "diabetes": (0.53160 + 0.77763) / 2}
cmp_tab = pd.DataFrame({"refit": fit.params, "lo95": ci[0], "hi95": ci[1]}).drop("const")
cmp_tab["published_avg_of_sexes"] = pd.Series(pub)
log(cmp_tab.round(3).to_string())
log("(published values are Cox log hazard ratios for all CVD; the refit is a logistic model for CHD,")
log(" so sizes are not expected to match exactly. Signs and rough ordering are the check.)")

# ---------------------------------------------------------------- R1
log("\n--- R1  References ------------------------------------------------------")
rng = np.random.default_rng(SEED)
bg = Xref[rng.choice(len(Xref), size=200, replace=False)]
phi, v0, vfull = exact_shapley(baseline_logodds_risk, Xref, bg)
err = np.abs(phi.sum(axis=1) + v0 - vfull).max()
log(f"Ref-A: exact Shapley values of the BMI equation (logit of 10-year risk), {len(Xref)} patients,")
log(f"       200 background rows, 128 coalitions. Max additivity error {err:.2e}")

phi_df = pd.DataFrame(phi, columns=REF_FEATURES, index=cc.index)
grp = phi_df.T.groupby(REF_GROUP).sum().T                  # signed sum within group
grp.to_csv(OUT / "refA_per_patient.csv")
refA = grp.abs().mean().sort_values(ascending=False)
refA_share = refA / refA.sum()

sd = {
    "Age": np.log(cc.age).std(), "Blood pressure": np.log(cc.sysBP).std(),
    "Smoking": cc.currentSmoker.std(), "Glycaemia": cc.diabetes.std(),
    "Lipids": np.log(fr.totChol.dropna()).std(),
}
avg = lambda k: (LIPID_MODEL[1][k] + LIPID_MODEL[0][k]) / 2
refB = pd.Series({
    "Age": avg("ln_age") * sd["Age"], "Blood pressure": avg("ln_sbp_untreated") * sd["Blood pressure"],
    "Smoking": avg("smoker") * sd["Smoking"], "Glycaemia": avg("diabetes") * sd["Glycaemia"],
    "Lipids": avg("ln_tc") * sd["Lipids"],
}).sort_values(ascending=False)
refB_share = refB / refB.sum()

refC = pd.Series({
    "Lipids": np.log(INTERHEART_OR["ApoB/ApoA1 top vs lowest quintile"]),
    "Smoking": np.log(INTERHEART_OR["Smoking current vs never"]),
    "Glycaemia": np.log(INTERHEART_OR["Diabetes"]),
    "Blood pressure": np.log(INTERHEART_OR["Hypertension"]),
}).sort_values(ascending=False)

# INTERHEART population attributable risk (Table 3, fully adjusted). An odds ratio is
# the effect in an exposed person; attributable risk also reflects how common the
# exposure is, which is what a population-average importance measures.
refCp = pd.Series({"Lipids": 49.2, "Smoking": 35.7, "Blood pressure": 17.9, "Glycaemia": 9.9})

tab = pd.DataFrame({
    "RefA_mean_abs_shap": refA, "RefA_share": refA_share,
    "RefB_beta_x_SD": refB, "RefB_share": refB_share,
    "RefC_ln_OR": refC, "RefCp_PAR_pct": refCp,
})
order = ["Age", "Sex", "Blood pressure", "Smoking", "Glycaemia", "Adiposity", "Lipids"]
tab = tab.reindex(order)
for c in ("RefA_mean_abs_shap", "RefB_beta_x_SD", "RefC_ln_OR", "RefCp_PAR_pct"):
    tab[c.split("_")[0] + "_rank"] = tab[c].rank(ascending=False)
log("\nFramingham references (blank = group not covered by that reference):")
log(tab.round(3).to_string())
log("\nRef-A: equation inputs are measured values, importance = mean |Shapley| over patients.")
log("Ref-B: lipid-model coefficient x SD of the variable in this data (HDL absent, sexes averaged).")
log("Ref-C: INTERHEART ln(odds ratio) for a yes/no exposure. Ordinal use only; smoking uses")
log("       current vs never (2.87) because Framingham records current smoking.")

# ---------------------------------------------------------------- R2
log("\n--- R2  Agreement among references (ceiling) ---------------------------")
pairs = [("RefA_mean_abs_shap", "RefB_beta_x_SD"), ("RefA_mean_abs_shap", "RefC_ln_OR"),
         ("RefB_beta_x_SD", "RefC_ln_OR"), ("RefA_mean_abs_shap", "RefCp_PAR_pct"),
         ("RefB_beta_x_SD", "RefCp_PAR_pct")]
for a, b in pairs:
    sh = tab[[a, b]].dropna()
    t = kendalltau(sh[a], sh[b])[0] if len(sh) > 2 else np.nan
    log(f"  {a.split('_')[0]} vs {b.split('_')[0]}: {len(sh)} shared groups {list(sh.index)}, Kendall tau {t:.2f}")
sh = tab[["RefA_share", "RefB_share"]].dropna()
a_, b_ = sh.RefA_share / sh.RefA_share.sum(), sh.RefB_share / sh.RefB_share.sum()
cos = float(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_))
log(f"  RefA vs RefB cosine similarity of importance shares on shared groups: {cos:.3f}")
log("  With 3-4 shared groups a rank correlation can only take a few values; read the table, not tau.")

# ---------------------------------------------------------------- BRFSS
log("\n--- Phase 2B  BRFSS reference (secondary) -------------------------------")
refD = pd.Series({
    "HighChol": np.log(3.25), "MentHlth": np.log(2.67), "Diabetes": np.log(2.37),
    "Smoker": np.log(2.04), "HighBP": np.log(1.91),
    "Fruits": abs(np.log(0.70)), "Veggies": abs(np.log(0.70)), "PhysActivity": abs(np.log(0.86)),
}).sort_values(ascending=False)
exact = {"Diabetes", "Smoker", "HighBP", "Fruits", "Veggies"}
par = pd.Series({"HighChol": 49.2, "MentHlth": 32.5, "Diabetes": 9.9, "Smoker": 35.7,
                 "HighBP": 17.9, "Fruits": 13.7, "Veggies": 13.7, "PhysActivity": 12.2})
refD_tab = pd.DataFrame({"abs_ln_OR": refD, "rank_OR": refD.rank(ascending=False),
                         "PAR_pct": par, "rank_PAR": par.rank(ascending=False),
                         "match": ["exact" if i in exact else "proxy" for i in refD.index]})
log("Ref-D (INTERHEART |ln OR| mapped to BRFSS items):")
log(refD_tab.round(3).to_string())
log("Ref-E (D'Agostino ordering on shared items, ordinal only): Age > HighBP > Smoker > Diabetes;")
log("       Sex and BMI present in the equation, HighChol from the lipid model.")

# ---------------------------------------------------------------- save
long = []
for ref, col in (("Ref-A", "RefA_mean_abs_shap"), ("Ref-B", "RefB_beta_x_SD"),
                 ("Ref-C (ln OR)", "RefC_ln_OR"), ("Ref-C (PAR)", "RefCp_PAR_pct")):
    for g, v in tab[col].dropna().items():
        long.append(dict(dataset="framingham", reference=ref, item=g, importance=v))
for g, v in refD.items():
    long.append(dict(dataset="brfss", reference="Ref-D (ln OR)", item=g, importance=v))
for g, v in par.items():
    long.append(dict(dataset="brfss", reference="Ref-D (PAR)", item=g, importance=v))
pd.DataFrame(long).to_csv(OUT / "reference_importance.csv", index=False)

gm = [dict(dataset="framingham", feature=k, group=v) for k, v in FRAM_GROUPS.items()]
gm += [dict(dataset="brfss", feature=k, group=v) for k, v in BRFSS_CLASSES.items()]
pd.DataFrame(gm).to_csv(OUT / "group_map.csv", index=False)

(OUT / "reference_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
