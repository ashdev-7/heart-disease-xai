"""Phase 1 - data validation (protocol v0.2, checks D1, D2, D3).

Run from the repository root:  python src/phase1_data_validation.py
Writes results/data_validation_report.txt
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_ci(y, p, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi


def plain_lr(X, y, name, must_be_positive, auc_range):
    """D3: unpenalised logistic regression, train-only preprocessing."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    sc = StandardScaler().fit(Xtr)
    Xtr_s = pd.DataFrame(sc.transform(Xtr), columns=X.columns, index=Xtr.index)
    Xte_s = pd.DataFrame(sc.transform(Xte), columns=X.columns, index=Xte.index)

    fit = sm.Logit(ytr, sm.add_constant(Xtr_s)).fit(disp=0, maxiter=200)
    ci = fit.conf_int()
    tab = pd.DataFrame(
        {"coef_per_SD": fit.params, "lo95": ci[0], "hi95": ci[1]}
    ).drop("const")
    tab = tab.reindex(tab.coef_per_SD.abs().sort_values(ascending=False).index)
    log(f"\n  {name}: logistic regression, coefficients per 1 SD (train n={len(Xtr)})")
    log(tab.round(3).to_string())

    p = fit.predict(sm.add_constant(Xte_s))
    auc, lo, hi = auc_ci(yte, p)
    log(f"\n  {name}: test AUC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f}), test n={len(Xte)}")

    ok = True
    for f in must_be_positive:
        r = tab.loc[f]
        if r.hi95 < 0:
            log(f"  FAIL  {f}: coefficient negative with CI excluding zero")
            ok = False
        elif r.lo95 > 0:
            log(f"  pass  {f}: positive, CI excludes zero")
        else:
            log(f"  note  {f}: sign not established (CI includes zero), coef {r.coef_per_SD:+.3f}")
    if not (auc_range[0] <= auc <= auc_range[1]):
        log(f"  FAIL  AUC {auc:.3f} outside expected range {auc_range}")
        ok = False
    else:
        log(f"  pass  AUC within expected range {auc_range}")
    return ok


# ------------------------------------------------------------------ Framingham
log("=" * 78)
log("PHASE 1 - DATA VALIDATION REPORT")
log("=" * 78)

fpath = DATA / "framingham.csv"
fr = pd.read_csv(fpath)
log("\n--- FRAMINGHAM ---------------------------------------------------------")
log(f"D1 file   : {fpath.name}   sha256 {sha(fpath)[:16]}...")
log(f"D1 shape  : {fr.shape[0]} rows x {fr.shape[1]} columns (draft paper: 4,240 x 16)")
log(f"D1 target : TenYearCHD prevalence {fr.TenYearCHD.mean():.4f} "
    f"({int(fr.TenYearCHD.sum())} events; draft paper: 15.2%, 644)")
log(f"D1 cohort : mean age {fr.age.mean():.1f} (range {fr.age.min()}-{fr.age.max()}), "
    f"{100 * (1 - fr.male.mean()):.1f}% women, {100 * fr.currentSmoker.mean():.1f}% current smokers, "
    f"mean sysBP {fr.sysBP.mean():.1f}, mean totChol {fr.totChol.mean():.1f}, mean BMI {fr.BMI.mean():.1f}")
log("           D'Agostino 2008 derivation cohort: 8,491 people, mean age 49, 53% women, ages 30-74")

log("\nD2 missing values:")
log(fr.isna().sum()[fr.isna().sum() > 0].to_string())
log(f"D2 exact duplicate rows: {int(fr.duplicated().sum())}")
log("D2 ranges:")
log(fr.describe().T[["min", "max", "mean"]].round(2).to_string())

plaus = {
    "age": (18, 100), "sysBP": (70, 300), "diaBP": (30, 160), "totChol": (80, 700),
    "BMI": (12, 70), "heartRate": (30, 200), "glucose": (30, 500), "cigsPerDay": (0, 100),
}
bad = {c: int(((fr[c] < lo) | (fr[c] > hi)).sum()) for c, (lo, hi) in plaus.items()}
log(f"D2 values outside plausible physiological limits: {bad}")
incons = int(((fr.currentSmoker == 0) & (fr.cigsPerDay > 0)).sum())
log(f"D2 non-smokers with cigsPerDay > 0: {incons}")
incons2 = int(((fr.currentSmoker == 1) & (fr.cigsPerDay == 0)).sum())
log(f"D2 smokers with cigsPerDay == 0: {incons2}")

log("\nD3 known-result replication")
ok_fr = plain_lr(
    fr.drop(columns="TenYearCHD"), fr.TenYearCHD, "Framingham",
    must_be_positive=["age", "sysBP", "cigsPerDay", "male"],
    auc_range=(0.60, 0.80),
)
log("  (diabetes and glucose are tested jointly in Phase 2; alone each is collinear with the other)")

# ----------------------------------------------------------------------- BRFSS
bpath = DATA / "heart_disease_health_indicators_BRFSS2015.csv"
br = pd.read_csv(bpath)
log("\n--- BRFSS 2015 ---------------------------------------------------------")
log(f"D1 file   : {bpath.name}   sha256 {sha(bpath)[:16]}...")
log(f"D1 shape  : {br.shape[0]} rows x {br.shape[1]} columns (draft paper: 253,680 x 22)")
log(f"D1 target : HeartDiseaseorAttack prevalence {br.HeartDiseaseorAttack.mean():.4f} "
    f"({int(br.HeartDiseaseorAttack.sum())} cases; draft paper: 9.42%, 23,893)")

# Expected coding per the CDC 2015 codebook as recoded in the Kaggle file.
expected = {
    "HeartDiseaseorAttack": {0, 1}, "HighBP": {0, 1}, "HighChol": {0, 1}, "CholCheck": {0, 1},
    "Smoker": {0, 1}, "Stroke": {0, 1}, "Diabetes": {0, 1, 2}, "PhysActivity": {0, 1},
    "Fruits": {0, 1}, "Veggies": {0, 1}, "HvyAlcoholConsump": {0, 1}, "AnyHealthcare": {0, 1},
    "NoDocbcCost": {0, 1}, "GenHlth": set(range(1, 6)), "DiffWalk": {0, 1}, "Sex": {0, 1},
    "Age": set(range(1, 14)), "Education": set(range(1, 7)), "Income": set(range(1, 9)),
}
log("\nD1 coding check against expected category sets:")
all_ok = True
for c, s in expected.items():
    got = set(br[c].unique().astype(int))
    flag = "pass" if got <= s else "FAIL"
    all_ok &= got <= s
    log(f"  {flag}  {c:<22} values {sorted(got)}")
for c, (lo, hi) in {"BMI": (12, 99), "MentHlth": (0, 30), "PhysHlth": (0, 30)}.items():
    flag = "pass" if br[c].between(lo, hi).all() else "FAIL"
    all_ok &= br[c].between(lo, hi).all()
    log(f"  {flag}  {c:<22} range {br[c].min():.0f}-{br[c].max():.0f} (expected {lo}-{hi})")

log(f"\nD2 missing values: {int(br.isna().sum().sum())}")
ndup = int(br.duplicated().sum())
log(f"D2 exact duplicate rows: {ndup} ({100 * ndup / len(br):.2f}%). Kept by default (protocol 3, D2).")
log(f"D2 prevalence among duplicated rows: {br[br.duplicated(keep=False)].HeartDiseaseorAttack.mean():.4f} "
    f"vs non-duplicated {br[~br.duplicated(keep=False)].HeartDiseaseorAttack.mean():.4f}")
log(f"D2 BMI >= 60: {int((br.BMI >= 60).sum())} rows ({100 * (br.BMI >= 60).mean():.2f}%)")

log("\nD3 known-result replication")
ok_br = plain_lr(
    br.drop(columns="HeartDiseaseorAttack"), br.HeartDiseaseorAttack, "BRFSS",
    must_be_positive=["Age", "HighBP", "HighChol", "Smoker", "Diabetes", "Sex"],
    auc_range=(0.75, 0.90),
)

log("\n--- STOP RULE 1 --------------------------------------------------------")
log(f"Framingham: {'PASS' if ok_fr else 'FAIL'}")
log(f"BRFSS     : {'PASS' if ok_br and all_ok else 'FAIL'}")

(OUT / "data_validation_report.txt").write_text("\n".join(lines), encoding="utf-8")
sys.exit(0 if (ok_fr and ok_br and all_ok) else 1)
