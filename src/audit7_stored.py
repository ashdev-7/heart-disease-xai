"""Audit 7 - remaining analysis choices, checked from stored explanations (no refits).

  python src/audit7_stored.py

B1 number of noise draws: 10 vs 20 (first 100 patients, both datasets)
B2 resample counts: 2,000 vs 20,000 bootstrap resamples for confidence intervals;
   5,000 vs 50,000 permutations for the plausibility null
B3 plausibility under other groupings of the Framingham predictors and other
   ways of combining attributions within a group
Appends to results/audit_report.txt
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import NATIVE, OUT, SEED, load
from metrics import boot_ci, jaccard_topk, ranks_desc

PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
R = open(OUT / "audit_report.txt", "a", encoding="utf-8")


def log(s=""):
    print(s, flush=True)
    R.write(s + "\n")


def icc(phi):
    a = np.abs(phi)
    S = a / a.sum(-1, keepdims=True)
    k = S.shape[0]
    msw = S.var(0, ddof=1).mean(0)
    msb = k * S.mean(0).var(0, ddof=1)
    between = np.clip((msb - msw) / k, 0, None).sum()
    return between / (between + msw.sum())


log("\n" + "=" * 78)
log(f"AUDIT 7 - REMAINING ANALYSIS CHOICES, FROM STORED EXPLANATIONS   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)

# ---------------------------------------------------------------- B1
log("\nB1 Number of noise draws (first 100 patients; top-3 agreement clean vs re-measured, and ICC)")
for ds in ("framingham", "brfss"):
    base = np.load(OUT / ds / "variants" / "base.npz")
    noise = np.load(OUT / ds / "variance_cache" / "noise.npz")
    for m, e in PAIRS:
        N = noise[f"{m}_{e}"]                                   # draws x 100 x features
        R0 = ranks_desc(base[f"phi_{m}_{e}"][:100])
        per_draw = np.array([jaccard_topk(ranks_desc(x), R0, 3).mean() for x in N])
        log(f"  {ds:<11}{m:<4}{e:<12} top-3: first 10 draws {per_draw[:10].mean():.3f}, last 10 {per_draw[10:].mean():.3f}, "
            f"all 20 {per_draw.mean():.3f} (SD across draws {per_draw.std(ddof=1):.3f}); "
            f"ICC: 10 draws {icc(N[:10]):.3f}, 20 draws {icc(N):.3f}")

# ---------------------------------------------------------------- B2
log("\nB2 Resample counts")
mp = np.load(OUT / "framingham" / "multiplicity_per_patient.npz")
for key in ("boot_rf_tree_shap_jaccard3", "boot_lr_linear_shap_jaccard3", "seed_mlp_ig_jaccard3"):
    a, b = boot_ci(mp[key], 2000), boot_ci(mp[key], 20000)
    log(f"  95% CI of {key}: 2,000 resamples {a[0]:.4f}-{a[1]:.4f}; 20,000 resamples {b[0]:.4f}-{b[1]:.4f}")

d = load("framingham")
FEATS = d["features"]
ref = pd.read_csv(OUT / "reference_importance.csv").query("dataset == 'framingham'")
REFA = ref[ref.reference == "Ref-A"].set_index("item").importance
base = np.load(OUT / "framingham" / "variants" / "base.npz")
GROUPS = ["Age", "Sex", "Blood pressure", "Smoking", "Glycaemia", "Adiposity", "Lipids", "Unreferenced"]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def null(v, r, n):
    rng = np.random.default_rng(SEED)
    return np.array([cosine(rng.permutation(v), r) for _ in range(n)])


def group_matrix(mapping):
    return np.array([[mapping[f] == g for f in FEATS] for g in GROUPS], dtype=float)


MAIN = pd.read_csv(OUT / "group_map.csv").query("dataset == 'framingham'").set_index("feature").group.to_dict()
idx = [GROUPS.index(g) for g in REFA.index]
r = REFA.to_numpy()
Gm = group_matrix(MAIN)
for m, e in (("lr", "linear_shap"), ("rf", "tree_shap")):
    v = np.abs(base[f"phi_{m}_{e}"] @ Gm.T).mean(0)[idx]
    c = cosine(v, r)
    n1, n2 = null(v, r, 5000), null(v, r, 50000)
    log(f"  permutation null, {m} {e}: 5,000 permutations mean {n1.mean():.4f}, share below the observed {np.mean(n1 < c):.4f}; "
        f"50,000 permutations mean {n2.mean():.4f}, share below {np.mean(n2 < c):.4f}")
log("  (with six groups there are only 720 distinct assignments, so the null is exhausted either way)")

# ---------------------------------------------------------------- B3
log("\nB3 Plausibility against the risk-equation reference (Ref-A) under other grouping rules")
log("   main      : groups as in the paper; attributions summed with sign inside a group, then absolute value")
log("   abs-sum   : same groups; absolute attributions summed inside a group")
log("   strict    : only predictors that appear in the published equation stay in a referenced group")
log("               (diastolic pressure, hypertension flag, cigarettes per day and glucose become unreferenced)")
log("   no-status : measured values only; yes/no status items (hypertension flag, BP medication, diabetes flag,")
log("               current-smoker flag) become unreferenced where a measured value exists in the group")
STRICT = dict(MAIN, diaBP="Unreferenced", prevalentHyp="Unreferenced", cigsPerDay="Unreferenced", glucose="Unreferenced")
NOSTAT = dict(MAIN, prevalentHyp="Unreferenced", BPMeds="Unreferenced", diabetes="Unreferenced", currentSmoker="Unreferenced")
rules = {"main": (MAIN, "signed"), "abs-sum": (MAIN, "abs"), "strict": (STRICT, "signed"), "no-status": (NOSTAT, "signed")}
for m, e in PAIRS:
    phi = base[f"phi_{m}_{e}"]
    parts = []
    for name, (mp_, how) in rules.items():
        G = group_matrix(mp_)
        gp = np.abs(phi @ G.T) if how == "signed" else np.abs(phi) @ G.T
        imp = gp.mean(0)
        v = imp[idx]
        nl = null(v, r, 5000)
        parts.append(f"{name} {cosine(v, r):.3f} (null {nl.mean():.3f}; Spearman {spearmanr(v, r)[0]:.2f}; "
                     f"unreferenced share {imp[-1] / imp.sum():.2f})")
    log(f"  {m:<4}{e:<12} " + " | ".join(parts))
R.close()
