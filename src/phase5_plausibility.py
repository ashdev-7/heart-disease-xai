"""Phase 5, Axis P - plausibility against published references (Framingham).

  python src/phase5_plausibility.py

Compares grouped attribution importance of every model x explainer with
  Ref-A  exact Shapley values of the D'Agostino BMI equation on this data
  Ref-B  D'Agostino lipid-model coefficient x SD
  Ref-C  INTERHEART attributable risk (ordinal)
Primary metric: cosine similarity of importance shares on the shared groups.
Context: permutation null, between-reference ceiling, spread over bootstraps.
Writes results/framingham/plausibility.csv, plausibility_report.txt,
       plausibility_per_patient.npz
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from common import NATIVE, OUT, SEED, load
from metrics import boot_ci

RES = OUT / "framingham"
d = load("framingham")
FEATS = d["features"]
gm = pd.read_csv(OUT / "group_map.csv").query("dataset == 'framingham'").set_index("feature").group
GROUPS = ["Age", "Sex", "Blood pressure", "Smoking", "Glycaemia", "Adiposity", "Lipids", "Unreferenced"]
G = np.array([[gm[f] == g for f in FEATS] for g in GROUPS], dtype=float)        # [groups, features]

ref = pd.read_csv(OUT / "reference_importance.csv").query("dataset == 'framingham'")
REF = {r: ref[ref.reference == r].set_index("item").importance for r in ref.reference.unique()}
refA_pp = pd.read_csv(OUT / "refA_per_patient.csv", index_col=0)

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def grouped(phi):
    """[..., features] -> [..., groups]: signed sum within group, then absolute value."""
    return np.abs(phi @ G.T)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def score(imp, refname):
    """imp: importance per group (len 8). Cosine on the groups the reference covers."""
    r = REF[refname]
    idx = [GROUPS.index(g) for g in r.index]
    return cosine(imp[idx], r.to_numpy())


def null_cosine(imp, refname, n=5000):
    r = REF[refname]
    idx = [GROUPS.index(g) for g in r.index]
    rng = np.random.default_rng(SEED)
    v = imp[idx]
    return np.array([cosine(rng.permutation(v), r.to_numpy()) for _ in range(n)])


base = np.load(RES / "variants" / "base.npz")
kern = np.load(RES / "kernel.npz")
boots = [np.load(f) for f in sorted((RES / "variants").glob("boot_*.npz"))]
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]

log("=" * 78)
log("AXIS P - PLAUSIBILITY AGAINST PUBLISHED REFERENCES   dataset: framingham")
log("=" * 78)
a, b = REF["Ref-A"], REF["Ref-B"]
sh = a.index.intersection(b.index)
log(f"Ceiling: Ref-A vs Ref-B cosine on shared groups {list(sh)} = {cosine(a[sh].to_numpy(), b[sh].to_numpy()):.3f}")
log("(both derive from D'Agostino 2008, so this is an upper bound on the ceiling, not an independent one)")
log(f"Bootstrap variants used for the spread: {len(boots)}")

rows, pp = [], {}
ex_idx = d["Xex"].index
have = ex_idx.isin(refA_pp.index)
refA_cols = [g for g in GROUPS if g in refA_pp.columns]
A_pp = refA_pp.reindex(ex_idx)[refA_cols].abs().to_numpy()
col_idx = [GROUPS.index(g) for g in refA_cols]

for m, e in PAIRS + [(m, "kernel_shap") for m in ("lr", "rf", "xgb", "mlp")]:
    phi = kern[m] if e == "kernel_shap" else base[f"phi_{m}_{e}"]
    gp = grouped(phi)                                    # [P, groups]
    imp = gp.mean(0)
    row = dict(model=m, explainer=e, unreferenced_share=imp[-1] / imp.sum())
    for rname in ("Ref-A", "Ref-B"):
        c = score(imp, rname)
        nul = null_cosine(imp, rname)
        row[f"cos_{rname}"] = c
        row[f"null_mean_{rname}"] = nul.mean()
        row[f"pct_null_below_{rname}"] = (nul < c).mean()
        if e != "kernel_shap":
            bs = [score(grouped(bz[f"phi_{m}_{e}"]).mean(0), rname) for bz in boots]
            row[f"boot_lo_{rname}"], row[f"boot_hi_{rname}"] = np.percentile(bs, [2.5, 97.5])
    r = REF["Ref-C (PAR)"]
    idx = [GROUPS.index(g) for g in r.index]
    row["tau_Ref-C_PAR"] = kendalltau(imp[idx], r.to_numpy())[0]
    order = [GROUPS[i] for i in np.argsort(-imp[:-1])]
    row["ranking"] = " > ".join(order)
    rows.append(row)

    # per patient: cosine between the patient's grouped |attribution| and Ref-A's for that patient
    num = (gp[:, col_idx] * A_pp).sum(1)
    den = np.linalg.norm(gp[:, col_idx], axis=1) * np.linalg.norm(A_pp, axis=1)
    pp[f"plaus_{m}_{e}"] = np.where(have, num / den, np.nan).astype(np.float32)
    lo, hi = boot_ci(pp[f"plaus_{m}_{e}"][have])
    row["per_patient_cos_Ref-A"] = float(np.nanmean(pp[f"plaus_{m}_{e}"]))
    row["pp_lo"], row["pp_hi"] = lo, hi

tab = pd.DataFrame(rows)
tab.to_csv(RES / "plausibility.csv", index=False)
np.savez_compressed(RES / "plausibility_per_patient.npz", **pp)
pd.set_option("display.width", 250)

log("\nGlobal importance shares vs references (cosine; 1 = same shares)")
log(tab[["model", "explainer", "cos_Ref-A", "boot_lo_Ref-A", "boot_hi_Ref-A", "null_mean_Ref-A",
         "pct_null_below_Ref-A", "cos_Ref-B", "boot_lo_Ref-B", "boot_hi_Ref-B", "tau_Ref-C_PAR",
         "unreferenced_share"]].round(3).to_string(index=False))
log("\nboot_lo / boot_hi: 2.5th and 97.5th percentile of the cosine across bootstrap-trained models.")
log("null_mean: mean cosine when the model's group importances are randomly reassigned to groups;")
log("pct_null_below: share of those random assignments that score lower than the model.")
log("unreferenced_share: share of attribution on education, heart rate, prior stroke.")

log("\nRef-A ranking : " + " > ".join(REF["Ref-A"].sort_values(ascending=False).index))
log("Ref-B ranking : " + " > ".join(REF["Ref-B"].sort_values(ascending=False).index))
log("Model rankings of the seven referenced groups:")
for _, r in tab.iterrows():
    log(f"  {r.model:<4}{r.explainer:<12} {r.ranking}")

log(f"\nPer-patient agreement with Ref-A (cosine of grouped |attribution|, {int(have.sum())} patients with a reference)")
log(tab[["model", "explainer", "per_patient_cos_Ref-A", "pp_lo", "pp_hi"]].round(3).to_string(index=False))

(RES / "plausibility_report.txt").write_text("\n".join(lines), encoding="utf-8")
