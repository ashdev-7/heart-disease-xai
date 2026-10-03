"""Phase 5, Axis M - multiplicity of explanations across retraining.

  python src/phase5_multiplicity.py <dataset>

For every model x native explainer, and separately for seed variants and
bootstrap variants, reports
  per patient : mean pairwise top-3 / top-5 Jaccard, Spearman, rank-biased overlap
                of that patient's attribution ranking across training variants
  global      : the same for rankings of mean |attribution|
  chance      : expected value under random rankings, and chance-corrected scores
  convergence : scores on the first half of the bootstraps (Decision Register D-06)
  prediction  : spread of the patient's predicted risk across variants
Writes results/<dataset>/multiplicity.csv, multiplicity_report.txt,
       multiplicity_per_patient.npz
"""
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import OUT, SEED, load, sigmoid

DS = sys.argv[1]
RES = OUT / DS
VAR = RES / "variants"
EXPLAINERS = [("lr", "linear_shap"), ("rf", "tree_shap"), ("xgb", "tree_shap"),
              ("mlp", "ig"), ("mlp", "eg")]
RBO_P = 0.9

d = load(DS)
FEATS = d["features"]
F = len(FEATS)
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def stack(prefix, key):
    files = sorted(VAR.glob(f"{prefix}_*.npz"))
    return np.stack([np.load(f)[key] for f in files]), len(files)


def expected_jaccard(F, k):
    """Exact expectation of Jaccard between two independent random k-subsets of F items."""
    tot = comb(F, k)
    return sum(comb(k, i) * comb(F - k, k - i) / tot * (i / (2 * k - i)) for i in range(k + 1))


def expected_rbo(F, p):
    """Expected (truncated, normalised) RBO of two independent random full rankings."""
    w = np.array([p ** (dd - 1) for dd in range(1, F + 1)])
    agree = np.arange(1, F + 1) / F          # E|top-d ∩ top-d| / d = d / F
    return float((w * agree).sum() / w.sum())


def ranks_desc(a):
    """Rank 1 = largest |attribution|, along the last axis (ties averaged)."""
    return rankdata(-np.abs(a), axis=-1)


def pairwise_scores(R):
    """R: ranks [V, P, F]. Mean over variant pairs, per patient."""
    V, P, Fn = R.shape
    iu = np.triu_indices(V, 1)
    out = {}
    for k in (3, 5):
        M = (R <= k).astype(np.float32)                              # [V, P, F]
        inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]     # [P, pairs]
        sa = M.sum(-1).T                                             # [P, V] (ties can enlarge a set)
        union = sa[:, iu[0]] + sa[:, iu[1]] - inter
        out[f"jaccard{k}"] = (inter / union).mean(1)
    Z = R - R.mean(-1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=-1, keepdims=True)
    out["spearman"] = np.einsum("apf,bpf->pab", Z, Z)[:, iu[0], iu[1]].mean(1)
    w = np.array([RBO_P ** (dd - 1) for dd in range(1, Fn + 1)])
    acc = np.zeros((P, len(iu[0])), dtype=np.float32)
    for dd in range(1, Fn + 1):
        M = (R <= dd).astype(np.float32)
        acc += w[dd - 1] * np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]] / dd
    out["rbo"] = (acc / w.sum()).mean(1)
    return out


def ci(x, n_boot=2000):
    rng = np.random.default_rng(SEED)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return np.percentile(m, [2.5, 97.5])


CHANCE = {"jaccard3": expected_jaccard(F, 3), "jaccard5": expected_jaccard(F, 5),
          "spearman": 0.0, "rbo": expected_rbo(F, RBO_P)}

log("=" * 78)
log(f"AXIS M - MULTIPLICITY ACROSS RETRAINING   dataset: {DS}   features: {F}")
log("=" * 78)
log("Chance level (two independent random rankings): "
    + ", ".join(f"{k} {v:.3f}" for k, v in CHANCE.items()))
log("corrected = (observed - chance) / (1 - chance); 1 = identical, 0 = no better than chance")

rows, per_patient = [], {}
for source in ("seed", "boot"):
    for model, ex in EXPLAINERS:
        if source == "seed" and model == "lr":
            continue                                   # LR fit is deterministic
        phi, V = stack(source, f"phi_{model}_{ex}")
        if V < 3:
            continue
        R = ranks_desc(phi)
        sc = pairwise_scores(R)
        gR = ranks_desc(np.abs(phi).mean(1))[:, None, :]          # global ranking per variant
        gsc = pairwise_scores(gR)
        half = pairwise_scores(R[: V // 2]) if source == "boot" else None
        marg, _ = stack(source, f"margin_{model}")
        risk_sd = sigmoid(marg).std(0)
        for name, v in sc.items():
            lo, hi = ci(v)
            rows.append(dict(
                source=source, model=model, explainer=ex, n_variants=V, metric=name,
                per_patient=v.mean(), ci_lo=lo, ci_hi=hi, chance=CHANCE[name],
                per_patient_corrected=(v.mean() - CHANCE[name]) / (1 - CHANCE[name]),
                pct_patients_below_half=float((v < 0.5).mean()),
                global_ranking=float(gsc[name][0]),
                half_B=float(half[name].mean()) if half else np.nan,
                risk_sd_mean=float(risk_sd.mean())))
            per_patient[f"{source}_{model}_{ex}_{name}"] = v.astype(np.float32)
        per_patient[f"{source}_{model}_risk_sd"] = risk_sd.astype(np.float32)

tab = pd.DataFrame(rows)
tab.to_csv(RES / "multiplicity.csv", index=False)
np.savez_compressed(RES / "multiplicity_per_patient.npz", **per_patient)

pd.set_option("display.width", 200)
for source, title in (("boot", "BOOTSTRAP of the training set (model seed fixed)"),
                      ("seed", "SEED only (training set fixed)")):
    t = tab[tab.source == source]
    if t.empty:
        continue
    log(f"\n--- {title}; variants = {int(t.n_variants.iloc[0])} ---")
    for metric in ("jaccard3", "jaccard5", "spearman", "rbo"):
        m = t[t.metric == metric]
        log(f"\n  {metric}   (chance {CHANCE[metric]:.3f})")
        log(m[["model", "explainer", "per_patient", "ci_lo", "ci_hi", "per_patient_corrected",
               "pct_patients_below_half", "global_ranking", "half_B"]].round(3).to_string(index=False))

log("\nColumns: per_patient = mean over the 500 explained patients of the mean pairwise agreement of")
log("that patient's attribution ranking across training variants (95% CI by bootstrap over patients).")
log("global_ranking = same agreement for rankings of mean |attribution| over patients.")
log("half_B = per_patient recomputed on the first half of the bootstraps (convergence check).")
log("pct_patients_below_half = share of patients whose own agreement is below 0.5.")

t = tab[(tab.metric == "jaccard3")][["source", "model", "risk_sd_mean"]].drop_duplicates()
log("\nPrediction instability: mean over patients of the SD of predicted risk across variants")
log(t.round(4).to_string(index=False))

(RES / "multiplicity_report.txt").write_text("\n".join(lines), encoding="utf-8")
