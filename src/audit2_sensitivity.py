"""Audit 2 - do the conclusions survive different analysis choices? (stored results only)

  python src/audit2_sensitivity.py

S1 complete-case check for imputed values (Decision Register D-04)
S2 metric choice: top-1, top-3, top-5, Spearman, rank-biased overlap, magnitude cosine
S3 grouped features instead of raw features (Framingham)
S4 sign stability of the leading features
S5 paired differences with confidence intervals for the two headline contrasts
S6 decision-threshold choice in the linking analysis
S7 patients with and without the outcome
Appends to results/audit_report.txt
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import NATIVE, OUT, load, sigmoid
from metrics import boot_ci, corrected, expected_jaccard, ranks_desc

PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
REPORT = open(OUT / "audit_report.txt", "a", encoding="utf-8")
pd.set_option("display.width", 250)


def log(s=""):
    print(s, flush=True)
    REPORT.write(s + "\n")


def stack(ds, prefix, key):
    return np.stack([np.load(f)[key] for f in sorted((OUT / ds / "variants").glob(f"{prefix}_*.npz"))])


def pair_mean(M):
    """M [V, P, F] membership/standardised vectors -> per-patient mean pairwise inner product."""
    V = len(M)
    s = M.sum(0)
    return ((s * s).sum(-1) - (M * M).sum((0, 2))) / (V * (V - 1))


def topk_jaccard(R, k):
    V = len(R)
    iu = np.triu_indices(V, 1)
    M = (R <= k).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return (inter / (2 * k - inter)).mean(1)


def paired_ci(x, n_boot=4000):
    rng = np.random.default_rng(42)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return np.percentile(b, [2.5, 97.5])


log("\n" + "=" * 78)
log(f"AUDIT 2 - SENSITIVITY OF THE CONCLUSIONS (stored results)   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)

# ------------------------------------------------------------------ S1
log("\nS1 Complete-case check (D-04): Framingham explained patients with and without imputed values")
d = load("framingham")
miss_any = d["Xex"].isna().any(axis=1).to_numpy()
miss_glu = d["Xex"]["glucose"].isna().to_numpy()
log(f"  500 explained patients: {int(miss_any.sum())} have at least one imputed value, "
    f"{int(miss_glu.sum())} have imputed glucose.")
mp = np.load(OUT / "framingham" / "multiplicity_per_patient.npz")
bp = np.load(OUT / "framingham" / "base_axes_per_patient.npz")
pl = np.load(OUT / "framingham" / "plausibility_per_patient.npz")
base = np.load(OUT / "framingham" / "variants" / "base.npz")
gi = d["features"].index("glucose")
rows = []
for m, e in PAIRS:
    sh = np.abs(base[f"phi_{m}_{e}"])
    sh = sh[:, gi] / sh.sum(1)
    for label, arr in (("bootstrap top-3", mp[f"boot_{m}_{e}_jaccard3"]),
                       ("noise top-3 (high)", bp[f"noise_high_{m}_{e}_top3"]),
                       ("plausibility cosine", pl[f"plaus_{m}_{e}"]),
                       ("faithfulness ratio", bp[f"faith_{m}_{e}_ratio"]),
                       ("glucose share of attribution", sh)):
        lo, hi = boot_ci(arr[~miss_any])
        rows.append(dict(model=m, explainer=e, score=label, all_500=np.nanmean(arr),
                         complete=np.nanmean(arr[~miss_any]), complete_lo=lo, complete_hi=hi,
                         imputed=np.nanmean(arr[miss_any]), glucose_imputed=np.nanmean(arr[miss_glu])))
t = pd.DataFrame(rows)
for label in t.score.unique():
    log(f"\n  {label}")
    log(t[t.score == label].drop(columns="score").round(3).to_string(index=False))
d1 = t[t.score != "glucose share of attribution"]
log(f"\n  Largest change in any score when restricted to complete cases: "
    f"{(d1.complete - d1.all_500).abs().max():.3f} (faithfulness ratio excluded: "
    f"{(d1[d1.score != 'faithfulness ratio'].complete - d1[d1.score != 'faithfulness ratio'].all_500).abs().max():.3f})")

# ------------------------------------------------------------------ S2
log("\nS2 Metric choice: is 'bootstrap agreement < seed agreement' true under every metric?")
for ds in ("framingham", "brfss"):
    rows = []
    F = len(load(ds)["features"])
    for m, e in PAIRS:
        if m == "lr":
            continue
        out = dict(model=m, explainer=e)
        for src in ("seed", "boot"):
            phi = stack(ds, src, f"phi_{m}_{e}")[:, :200]
            R = ranks_desc(phi)
            out[f"top1_{src}"] = topk_jaccard(R, 1).mean()
            out[f"top3_{src}"] = topk_jaccard(R, 3).mean()
            out[f"top5_{src}"] = topk_jaccard(R, 5).mean()
            Z = R - R.mean(-1, keepdims=True)
            Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
            out[f"spearman_{src}"] = pair_mean(Z).mean()
            U = phi / np.linalg.norm(phi, axis=-1, keepdims=True)
            out[f"cosine_{src}"] = pair_mean(U).mean()
        rows.append(out)
    t = pd.DataFrame(rows)
    log(f"\n  {ds} (first 200 patients; top-1 chance {1 / F:.3f}, top-3 {expected_jaccard(F, 3):.3f})")
    log(t.round(3).to_string(index=False))
    n_true = sum((t[f"{k}_boot"] < t[f"{k}_seed"]).sum() for k in ("top1", "top3", "top5", "spearman", "cosine"))
    log(f"  bootstrap < seed in {n_true} of {5 * len(t)} model x metric cells")
log("\n  cosine = similarity of the raw signed attribution vectors (a magnitude-based measure).")
log("  It stays high where rank-based measures fall, as Hwang et al. 2026 warn.")

# ------------------------------------------------------------------ S3
log("\nS3 Grouped features (Framingham): does instability survive when the four blood-pressure,")
log("   two smoking and two glycaemia columns are merged before ranking?")
gm = pd.read_csv(OUT / "group_map.csv").query("dataset == 'framingham'").set_index("feature").group
groups = list(dict.fromkeys(gm.reindex(d["features"])))
G = np.array([[gm[f] == g for f in d["features"]] for g in groups], dtype=np.float32)
ch_raw, ch_grp = expected_jaccard(15, 3), expected_jaccard(len(groups), 3)
rows = []
for m, e in PAIRS:
    out = dict(model=m, explainer=e)
    for src in ("seed", "boot"):
        if m == "lr" and src == "seed":
            continue
        phi = stack("framingham", src, f"phi_{m}_{e}")
        raw = topk_jaccard(ranks_desc(phi), 3).mean()
        grp = topk_jaccard(ranks_desc(np.abs(phi @ G.T)), 3).mean()
        g1 = topk_jaccard(ranks_desc(np.abs(phi @ G.T)), 1).mean()
        out.update({f"raw_{src}": raw, f"raw_{src}_corr": corrected(raw, ch_raw),
                    f"grouped_{src}": grp, f"grouped_{src}_corr": corrected(grp, ch_grp),
                    f"grouped_top1_{src}": g1})
    rows.append(out)
log(f"  chance level: raw 15 features {ch_raw:.3f}; {len(groups)} groups {ch_grp:.3f}; "
    f"'_corr' = chance-corrected (0 = chance, 1 = identical)")
log(pd.DataFrame(rows).round(3).to_string(index=False))

# ------------------------------------------------------------------ S4
log("\nS4 Sign stability: for each patient's three leading features in the base explanation,")
log("   how often does a bootstrap-retrained model give the attribution the SAME sign?")
for ds in ("framingham", "brfss"):
    b0 = np.load(OUT / ds / "variants" / "base.npz")
    rows = []
    for m, e in PAIRS:
        phi0 = b0[f"phi_{m}_{e}"]
        top = np.argsort(-np.abs(phi0), axis=1)[:, :3]
        phi = stack(ds, "boot", f"phi_{m}_{e}")
        s0 = np.sign(np.take_along_axis(phi0, top, 1))
        sb = np.sign(np.take_along_axis(phi, np.broadcast_to(top, (len(phi),) + top.shape), 2))
        same = (sb == s0[None]).mean(0)                       # [P, 3]
        rows.append(dict(model=m, explainer=e, rank1=same[:, 0].mean(), rank2=same[:, 1].mean(),
                         rank3=same[:, 2].mean(), pct_patients_rank1_always_same=(same[:, 0] == 1).mean()))
    log(f"\n  {ds}")
    log(pd.DataFrame(rows).round(3).to_string(index=False))

# ------------------------------------------------------------------ S5
log("\nS5 Paired differences with 95% confidence intervals (bootstrap over patients)")
for ds in ("framingham", "brfss"):
    mpd = np.load(OUT / ds / "multiplicity_per_patient.npz")
    bpd = np.load(OUT / ds / "base_axes_per_patient.npz")
    log(f"\n  {ds}: seed-only agreement minus bootstrap agreement (top-3), per patient")
    for m, e in PAIRS:
        if m == "lr":
            continue
        diff = mpd[f"seed_{m}_{e}_jaccard3"] - mpd[f"boot_{m}_{e}_jaccard3"]
        lo, hi = paired_ci(diff)
        log(f"    {m:<4}{e:<10} {diff.mean():+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    names = ["lr", "rf", "xgb", "mlp"]
    model_pairs = [f"model_{a}_{b}_top3" for i, a in enumerate(names) for b in names[i + 1:]]
    model_swap = np.mean([bpd[k] for k in model_pairs], axis=0)
    expl_all = np.mean([bpd[f"explainer_{m}_{e}_top3"] for m, e in PAIRS], axis=0)
    diff = expl_all - model_swap
    lo, hi = paired_ci(diff)
    log(f"  {ds}: explainer-swap agreement minus model-swap agreement (top-3), per patient")
    log(f"    all 5 explainer pairs vs all 6 model pairs      {diff.mean():+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    no_rf_pairs = [k for k in model_pairs if "rf" not in k]
    ms2 = np.mean([bpd[k] for k in no_rf_pairs], axis=0)
    ex2 = np.mean([bpd[f"explainer_{m}_{e}_top3"] for m, e in PAIRS if m != "rf"], axis=0)
    diff = ex2 - ms2
    lo, hi = paired_ci(diff)
    log(f"    excluding the random forest from both sides    {diff.mean():+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
    shap_only = np.mean([bpd[f"explainer_{m}_{e}_top3"] for m, e in PAIRS if e != "ig" and m != "rf"], axis=0)
    diff = shap_only - ms2
    lo, hi = paired_ci(diff)
    log(f"    Shapley-type explainers only, no random forest {diff.mean():+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")

# ------------------------------------------------------------------ S6
log("\nS6 Threshold choice: correlation between bootstrap agreement and distance from the threshold")
for ds, thrs in (("framingham", (0.20, 0.152, 0.10, 0.075)), ("brfss", (0.094, 0.20, 0.05))):
    mpd = np.load(OUT / ds / "multiplicity_per_patient.npz")
    b0 = np.load(OUT / ds / "variants" / "base.npz")
    rows = []
    for m, e in PAIRS:
        risk = sigmoid(b0[f"margin_{m}"])
        rows.append(dict(model=m, explainer=e, **{f"thr_{t}": spearmanr(
            mpd[f"boot_{m}_{e}_jaccard3"], np.abs(risk - t))[0] for t in thrs}))
    log(f"\n  {ds}")
    log(pd.DataFrame(rows).round(3).to_string(index=False))

# ------------------------------------------------------------------ S7
log("\nS7 Patients with and without the outcome: bootstrap top-3 agreement")
for ds in ("framingham", "brfss"):
    dd = load(ds)
    y = dd["yex"].to_numpy().astype(bool)
    mpd = np.load(OUT / ds / "multiplicity_per_patient.npz")
    rows = [dict(model=m, explainer=e, with_outcome=mpd[f"boot_{m}_{e}_jaccard3"][y].mean(),
                 without_outcome=mpd[f"boot_{m}_{e}_jaccard3"][~y].mean()) for m, e in PAIRS]
    log(f"\n  {ds}: {int(y.sum())} with the outcome, {int((~y).sum())} without")
    log(pd.DataFrame(rows).round(3).to_string(index=False))
REPORT.close()
