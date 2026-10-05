"""Audit 5 - second-degree checks on analysis settings, from stored explanations (no refits).

  python src/audit5_stored.py

For both datasets and every model-explainer pair, asks whether the headline comparison
(agreement across bootstrap retrains vs across seeds) depends on:
  A1 the number of explained patients (two random halves of 250)
  A2 the number of bootstrap retrains (first 25 vs all 50)
  A3 the choice of k in top-k agreement (k = 1..5, raw and chance-corrected)
  A4 the persistence parameter of rank-biased overlap (0.8, 0.9, 0.95)
  A5 how pairs are formed (all pairs of retrains vs each retrain against the base model)
Appends to results/audit_report.txt
"""
import time

import numpy as np

from common import NATIVE, OUT
from metrics import corrected, expected_jaccard, ranks_desc

PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
R = open(OUT / "audit_report.txt", "a", encoding="utf-8")


def log(s=""):
    print(s, flush=True)
    R.write(s + "\n")


def topk_pairs(Rk, k):
    """Per-patient mean top-k Jaccard over all pairs of runs. Rk: runs x patients x features."""
    M = (Rk <= k).astype(np.float32)
    iu = np.triu_indices(len(Rk), 1)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return (inter / (2 * k - inter)).mean(1)


def topk_vs(Rk, R0, k):
    inter = ((Rk <= k) & (R0[None] <= k)).sum(2)
    return (inter / (2 * k - inter)).mean(0)


def rbo_pairs(Rk, p):
    """Per-patient mean rank-biased overlap (truncated at full depth, normalised) over all pairs."""
    n, P, F = Rk.shape
    iu = np.triu_indices(n, 1)
    acc = np.zeros((P, len(iu[0])), dtype=np.float32)
    w = p ** np.arange(F)
    for dpt in range(1, F + 1):
        M = (Rk <= dpt).astype(np.float32)
        acc += w[dpt - 1] * np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]] / dpt
    return (acc / w.sum()).mean(1)


log("\n" + "=" * 78)
log(f"AUDIT 5 - ANALYSIS SETTINGS, CHECKED FROM STORED EXPLANATIONS   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)
for ds in ("framingham", "brfss"):
    V = OUT / ds / "variants"
    base = np.load(V / "base.npz")
    boots = [np.load(f) for f in sorted(V.glob("boot_*.npz"))]
    seeds = [np.load(f) for f in sorted(V.glob("seed_*.npz"))]
    F = base["phi_lr_linear_shap"].shape[1]
    log(f"\n--- {ds}: {len(boots)} bootstrap retrains, {len(seeds)} seed retrains, {F} features ---")
    rows = {k: [] for k in ("A1", "A2", "A3", "A4", "A5")}
    for m, e in PAIRS:
        key = f"phi_{m}_{e}"
        Rb = ranks_desc(np.stack([z[key] for z in boots]))
        Rs = ranks_desc(np.stack([z[key] for z in seeds])) if m != "lr" and seeds else None
        R0 = ranks_desc(base[key])
        b3 = topk_pairs(Rb, 3)
        s3 = topk_pairs(Rs, 3) if Rs is not None else None
        tag = f"{m:<4}{e:<12}"
        rows["A1"].append(f"  {tag} bootstrap: first 250 {b3[:250].mean():.3f}, last 250 {b3[250:].mean():.3f}"
                          + (f"; seed: {s3[:250].mean():.3f}, {s3[250:].mean():.3f}" if s3 is not None else ""))
        rows["A2"].append(f"  {tag} first 25 bootstraps {topk_pairs(Rb[:25], 3).mean():.3f}, all {len(Rb)} {b3.mean():.3f}")
        parts = []
        for k in (1, 2, 3, 4, 5):
            ch = expected_jaccard(F, k)
            b = topk_pairs(Rb, k).mean()
            t = f"k={k} boot {b:.2f} ({corrected(b, ch):.2f})"
            if Rs is not None:
                s = topk_pairs(Rs, k).mean()
                t += f" seed {s:.2f} ({corrected(s, ch):.2f})"
            parts.append(t)
        rows["A3"].append(f"  {tag} " + " | ".join(parts))
        rows["A4"].append(f"  {tag} " + " | ".join(
            f"p={p} boot {rbo_pairs(Rb, p).mean():.3f}" + (f" seed {rbo_pairs(Rs, p).mean():.3f}" if Rs is not None else "")
            for p in (0.8, 0.9, 0.95)))
        rows["A5"].append(f"  {tag} all pairs of bootstraps {b3.mean():.3f}; each bootstrap against the base model "
                          f"{topk_vs(Rb, R0, 3).mean():.3f}")
    for k, title in (("A1", "A1 Number of explained patients: top-3 agreement in two random halves of the 500"),
                     ("A2", "A2 Number of bootstrap retrains"),
                     ("A3", "A3 Choice of k: top-k agreement, raw (chance-corrected)"),
                     ("A4", "A4 Rank-biased overlap at three persistence values"),
                     ("A5", "A5 How pairs are formed")):
        log("\n" + title)
        for r in rows[k]:
            log(r)
R.close()
