"""Audit 9 - sample-size experiment summarised over all independently drawn subsamples.

  python src/audit9_sample_size_summary.py

Each folder results/brfss/sample_size/n<size>[_s<seed>] holds 30 bootstrap models of one
subsample. For every size this reports, per model and explainer, the mean, SD and range
across subsamples of the per-patient top-3 agreement, and the mean test AUC.
Writes results/brfss/sample_size_summary.txt and sample_size_summary.csv
"""
import numpy as np
import pandas as pd

from common import NATIVE, OUT, load
from metrics import ranks_desc

ROOT = OUT / "brfss" / "sample_size"
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
N_BOOT, N_PAT = 30, 200
d = load("brfss")
F, PREV, NFULL = len(d["features"]), d["ytr"].mean(), len(d["ytr"])


def top3(phi):
    Rk = ranks_desc(phi)
    iu = np.triu_indices(len(Rk), 1)
    M = (Rk <= 3).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return float((inter / (6 - inter)).mean())


rows = []
for p in sorted(ROOT.glob("n*")):
    files = sorted(p.glob("boot_*.npz"))[:N_BOOT]
    if len(files) < N_BOOT:
        continue
    n = int(p.name[1:].split("_s")[0])
    Z = [np.load(f) for f in files]
    for m, e in PAIRS:
        rows.append(dict(n=n, subsample=p.name, model=m, explainer=e,
                         top3=top3(np.stack([z[f"phi_{m}_{e}"] for z in Z])),
                         auc=float(np.mean([z[f"auc_{m}"] for z in Z]))))
Z = [np.load(f) for f in sorted((OUT / "brfss" / "variants").glob("boot_*.npz"))[:N_BOOT]]
for m, e in PAIRS:
    rows.append(dict(n=NFULL, subsample="full", model=m, explainer=e,
                     top3=top3(np.stack([z[f"phi_{m}_{e}"][:N_PAT] for z in Z])),
                     auc=float(np.mean([z[f"auc_{m}"] for z in Z]))))
t = pd.DataFrame(rows)
g = t.groupby(["model", "explainer", "n"]).agg(subsamples=("top3", "size"), mean=("top3", "mean"), sd=("top3", "std"),
                                               lo=("top3", "min"), hi=("top3", "max"), auc=("auc", "mean")).reset_index()
g["events_per_predictor"] = (g.n * PREV / F).round(0)
g.to_csv(OUT / "brfss" / "sample_size_summary.csv", index=False)
pd.set_option("display.width", 250)
txt = ["=" * 78, "SAMPLE-SIZE EXPERIMENT, ALL SUBSAMPLES (BRFSS)", "=" * 78,
       f"{N_BOOT} bootstraps per subsample, {N_PAT} explained patients. One row per model, explainer and size:",
       "mean, SD, lowest and highest per-patient top-3 agreement across independently drawn subsamples.", "",
       g.round(3).to_string(index=False), ""]
sizes = sorted(g.n.unique())
txt.append("Separation between adjacent sizes (does the lowest subsample at the larger size exceed the highest at the smaller?):")
for (m, e), s in g.groupby(["model", "explainer"]):
    s = s.set_index("n")
    parts = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        sep = s.loc[b, "lo"] > s.loc[a, "hi"]
        parts.append(f"{a}->{b}: mean {s.loc[b, 'mean'] - s.loc[a, 'mean']:+.3f}, {'separated' if sep else 'ranges overlap'}")
    txt.append(f"  {m:<4}{e:<12} " + "; ".join(parts))
txt += ["", "Mean over the five model-explainer rows, per subsample (for a rank test of trend):"]
per = t.groupby(["n", "subsample"]).top3.mean().reset_index()
for n in sizes:
    v = per[per.n == n].top3
    txt.append(f"  n={n}: {len(v)} subsamples, mean {v.mean():.3f}, range {v.min():.3f} to {v.max():.3f}")
from scipy.stats import spearmanr
rho, pval = spearmanr(per.n, per.top3)
txt.append(f"  Spearman correlation between training size and subsample-level agreement: {rho:.2f} "
           f"({len(per)} subsamples, p = {pval:.1e})")
(OUT / "brfss" / "sample_size_summary.txt").write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
