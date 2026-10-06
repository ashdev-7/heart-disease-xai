"""Audit 8 - the Framingham-only design alternatives, repeated on BRFSS.

  python src/audit8_brfss_alternatives.py [results-root]

Reads the arrays written by  XAI_AUDIT_DS=brfss python src/audit3_alternatives.py <stage>
(fixedref, pathdep, subsample, retune_grouped) and reports per-patient top-3 agreement
across resamples for the first 200 explained patients.
Appends to results/audit_report.txt and writes results/brfss/audit8_report.txt
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from common import OUT
from metrics import expected_jaccard, ranks_desc

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
AUD = ROOT / "brfss" / "audit"
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def pair_top3(phi):
    Rk = ranks_desc(phi)
    iu = np.triu_indices(len(Rk), 1)
    M = (Rk <= 3).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    return (inter / (6 - inter)).mean()


def stack(prefix, key, n=None):
    files = sorted(AUD.glob(f"{prefix}*.npz"))[:n]
    return np.stack([np.load(f)[key] for f in files]) if files else None


log("\n" + "=" * 78)
log("AUDIT 8 - DESIGN ALTERNATIVES ON BRFSS (first 200 explained patients)")
log("=" * 78)
log(f"chance level of top-3 agreement {expected_jaccard(21, 3):.3f}")

n_fix = len(list(AUD.glob("fixedref_boot*.npz")))
log(f"\nReference: bootstrap, hyperparameters tuned once ({n_fix} resamples)")
ref = {}
for m, e in (("lr", "linear_shap"), ("rf", "tree_shap"), ("xgb", "tree_shap")):
    phi = stack("fixedref_boot", f"phi_{m}_{e}")
    if phi is not None:
        ref[m] = pair_top3(phi)
        log(f"  {m:<4}{e:<12} {ref[m]:.3f}")

n_pd = len(list(AUD.glob("pathdep_boot*.npz")))
if n_pd:
    log(f"\nPath-dependent instead of interventional TreeSHAP ({n_pd} resamples)")
    for m in ("rf", "xgb"):
        a, b = stack("pathdep_boot", f"{m}_interventional"), stack("pathdep_boot", f"{m}_pathdep")
        Ra, Rb = ranks_desc(a), ranks_desc(b)
        inter = ((Ra <= 3) & (Rb <= 3)).sum(-1)
        log(f"  {m:<4} bootstrap agreement: interventional {pair_top3(a):.3f}, path-dependent {pair_top3(b):.3f}; "
            f"the two variants agree with each other at {(inter / (6 - inter)).mean():.3f}")

n_ss = len(list(AUD.glob("subsample_*.npz")))
if n_ss:
    log(f"\n80% subsampling without replacement instead of the bootstrap ({n_ss} resamples)")
    for m, e in (("lr", "linear_shap"), ("rf", "tree_shap"), ("xgb", "tree_shap"), ("mlp", "ig"), ("mlp", "eg")):
        log(f"  {m:<4}{e:<12} {pair_top3(stack('subsample_', f'phi_{m}_{e}')):.3f}"
            + (f"   (bootstrap {ref[m]:.3f})" if m in ref else ""))

for tag, models in (("lr-xgb", ("lr", "xgb")), ("rf", ("rf",))):
    files = sorted(AUD.glob(f"retunegrp_{tag}_boot*.npz"))
    if not files:
        continue
    log(f"\nHyperparameters re-tuned inside each bootstrap with grouped folds: {', '.join(models)} ({len(files)} resamples)")
    hp = [json.loads(str(np.load(f)["hp"])) for f in files]
    for m in models:
        e = "linear_shap" if m == "lr" else "tree_shap"
        tuned = pair_top3(stack(f"retunegrp_{tag}_boot", f"phi_{m}_{e}"))
        once = pair_top3(stack("fixedref_boot", f"phi_{m}_{e}", len(files)))
        chosen = Counter(json.dumps({k: v for k, v in h[m].items() if k != "learning_rate"}) for h in hp)
        log(f"  {m:<4} re-tuned {tuned:.3f}; tuned once, same {len(files)} resamples {once:.3f}; difference {tuned - once:+.3f}")
        log(f"       settings chosen: " + "; ".join(f"{k} x{v}" for k, v in chosen.most_common()))
(OUT / "brfss" / "audit8_report.txt").write_text("\n".join(lines), encoding="utf-8")
with open(OUT / "audit_report.txt", "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
