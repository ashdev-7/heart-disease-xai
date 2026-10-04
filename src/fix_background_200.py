"""D-27 fix: redo the 200-row background lines for the tree explainers with all 200 rows used."""
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from common import NATIVE, OUT, load
from metrics import boot_ci, corrected, expected_jaccard, jaccard_topk, ranks_desc, spearman

warnings.filterwarnings("ignore")
rows = []
for ds in ("framingham", "brfss"):
    d = load(ds)
    B = joblib.load(OUT / ds / "base_models.joblib")
    prep, models = B["prep"], B["models"]
    Xex, Xbg, Xtr = prep.transform(d["Xex"])[:100], prep.transform(d["Xbg"]), prep.transform(d["Xtr"])
    ch = expected_jaccard(len(d["features"]), 3)
    for m in ("xgb", "rf"):
        fn = NATIVE[m]["tree_shap"]
        base = ranks_desc(fn(models[m], Xex, Xbg))
        alt = np.stack([ranks_desc(fn(models[m], Xex, Xtr[np.random.default_rng(500 + s).choice(
            len(Xtr), 200, replace=False)])) for s in range(5)])
        used = 200
        j3 = jaccard_topk(alt, base[None], 3).mean(0)
        lo, hi = boot_ci(j3)
        rows.append(dict(dataset=ds, item=f"{m} tree_shap, 200 rows (all used)", top3=j3.mean(), top3_lo=lo,
                         top3_hi=hi, top3_corrected=corrected(j3.mean(), ch),
                         top5=jaccard_topk(alt, base[None], 5).mean(),
                         spearman=spearman(alt, base[None]).mean()))
        print(rows[-1], flush=True)
t = pd.DataFrame(rows).round(3)
for ds in ("framingham", "brfss"):
    with open(OUT / ds / "base_axes_report.txt", "a", encoding="utf-8") as f:
        f.write("\n--- D-27 correction: tree explainers with a 200-row background, all 200 rows used ---\n")
        f.write("The earlier 'xgb tree_shap, 200 rows' line used only 100 of the 200 rows (shap default cap)\n")
        f.write("and must be read as another 100-row draw. Corrected values:\n")
        f.write(t[t.dataset == ds].drop(columns="dataset").to_string(index=False) + "\n")
