"""Phase 6: SMOTE sensitivity arm (Decision Register D-29). Base models only.

  python src/phase6_smote_arm.py <dataset>
Refits the four models on a SMOTE-balanced training set with the same
hyperparameters and compares calibration and explanations with the
no-resampling base models. Writes results/<dataset>/smote_report.txt
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

from common import NATIVE, OUT, SEED, Prep, load, make_model, sigmoid
from metrics import boot_ci, corrected, expected_jaccard, jaccard_topk, ranks_desc, spearman

warnings.filterwarnings("ignore")
DS = sys.argv[1]
RES = OUT / DS
d = load(DS)
F = len(d["features"])
hps = json.loads((RES / "hyperparameters.json").read_text())
# same-environment reference: the laptop's no-resampling base explanations
ref_dir = OUT / "brfss_local_discarded" if (DS == "brfss" and (OUT / "brfss_local_discarded").exists()) else RES
base = np.load(ref_dir / "variants" / "base.npz")
perf0 = pd.read_csv(ref_dir / "performance.csv").set_index("model")

prep = Prep().fit(d["Xtr"])
Xs, ys = SMOTE(k_neighbors=5, random_state=SEED).fit_resample(prep.transform(d["Xtr"]), d["ytr"].to_numpy())
Xs = Xs.astype(np.float32)
Xex, Xbg, Xte = prep.transform(d["Xex"]), prep.transform(d["Xbg"]), prep.transform(d["Xte"])
yte = d["yte"].to_numpy()
ch = expected_jaccard(F, 3)

lines = ["=" * 78, f"SMOTE SENSITIVITY ARM   dataset: {DS}", "=" * 78,
         f"Training rows {len(d['ytr'])} -> {len(ys)} after SMOTE (positives {int(d['ytr'].sum())} -> {int(ys.sum())}).",
         "Same hyperparameters as the no-resampling models; background and explained patients unchanged.", ""]
perf, agree, glob = [], [], []
for m in ("lr", "rf", "xgb", "mlp"):
    model = make_model(m, hps[m], SEED).fit(Xs, ys)
    mg = model.margin(Xte)
    p = sigmoid(mg)
    slope = sm.Logit(yte, sm.add_constant(mg)).fit(disp=0).params[1]
    citl = sm.GLM(yte, np.ones((len(yte), 1)), family=sm.families.Binomial(), offset=mg).fit().params[0]
    perf.append(dict(model=m, auc_none=perf0.loc[m, "auc"], auc_smote=roc_auc_score(yte, p),
                     mean_pred_none=perf0.loc[m, "mean_pred"], mean_pred_smote=p.mean(),
                     prevalence=yte.mean(), slope_none=perf0.loc[m, "cal_slope"], slope_smote=slope,
                     citl_none=perf0.loc[m, "cal_in_large"], citl_smote=citl))
    for e, fn in NATIVE[m].items():
        phi = fn(model, Xex, Xbg)
        R1, R0 = ranks_desc(phi), ranks_desc(base[f"phi_{m}_{e}"])
        j3 = jaccard_topk(R1, R0, 3)
        lo, hi = boot_ci(j3)
        agree.append(dict(model=m, explainer=e, top3=j3.mean(), top3_lo=lo, top3_hi=hi,
                          top3_corrected=corrected(j3.mean(), ch), top5=jaccard_topk(R1, R0, 5).mean(),
                          spearman=spearman(R1, R0).mean(), pct_top3_below_half=(j3 < 0.5).mean()))
        g1 = pd.Series(np.abs(phi).mean(0), index=d["features"]).sort_values(ascending=False)
        g0 = pd.Series(np.abs(base[f"phi_{m}_{e}"]).mean(0), index=d["features"]).sort_values(ascending=False)
        glob.append(f"  {m:<4}{e:<12} none : {' > '.join(g0.index[:5])}")
        glob.append(f"  {'':<16} SMOTE: {' > '.join(g1.index[:5])}")
    print(m, "done", flush=True)

pd.set_option("display.width", 250)
lines += ["Performance and calibration on the untouched test set:",
          pd.DataFrame(perf).round(3).to_string(index=False), "",
          "mean_pred should equal prevalence; slope should be 1; citl (calibration-in-the-large) should be 0.", "",
          "Per-patient agreement between the SMOTE-model explanation and the no-resampling explanation:",
          pd.DataFrame(agree).round(3).to_string(index=False), "",
          f"(chance level of top-3 Jaccard: {ch:.3f})", "", "Global top-5 features:"] + glob
(RES / "smote_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
