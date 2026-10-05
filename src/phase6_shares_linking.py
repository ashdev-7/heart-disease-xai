"""Phase 6: BRFSS attribution shares by feature class, and per-patient linking (A2).

  python src/phase6_shares_linking.py
Writes results/brfss/class_shares_report.txt and results/<dataset>/linking_report.txt
"""
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from common import NATIVE, OUT, load, sigmoid

PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
pd.set_option("display.width", 250)

# ----------------------------------------------------------- BRFSS class shares
if (OUT / "brfss" / "kernel.npz").exists():
    RES = OUT / "brfss"
    d = load("brfss")
    FEATS = d["features"]
    gm = pd.read_csv(OUT / "group_map.csv").query("dataset == 'brfss'").set_index("feature").group
    cls = gm.reindex(FEATS).str.replace(" (Ref-E only)", "", regex=False)
    CLASSES = ["Referenced", "Downstream", "Prior disease", "Access / social", "Unreferenced"]
    ref = pd.read_csv(OUT / "reference_importance.csv").query("dataset == 'brfss'")
    refD = {r: ref[ref.reference == r].set_index("item").importance for r in ref.reference.unique()}
    base, kern = np.load(RES / "variants" / "base.npz"), np.load(RES / "kernel.npz")
    boots = [np.load(f) for f in sorted((RES / "variants").glob("boot_*.npz"))]

    out = ["=" * 78, "BRFSS - WHERE THE ATTRIBUTION GOES, BY FEATURE CLASS", "=" * 78,
           "Classes: " + "; ".join(f"{c}: {', '.join(cls[cls == c].index)}" for c in CLASSES), ""]
    rows = []
    for m, e in PAIRS + [(m, "kernel_shap") for m in ("lr", "rf", "xgb", "mlp")]:
        phi = kern[m] if e == "kernel_shap" else base[f"phi_{m}_{e}"]
        imp = pd.Series(np.abs(phi).mean(0), index=FEATS)
        share = imp.groupby(cls).sum() / imp.sum()
        row = dict(model=m, explainer=e, **{c: share.get(c, 0.0) for c in CLASSES})
        if e != "kernel_shap":
            bs = []
            for bz in boots:
                i = pd.Series(np.abs(bz[f"phi_{m}_{e}"]).mean(0), index=FEATS)
                bs.append(i[cls == "Downstream"].sum() / i.sum())
            row["downstream_boot_lo"], row["downstream_boot_hi"] = np.percentile(bs, [2.5, 97.5])
        for rname, short in (("Ref-D (PAR)", "tau_PAR"), ("Ref-D (ln OR)", "tau_lnOR")):
            r = refD[rname]
            row[short] = kendalltau(imp[r.index], r.to_numpy())[0]
        row["top5"] = " > ".join(imp.sort_values(ascending=False).index[:5])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "class_shares.csv", index=False)
    out.append(tab.drop(columns="top5").round(3).to_string(index=False))
    out += ["", "downstream_boot_lo/hi: 2.5th-97.5th percentile of the downstream share across 50 bootstrap models.",
            "tau_PAR / tau_lnOR: Kendall tau between mean |attribution| and INTERHEART attributable risk /",
            "odds ratio on the 8 mapped items (secondary; see protocol 4.7).", "", "Top five features:"]
    out += [f"  {r.model:<4}{r.explainer:<12} {r.top5}" for r in tab.itertuples()]
    (RES / "class_shares_report.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))

# ------------------------------------------------------------------- linking
THR = {"framingham": 0.20, "brfss": float(load("brfss")["ytr"].mean())}
for ds, noise_level in (("framingham", "high"), ("brfss", "retest")):
    R = OUT / ds
    if not (R / "multiplicity_per_patient.npz").exists() or not (R / "base_axes_per_patient.npz").exists():
        continue
    mp = np.load(R / "multiplicity_per_patient.npz")
    bp = np.load(R / "base_axes_per_patient.npz")
    b0 = np.load(R / "variants" / "base.npz")
    pl = np.load(R / "plausibility_per_patient.npz") if (R / "plausibility_per_patient.npz").exists() else {}
    rows = []
    for m, e in PAIRS:
        risk = sigmoid(b0[f"margin_{m}"])
        v = dict(
            boot=mp[f"boot_{m}_{e}_jaccard3"], noise=bp[f"noise_{noise_level}_{m}_{e}_top3"],
            faith=bp[f"faith_{m}_{e}_ratio"], risk=risk, dist=np.abs(risk - THR[ds]),
            risk_sd=mp[f"boot_{m}_risk_sd"])
        if f"plaus_{m}_{e}" in pl:
            v["plaus"] = pl[f"plaus_{m}_{e}"]

        def rho(a, b):
            ok = ~(np.isnan(v[a]) | np.isnan(v[b]))
            return spearmanr(v[a][ok], v[b][ok])[0]

        row = dict(model=m, explainer=e, boot_vs_noise=rho("boot", "noise"),
                   boot_vs_risk_sd=rho("boot", "risk_sd"), boot_vs_dist=rho("boot", "dist"),
                   boot_vs_risk=rho("boot", "risk"), noise_vs_dist=rho("noise", "dist"),
                   noise_vs_risk=rho("noise", "risk"), boot_vs_faith=rho("boot", "faith"))
        if "plaus" in v:
            row["boot_vs_plaus"] = rho("boot", "plaus")
        # how much stabler is the top third of patients by predicted risk than the bottom third
        q = np.quantile(risk, [1 / 3, 2 / 3])
        row["boot_low_risk_third"] = v["boot"][risk <= q[0]].mean()
        row["boot_high_risk_third"] = v["boot"][risk >= q[1]].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    txt = ["=" * 78, f"LINKING ANALYSIS (A2)   dataset: {ds}   500 patients", "=" * 78,
           "Spearman correlation across patients. boot = per-patient top-3 agreement across bootstrap",
           "retrains; noise = top-3 agreement clean vs re-measured; risk_sd = SD of predicted risk across",
           f"bootstraps; dist = |predicted risk - {THR[ds]:.3f}|; faith = top-3 / random-3 prediction-gap ratio.",
           "With 500 patients a correlation has a standard error of about 0.045.", "",
           t.round(3).to_string(index=False)]
    (R / "linking_report.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
