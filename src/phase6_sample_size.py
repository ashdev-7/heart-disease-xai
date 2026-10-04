"""Phase 6: sample-size experiment on BRFSS (Decision Register D-28, exploratory).

  python src/phase6_sample_size.py run <n>     tune + 30 bootstraps (resumable, stops after ~8.5 min)
  python src/phase6_sample_size.py report      table across sizes

A fixed stratified subsample of n BRFSS training rows is bootstrapped 30 times.
Hyperparameters are re-tuned at each n. Explainer settings, background rows and
the 200 explained patients are the same at every size.
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import MLP, NATIVE, OUT, SEED, Prep, load, make_model, sigmoid
from metrics import boot_ci, corrected, expected_jaccard, ranks_desc

warnings.filterwarnings("ignore")
N_BOOT, N_PAT, BUDGET = 30, 200, 510
ROOT = OUT / "brfss" / "sample_size"
PAIRS = [(m, e) for m in ("lr", "rf", "xgb", "mlp") for e in NATIVE[m]]
d = load("brfss")
ytr_all, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()
F = len(d["features"])


def pair_scores(phi):
    """phi [V, P, F] -> per-patient mean pairwise top-3 Jaccard and Spearman across variants."""
    R = ranks_desc(phi)
    V = len(R)
    iu = np.triu_indices(V, 1)
    M = (R <= 3).astype(np.float32)
    inter = np.einsum("apf,bpf->pab", M, M)[:, iu[0], iu[1]]
    sa = M.sum(-1).T
    j3 = (inter / (sa[:, iu[0]] + sa[:, iu[1]] - inter)).mean(1)
    Z = R - R.mean(-1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=-1, keepdims=True)
    sp = np.einsum("apf,bpf->pab", Z, Z)[:, iu[0], iu[1]].mean(1)
    return j3, sp


if sys.argv[1] == "run":
    n = int(sys.argv[2])
    t0 = time.time()
    RES = ROOT / f"n{n}"
    RES.mkdir(parents=True, exist_ok=True)
    sub, _ = train_test_split(np.arange(len(ytr_all)), train_size=n, stratify=ytr_all, random_state=SEED)
    Xs, ys = d["Xtr"].iloc[sub], ytr_all[sub]
    batch = 64 if n < 10000 else 256

    hp_path = RES / "hyperparameters.json"
    if hp_path.exists():
        hps = json.loads(hp_path.read_text())
    else:
        grids = {
            "lr": [dict(C=c) for c in (0.001, 0.01, 0.1, 1, 10, 100)],
            "rf": [dict(n_estimators=300, min_samples_leaf=l, max_depth=dp) for l in (5, 10, 25) for dp in (6, 12)],
            "xgb": [dict(n_estimators=k, max_depth=dp, learning_rate=0.05) for dp in (2, 3, 4) for k in (100, 300, 600)],
        }
        hps = {}
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for name, grid in grids.items():
            sc = []
            for hp in grid:
                ls = []
                for tr, va in skf.split(Xs, ys):
                    pr = Prep().fit(Xs.iloc[tr])
                    mdl = make_model(name, hp, SEED).fit(pr.transform(Xs.iloc[tr]), ys[tr])
                    ls.append(log_loss(ys[va], sigmoid(mdl.margin(pr.transform(Xs.iloc[va])))))
                sc.append(np.mean(ls))
            hps[name] = grid[int(np.argmin(sc))]
        tr, va = train_test_split(np.arange(n), test_size=0.15, stratify=ys, random_state=SEED)
        pr = Prep().fit(Xs.iloc[tr])
        probe = MLP(epochs=100, batch=batch, seed=SEED).fit(
            pr.transform(Xs.iloc[tr]), ys[tr], pr.transform(Xs.iloc[va]), ys[va])
        hps["mlp"] = dict(epochs=int(probe.best_epoch), batch=batch)
        hp_path.write_text(json.dumps(hps, indent=2))
    print(f"n={n} events={int(ys.sum())} per predictor={ys.sum() / F:.1f} hyperparameters={hps}", flush=True)

    for b in range(N_BOOT):
        path = RES / f"boot_{b:03d}.npz"
        if path.exists():
            continue
        if time.time() - t0 > BUDGET:
            print(f"time budget reached at bootstrap {b}/{N_BOOT}; run again to resume", flush=True)
            sys.exit(0)
        rows = np.random.default_rng(1000 + b).integers(0, n, n)
        pr = Prep().fit(Xs.iloc[rows])
        Xb = pr.transform(Xs.iloc[rows])
        Xex, Xbg, Xte = pr.transform(d["Xex"].iloc[:N_PAT]), pr.transform(d["Xbg"]), pr.transform(d["Xte"])
        out = {}
        for m in ("lr", "rf", "xgb", "mlp"):
            mdl = make_model(m, hps[m], SEED).fit(Xb, ys[rows])
            out[f"auc_{m}"] = np.float32(roc_auc_score(yte, mdl.margin(Xte)))
            out[f"risk_{m}"] = sigmoid(mdl.margin(Xex)).astype(np.float32)
            for e, fn in NATIVE[m].items():
                out[f"phi_{m}_{e}"] = fn(mdl, Xex, Xbg).astype(np.float32)
        np.savez_compressed(path, **out)
        print(f"n={n} boot {b + 1}/{N_BOOT} ({time.time() - t0:.0f}s)", flush=True)
    print(f"n={n} COMPLETE", flush=True)

if sys.argv[1] == "report":
    ch = expected_jaccard(F, 3)
    rows = []
    sizes = sorted(int(p.name[1:]) for p in ROOT.glob("n*") if len(list(p.glob("boot_*.npz"))) >= N_BOOT)
    for n in sizes + [len(ytr_all)]:
        if n == len(ytr_all):
            files = sorted((OUT / "brfss" / "variants").glob("boot_*.npz"))[:N_BOOT]
            get = lambda z, k: z[k][:N_PAT]
            risk = lambda z, m: sigmoid(z[f"margin_{m}"][:N_PAT])
        else:
            files = sorted((ROOT / f"n{n}").glob("boot_*.npz"))[:N_BOOT]
            get = lambda z, k: z[k]
            risk = lambda z, m: z[f"risk_{m}"]
        Z = [np.load(f) for f in files]
        events = ytr_all.mean() * n
        for m, e in PAIRS:
            j3, sp = pair_scores(np.stack([get(z, f"phi_{m}_{e}") for z in Z]))
            lo, hi = boot_ci(j3)
            rows.append(dict(n=n, events_per_predictor=round(events / F, 1), model=m, explainer=e,
                             top3=j3.mean(), top3_lo=lo, top3_hi=hi, top3_corrected=corrected(j3.mean(), ch),
                             spearman=sp.mean(), pct_patients_below_half=(j3 < 0.5).mean(),
                             risk_sd=np.stack([risk(z, m) for z in Z]).std(0).mean(),
                             test_auc=float(np.mean([z[f"auc_{m}"] for z in Z]))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "brfss" / "sample_size.csv", index=False)
    pd.set_option("display.width", 250)
    piv = t.pivot_table(index=["model", "explainer"], columns="n", values="top3").round(3)
    auc = t.pivot_table(index="model", columns="n", values="test_auc").round(3)
    rsd = t.pivot_table(index="model", columns="n", values="risk_sd").round(4)
    epp = t.drop_duplicates("n").set_index("n").events_per_predictor
    txt = ["=" * 78, "SAMPLE-SIZE EXPERIMENT (BRFSS, exploratory, D-28)", "=" * 78,
           f"{N_BOOT} bootstraps per size, {N_PAT} explained patients, same background rows and explainer settings.",
           f"Chance level of top-3 Jaccard: {ch:.3f}", "",
           "Events per predictor at each training size:", epp.to_string(), "",
           "Per-patient top-3 agreement across bootstrap retrains, by training size:", piv.to_string(), "",
           "Mean test AUC across the bootstrap models:", auc.to_string(), "",
           "Prediction instability (mean SD of predicted risk across bootstraps):", rsd.to_string(), "",
           "Full detail with confidence intervals: results/brfss/sample_size.csv"]
    (OUT / "brfss" / "sample_size_report.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
