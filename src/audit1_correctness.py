"""Audit 1 - is the code computing what we say it computes?

  python src/audit1_correctness.py [metrics|shapley|leakage|determinism|all]

T1 metrics      unit tests of every agreement metric and the ICC estimator against
                independent implementations / known answers; production outputs
                recomputed by brute force
T2 shapley      exact Shapley values by full enumeration (2^15 coalitions) as ground
                truth for every explainer on the Framingham base models
T3 leakage      index disjointness; label-permutation test (AUC must fall to 0.5);
                model-randomisation sanity check for explanations (Adebayo et al. 2018)
T4 determinism  refitting the base models reproduces the stored attributions
Appends to results/audit_report.txt
"""
import itertools
import sys
import time
import warnings

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import NATIVE, OUT, SEED, Prep, load, make_model, sigmoid
from metrics import expected_jaccard, jaccard_topk, ranks_desc, spearman

warnings.filterwarnings("ignore")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "all"
REPORT = open(OUT / "audit_report.txt", "a", encoding="utf-8")
FAILS = []


def log(s=""):
    print(s, flush=True)
    REPORT.write(s + "\n")
    REPORT.flush()


def check(name, ok, detail=""):
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


log("\n" + "=" * 78)
log(f"AUDIT 1 - CORRECTNESS   stage: {STAGE}   {time.strftime('%Y-%m-%d %H:%M')}")
log("=" * 78)

# ============================================================== T1 metrics
if STAGE in ("metrics", "all"):
    log("\nT1 Metrics and estimators")
    rng = np.random.default_rng(0)
    # hand example: ranks 1..5; top-3 sets {a,b,c} vs {a,b,d} -> 2/4
    Ra = np.array([1, 2, 3, 4, 5.0])
    Rb = np.array([1, 2, 4, 3, 5.0])
    check("top-3 Jaccard hand example = 0.5", abs(jaccard_topk(Ra, Rb, 3) - 0.5) < 1e-12)
    check("identical rankings -> Jaccard 1, Spearman 1",
          jaccard_topk(Ra, Ra, 3) == 1 and abs(spearman(Ra, Ra) - 1) < 1e-12)
    A, Bm = rng.normal(size=(200, 15)), rng.normal(size=(200, 15))
    mine = spearman(ranks_desc(A), ranks_desc(Bm))
    ref = np.array([spearmanr(np.abs(a), np.abs(b))[0] for a, b in zip(A, Bm)])
    check("Spearman equals scipy.stats.spearmanr on 200 random pairs", np.abs(mine - ref).max() < 1e-10,
          f"max diff {np.abs(mine - ref).max():.1e}")
    for F, k in ((15, 3), (15, 5), (21, 3), (21, 5), (8, 3)):
        sims = []
        for _ in range(100000):
            a, b = rng.permutation(F)[:k], rng.permutation(F)[:k]
            i = len(np.intersect1d(a, b))
            sims.append(i / (2 * k - i))
        mc, ex = np.mean(sims), expected_jaccard(F, k)
        check(f"chance level of top-{k} Jaccard, {F} features: formula {ex:.4f} vs simulation {mc:.4f}",
              abs(mc - ex) < 0.003)
    # random attributions really do score at the chance level through the production path
    P1, P2 = rng.normal(size=(5000, 15)), rng.normal(size=(5000, 15))
    obs = jaccard_topk(ranks_desc(P1), ranks_desc(P2), 3).mean()
    check("random attribution vectors score at chance through ranks_desc + jaccard_topk",
          abs(obs - expected_jaccard(15, 3)) < 0.01, f"observed {obs:.4f}")

    # production per-patient multiplicity recomputed by brute force (no shared code path)
    mp = np.load(OUT / "framingham" / "multiplicity_per_patient.npz")
    files = sorted((OUT / "framingham" / "variants").glob("boot_*.npz"))
    phi = np.stack([np.load(f)["phi_xgb_tree_shap"][:25] for f in files])          # [100, 25, 15]
    brute_j, brute_s = np.zeros(25), np.zeros(25)
    pairs = list(itertools.combinations(range(len(phi)), 2))
    for p in range(25):
        js, ss = [], []
        for a, b in pairs:
            ta = set(np.argsort(-np.abs(phi[a, p]))[:3])
            tb = set(np.argsort(-np.abs(phi[b, p]))[:3])
            js.append(len(ta & tb) / len(ta | tb))
            ss.append(spearmanr(np.abs(phi[a, p]), np.abs(phi[b, p]))[0])
        brute_j[p], brute_s[p] = np.mean(js), np.mean(ss)
    dj = np.abs(brute_j - mp["boot_xgb_tree_shap_jaccard3"][:25]).max()
    ds_ = np.abs(brute_s - mp["boot_xgb_tree_shap_spearman"][:25]).max()
    check("stored bootstrap top-3 Jaccard (XGB, 25 patients, 4,950 pairs) = brute force", dj < 1e-3,
          f"max diff {dj:.1e}")
    check("stored bootstrap Spearman = brute force with scipy", ds_ < 1e-3, f"max diff {ds_:.1e}")

    # ICC estimator on synthetic data with known variance components
    sys.argv = ["x", "framingham", "none"]
    import phase6_variance as pv
    est = []
    for rep in range(200):
        r = np.random.default_rng(rep)
        between = r.normal(0, np.sqrt(0.03), size=(1, 100, 4))       # patients
        within = r.normal(0, np.sqrt(0.01), size=(20, 100, 4))       # levels of the source
        est.append(pv.icc(between + within)[0])
    check("ICC estimator recovers a known value (true 0.75)", abs(np.mean(est) - 0.75) < 0.01,
          f"mean of 200 simulations {np.mean(est):.4f}, SD {np.std(est):.4f}")

# ============================================================== T2 shapley
if STAGE in ("shapley", "all"):
    import shap
    log("\nT2 Exact Shapley values by full enumeration (ground truth), Framingham base models")
    d = load("framingham")
    B = joblib.load(OUT / "framingham" / "base_models.joblib")
    prep, models = B["prep"], B["models"]
    Xex, Xbg = prep.transform(d["Xex"]), prep.transform(d["Xbg"])
    F = Xex.shape[1]
    masks = np.arange(2 ** F)
    bits = ((masks[:, None] >> np.arange(F)) & 1).astype(bool)        # [2^F, F]
    size = bits.sum(1)
    from math import factorial
    w_size = np.array([factorial(s) * factorial(F - s - 1) / factorial(F) for s in range(F)])

    def exact(f, x):
        v = np.empty(2 ** F)
        for s in range(0, 2 ** F, 2048):
            bb = bits[s:s + 2048]                                     # [c, F]
            Xc = np.where(bb[:, None, :], x[None, None, :], Xbg[None, :, :])   # [c, bg, F]
            v[s:s + 2048] = f(Xc.reshape(-1, F).astype(np.float32)).reshape(len(bb), len(Xbg)).mean(1)
        phi = np.zeros(F)
        for j in range(F):
            without = masks[~bits[:, j]]
            phi[j] = (w_size[size[without]] * (v[without | (1 << j)] - v[without])).sum()
        return phi

    kern = np.load(OUT / "framingham" / "kernel.npz")
    base = np.load(OUT / "framingham" / "variants" / "base.npz")
    rf_prob = lambda X: models["rf"].est.predict_proba(X)[:, 1]
    jobs = [("lr", models["lr"].margin, 10), ("xgb", models["xgb"].margin, 10),
            ("mlp", models["mlp"].margin, 10), ("rf", models["rf"].margin, 4), ("rf_prob", rf_prob, 4)]
    for name, f, n_pat in jobs:
        t = time.time()
        E = np.stack([exact(f, Xex[i]) for i in range(n_pat)])
        scale = np.abs(E).sum(1).mean()
        log(f"  {name}: {n_pat} patients, exact enumeration {time.time() - t:.0f}s, mean total |phi| {scale:.3f}")

        def cmp(label, est, tol_rel):
            err = np.abs(est[:n_pat] - E)
            j3 = jaccard_topk(ranks_desc(est[:n_pat]), ranks_desc(E), 3).mean()
            sp = spearman(ranks_desc(est[:n_pat]), ranks_desc(E)).mean()
            check(f"{label} vs exact", err.max() / scale < tol_rel,
                  f"max error {err.max():.2e} ({100 * err.max() / scale:.2f}% of total), "
                  f"mean error {err.mean():.2e}, top-3 agreement {j3:.3f}, Spearman {sp:.3f}")

        if name == "lr":
            cmp("LR linear SHAP", base["phi_lr_linear_shap"], 1e-4)
            cmp("LR KernelSHAP", kern["lr"], 0.10)
        if name == "xgb":
            cmp("XGB TreeSHAP (interventional)", base["phi_xgb_tree_shap"], 1e-3)
            cmp("XGB KernelSHAP", kern["xgb"], 0.10)
        if name == "mlp":
            cmp("MLP KernelSHAP", kern["mlp"], 0.10)
            cmp("MLP Expected Gradients (not a Shapley estimator; closeness reported)",
                base["phi_mlp_eg"], 10.0)
            cmp("MLP Integrated Gradients (single baseline; closeness reported)",
                base["phi_mlp_ig"], 10.0)
        if name == "rf":
            cmp("RF KernelSHAP (log-odds)", kern["rf"], 0.10)
        if name == "rf_prob":
            cmp("RF TreeSHAP (probability) as stored", base["phi_rf_tree_shap"], 0.01)

# ============================================================== T3 leakage
if STAGE in ("leakage", "all"):
    log("\nT3 Leakage and sanity checks")
    for ds in ("framingham", "brfss"):
        d = load(ds)
        check(f"{ds}: train and test indices disjoint", len(d["Xtr"].index.intersection(d["Xte"].index)) == 0)
        check(f"{ds}: explained patients are a subset of the test set", d["Xex"].index.isin(d["Xte"].index).all())
        check(f"{ds}: background rows are a subset of the training set", d["Xbg"].index.isin(d["Xtr"].index).all())
        check(f"{ds}: no explained patient is in the training set",
              len(d["Xex"].index.intersection(d["Xtr"].index)) == 0)
    d = load("framingham")
    hps = joblib.load(OUT / "framingham" / "base_models.joblib")["hps"]
    ytr, yte = d["ytr"].to_numpy(), d["yte"].to_numpy()
    prep = Prep().fit(d["Xtr"])
    Xtr, Xte, Xex, Xbg = (prep.transform(d[k]) for k in ("Xtr", "Xte", "Xex", "Xbg"))
    base = np.load(OUT / "framingham" / "variants" / "base.npz")
    aucs = {m: [] for m in ("lr", "xgb", "rf", "mlp")}
    agree = {"xgb_tree_shap": [], "lr_linear_shap": [], "mlp_eg": []}
    for r in range(10):
        yp = np.random.default_rng(100 + r).permutation(ytr)
        for m in aucs:
            mdl = make_model(m, hps[m], SEED).fit(Xtr, yp)
            aucs[m].append(roc_auc_score(yte, mdl.margin(Xte)))
            for e in NATIVE[m]:
                if f"{m}_{e}" in agree:
                    phi = NATIVE[m][e](mdl, Xex, Xbg)
                    agree[f"{m}_{e}"].append(jaccard_topk(ranks_desc(phi), ranks_desc(base[f"phi_{m}_{e}"]), 3).mean())
    for m, v in aucs.items():
        check(f"label permutation: {m} test AUC falls to chance", abs(np.mean(v) - 0.5) < 0.04,
              f"mean {np.mean(v):.3f}, range {min(v):.3f}-{max(v):.3f} over 10 permutations")
    ch = expected_jaccard(15, 3)
    log(f"  Model-randomisation check: top-3 agreement between explanations of a model trained on")
    log(f"  shuffled labels and of the real model (chance {ch:.3f}; real seed-to-seed 0.82-0.90):")
    for k, v in agree.items():
        check(f"    {k}: shuffled-label explanations do not resemble the real ones", np.mean(v) < 0.45,
              f"mean {np.mean(v):.3f}")

# ============================================================== T4 determinism
if STAGE in ("determinism", "all"):
    log("\nT4 Determinism: refit Framingham base models and compare with stored attributions")
    d = load("framingham")
    hps = joblib.load(OUT / "framingham" / "base_models.joblib")["hps"]
    prep = Prep().fit(d["Xtr"])
    Xtr, Xex, Xbg = prep.transform(d["Xtr"]), prep.transform(d["Xex"]), prep.transform(d["Xbg"])
    base = np.load(OUT / "framingham" / "variants" / "base.npz")
    for m in ("lr", "rf", "xgb", "mlp"):
        mdl = make_model(m, hps[m], SEED).fit(Xtr, d["ytr"].to_numpy())
        dm = np.abs(mdl.margin(Xex) - base[f"margin_{m}"]).max()
        check(f"{m}: refit reproduces stored predictions", dm < 1e-4, f"max diff in log-odds {dm:.1e}")
        for e in NATIVE[m]:
            dp = np.abs(NATIVE[m][e](mdl, Xex, Xbg) - base[f"phi_{m}_{e}"]).max()
            check(f"{m} {e}: refit reproduces stored attributions", dp < 1e-4, f"max diff {dp:.1e}")
    rows = np.random.default_rng(1000 + 7).integers(0, len(d["ytr"]), len(d["ytr"]))
    pr = Prep().fit(d["Xtr"].iloc[rows])
    mdl = make_model("xgb", hps["xgb"], SEED).fit(pr.transform(d["Xtr"].iloc[rows]), d["ytr"].to_numpy()[rows])
    st = np.load(OUT / "framingham" / "variants" / "boot_007.npz")
    dp = np.abs(NATIVE["xgb"]["tree_shap"](mdl, pr.transform(d["Xex"]), pr.transform(d["Xbg"])) - st["phi_xgb_tree_shap"]).max()
    check("bootstrap variant boot_007 (XGB) reproduces from its seed", dp < 1e-4, f"max diff {dp:.1e}")

log(f"\nAudit 1 stage '{STAGE}': {'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
REPORT.close()
