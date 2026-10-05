"""Write one Kaggle kernel folder per job (kaggle/kernel_<name>/). Usage: python kaggle/make_jobs.py <user>"""
import json
import sys
from pathlib import Path

user = sys.argv[1]
k = Path(__file__).resolve().parent
tpl = (k / "run_jobs.py").read_text()


def rep(seed, ds="framingham"):
    e = {"XAI_SEED": str(seed), "XAI_OUT": f"/kaggle/working/results_s{seed}"}
    t = f"s{seed}"
    cmds = [f"PY src/phase2_reference.py > {t}_phase2.log 2> {t}_phase2.err",
            f"PY src/phase3_models.py {ds} base 2> {t}_base.err",
            f"PY src/phase3_models.py {ds} variants 2> {t}_variants.err",
            f"PY src/phase5_multiplicity.py {ds} > {t}_mult.log 2> {t}_mult.err"]
    cmds += [f"PY src/phase4_base_axes.py {ds} {s} > {t}_{s}.log 2> {t}_{s}.err" for s in ("faith", "noise", "floor", "kernel")]
    if ds == "framingham":
        cmds.append(f"PY src/phase5_plausibility.py > {t}_plaus.log 2> {t}_plaus.err")
    cmds += [f"PY src/phase6_variance.py {ds} {s} > {t}_var_{s}.log 2> {t}_var_{s}.err" for s in ("noise", "background", "report")]
    cmds += [f"PY src/phase6_shares_linking.py > {t}_link.log 2> {t}_link.err",
             f"PY src/phase6_smote_arm.py {ds} > {t}_smote.log 2> {t}_smote.err"]
    return [(e, c) for c in cmds]


def ss(n, seeds):
    return [({}, f"PY src/phase6_sample_size.py run {n} {s} 2> ss_{n}_{s}.err") for s in seeds]


B = {"XAI_AUDIT_DS": "brfss"}
JOBS = {
    "rep-fram-a": (rep(11) + rep(23), False),
    "rep-fram-b": (rep(101) + rep(42), False),
    "ss-3392": (ss(3392, (13, 17, 19, 23, 29, 31, 37)), False),
    "ss-7580": (ss(7580, (13, 17, 19, 23, 29, 31, 37)), False),
    "ss-30000": (ss(30000, (11, 13, 17)), False),
    "audit6-brfss": ([({}, "PY src/audit6_hyperparameters.py brfss 2> audit6.err")], True),
    "alt-brfss-a": ([(dict(B, XAI_NRES="20"), f"PY src/audit3_alternatives.py {s} 2> {s}.err")
                     for s in ("fixedref", "pathdep", "subsample")], True),
    "alt-brfss-b": ([(dict(B, XAI_NRES="10"), "PY src/audit3_alternatives.py retune_grouped lr,xgb 2> rt_lrxgb.err"),
                     (dict(B, XAI_NRES="10"), "PY src/audit3_alternatives.py retune_grouped rf 2> rt_rf.err")], True),
}
for name, (steps, copy) in JOBS.items():
    kd = k / f"kernel_{name.replace('-', '_')}"
    kd.mkdir(exist_ok=True)
    code = tpl.replace("__NAME__", name).replace("__STEPS__", repr(steps)).replace("__COPY__", repr(copy))
    (kd / "run_jobs.py").write_text(code)
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/xai-heart-{name}", "title": f"xai-heart-{name}", "code_file": "run_jobs.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
        "enable_internet": True, "dataset_sources": [f"{user}/xai-heart-bundle"]}, indent=1))
    print(kd.name, len(steps), "steps")
