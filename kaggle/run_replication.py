"""Kaggle runner: full replication of the pipeline for one dataset under a different
master seed (new train/test split, new explained patients, new background rows,
hyperparameters re-tuned). Second-degree check on every headline result.

DATASET and MASTER_SEED are set at the top. Writes replication_<dataset>.zip.

This is the runner used for the seed-7 replication (results/replication_seed7). It ran
KernelSHAP last, so the plausibility and variance-report stages, which need its output,
failed on Kaggle and were re-run locally on the downloaded results. Later replications
use kaggle/make_jobs.py, which runs the stages in dependency order.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

DATASET = "__DATASET__"
MASTER_SEED = "7"
PINS = ["scikit-learn==1.8.0", "xgboost==3.1.3", "shap==0.52.0", "numpy==2.2.6",
        "pandas==2.3.2", "scipy==1.16.3", "statsmodels==0.14.6", "imbalanced-learn==0.14.2"]
WORK = "/kaggle/working"


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found"
    return hits[0]


def sh(cmd, check=False):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}, {(time.time() - t) / 60:.1f} min]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    pack()


def pack():
    src = f"{WORK}/results"
    if not os.path.isdir(src):
        return
    stage = f"{WORK}/pack"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(src, stage, ignore=shutil.ignore_patterns("*.joblib"))
    shutil.make_archive(f"{WORK}/replication_{DATASET}", "zip", stage)
    shutil.rmtree(stage, ignore_errors=True)


SRC = os.path.dirname(find("common.py"))
DATA_DIR = os.path.dirname(find("framingham.csv"))
subprocess.run(f"{sys.executable} -m pip install --quiet " + " ".join(PINS), shell=True)
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
os.makedirs(f"{WORK}/results", exist_ok=True)
os.environ.update(XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results", XAI_SEED=MASTER_SEED)
sh(f"{sys.executable} -c \"import sklearn,xgboost,shap,numpy,torch,imblearn,sys;"
   "print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,"
   "'shap',shap.__version__,'numpy',numpy.__version__,'torch',torch.__version__,'imblearn',imblearn.__version__)\"",
   check=True)
os.chdir(WORK)
py = f"{sys.executable} -u"
D = DATASET

sh(f"{py} src/phase2_reference.py > phase2.log 2> phase2.err", check=True)
sh(f"{py} src/phase3_models.py {D} base 2> base.err", check=True)
sh(f"{py} src/phase3_models.py {D} variants 2> variants.err", check=True)
sh(f"{py} src/phase5_multiplicity.py {D} > multiplicity.log 2> multiplicity.err")
for stage in ("faith", "noise", "floor"):
    sh(f"{py} src/phase4_base_axes.py {D} {stage} > {stage}.log 2> {stage}.err")
if D == "framingham":
    sh(f"{py} src/phase5_plausibility.py > plausibility.log 2> plausibility.err")
for stage in ("noise", "background", "report"):
    sh(f"{py} src/phase6_variance.py {D} {stage} > variance_{stage}.log 2> variance_{stage}.err")
sh(f"{py} src/phase6_shares_linking.py > linking.log 2> linking.err")
sh(f"{py} src/phase6_smote_arm.py {D} > smote.log 2> smote.err")
# KernelSHAP is the slowest stage, so it runs last; class shares need its output
sh(f"{py} src/phase4_base_axes.py {D} kernel > kernel.log 2> kernel.err")
sh(f"{py} src/phase6_shares_linking.py > linking2.log 2> linking2.err")
shutil.rmtree(f"{WORK}/results", ignore_errors=True)
print("done", flush=True)
