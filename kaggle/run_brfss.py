"""Kaggle runner: all BRFSS stages on Kaggle hardware (more RAM than the laptop).

Expects one attached private dataset (the bundle built by kaggle/build_bundle.py)
containing src/, data/ and seed_results/. Writes /kaggle/working/brfss_results.zip.
Pinned library versions match the laptop run that produced the Framingham results.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

PINS = ["scikit-learn==1.8.0", "xgboost==3.1.3", "shap==0.52.0", "numpy==2.2.6",
        "pandas==2.3.2", "scipy==1.16.3", "statsmodels==0.14.6"]
WORK = "/kaggle/working"


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found in attached datasets"
    return hits[0]


SRC = os.path.dirname(find("common.py"))
DATA_DIR = os.path.dirname(find("heart_disease_health_indicators_BRFSS2015.csv"))
HP = find("hyperparameters.json")
print("src:", SRC, "| data:", DATA_DIR, "| hp:", HP, flush=True)


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}, {(time.time() - t) / 60:.1f} min]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)


sh(f"{sys.executable} -m pip install --quiet " + " ".join(PINS), check=False)
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
os.makedirs(f"{WORK}/results/brfss", exist_ok=True)
shutil.copy(HP, f"{WORK}/results/brfss/hyperparameters.json")
os.environ.update(XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results", XAI_REUSE_HP="1")
sh(f"{sys.executable} -c \"import sklearn,xgboost,shap,numpy,torch,sys;"
   "print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,"
   "'shap',shap.__version__,'numpy',numpy.__version__,'torch',torch.__version__)\"")

os.chdir(WORK)
py = f"{sys.executable} -u"
sh(f"{py} src/phase3_models.py brfss base 2> base.err")
sh(f"{py} src/phase3_models.py brfss variants 2> variants.err")
sh(f"{py} src/phase5_multiplicity.py brfss", check=False)
for stage in ("kernel", "faith", "noise", "floor"):
    sh(f"{py} src/phase4_base_axes.py brfss {stage} 2> {stage}.err", check=False)
    shutil.make_archive(f"{WORK}/brfss_results", "zip", f"{WORK}/results/brfss")   # checkpoint
shutil.make_archive(f"{WORK}/brfss_results", "zip", f"{WORK}/results/brfss")
print("done", flush=True)
