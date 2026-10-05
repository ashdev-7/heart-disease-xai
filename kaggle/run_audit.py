"""Kaggle runner for the remaining audit stages (laptop memory too small).

  A2  hyperparameters re-tuned inside every bootstrap, plus the tuned-once reference
      computed in the same environment (Framingham)
  A6  second, independently drawn subsamples for the BRFSS sample-size experiment
Writes /kaggle/working/audit_results.zip
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
    assert hits, f"{pattern} not found"
    return hits[0]


def sh(cmd, check=False):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}, {(time.time() - t) / 60:.1f} min]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)


SRC = os.path.dirname(find("common.py"))
DATA_DIR = os.path.dirname(find("framingham.csv"))
SEEDRES = os.path.dirname(os.path.dirname(find("framingham/hyperparameters.json")))
sh(f"{sys.executable} -m pip install --quiet " + " ".join(PINS))
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
shutil.copytree(SEEDRES, f"{WORK}/results", dirs_exist_ok=True)
os.environ.update(XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results", XAI_BUDGET="1000000")
sh(f"{sys.executable} -c \"import sklearn,xgboost,shap,numpy,torch,sys;"
   "print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,"
   "'shap',shap.__version__,'numpy',numpy.__version__,'torch',torch.__version__)\"", check=True)
os.chdir(WORK)
py = f"{sys.executable} -u"


def pack():
    stage = f"{WORK}/pack"
    shutil.rmtree(stage, ignore_errors=True)
    for src, dst in ((f"{WORK}/results/framingham/audit", f"{stage}/framingham_audit"),
                     (f"{WORK}/results/brfss/sample_size", f"{stage}/brfss_sample_size")):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
    shutil.make_archive(f"{WORK}/audit_results", "zip", stage)
    shutil.rmtree(stage, ignore_errors=True)


sh(f"{py} src/audit3_alternatives.py fixedref 2> fixedref.err"); pack()
sh(f"{py} src/audit3_alternatives.py retune 2> retune.err"); pack()
for n, seed in ((3392, 7), (7580, 7), (30000, 7), (3392, 11), (7580, 11)):
    sh(f"{py} src/phase6_sample_size.py run {n} {seed} 2> ss_{n}_{seed}.err"); pack()
shutil.rmtree(f"{WORK}/results", ignore_errors=True)
print("done", flush=True)
