"""Kaggle runner: BRFSS base models refitted, then KernelSHAP without shap's default
feature truncation (Decision Register D-32), plus the stages that depend on it."""
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
SEEDRES = os.path.dirname(os.path.dirname(find("brfss/hyperparameters.json")))
sh(f"{sys.executable} -m pip install --quiet " + " ".join(PINS))
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
shutil.copytree(SEEDRES, f"{WORK}/results", dirs_exist_ok=True)
os.environ.update(XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results", XAI_REUSE_HP="1")
sh(f"{sys.executable} -c \"import sklearn,xgboost,shap,numpy,torch,sys;"
   "print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,"
   "'shap',shap.__version__,'numpy',numpy.__version__,'torch',torch.__version__)\"", check=True)
os.chdir(WORK)
py = f"{sys.executable} -u"
sh(f"{py} src/phase3_models.py brfss base 2> base.err", check=True)
for stage in ("kernel", "faith", "floor"):
    sh(f"{py} src/phase4_base_axes.py brfss {stage} 2> {stage}.err")
os.remove(f"{WORK}/results/brfss/base_models.joblib")
shutil.make_archive(f"{WORK}/kernel_fix_results", "zip", f"{WORK}/results/brfss")
shutil.rmtree(f"{WORK}/results", ignore_errors=True)
print("done", flush=True)
