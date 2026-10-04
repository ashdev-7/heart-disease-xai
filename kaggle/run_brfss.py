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
hits = glob.glob("/kaggle/input/**/src/common.py", recursive=True)
assert hits, "bundle dataset not attached"
BUNDLE = os.path.dirname(os.path.dirname(hits[0]))
print("bundle:", BUNDLE, flush=True)


def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit {r.returncode}, {(time.time() - t) / 60:.1f} min]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)


sh(f"{sys.executable} -m pip install --quiet " + " ".join(PINS), check=False)
shutil.copytree(f"{BUNDLE}/src", f"{WORK}/src", dirs_exist_ok=True)
shutil.copytree(f"{BUNDLE}/seed_results", f"{WORK}/results", dirs_exist_ok=True)
os.environ.update(XAI_DATA=f"{BUNDLE}/data", XAI_OUT=f"{WORK}/results", XAI_REUSE_HP="1")
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
