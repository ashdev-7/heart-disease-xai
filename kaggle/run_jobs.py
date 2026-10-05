"""Generic Kaggle runner. NAME and STEPS are filled in by kaggle/make_jobs.py.

Each step is (environment overrides, command). Results folders are zipped to
out_<NAME>.zip after every step (model files excluded), so a timeout loses one step only.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

NAME = "__NAME__"
STEPS = __STEPS__
COPY_SEED_RESULTS = __COPY__
PINS = ["scikit-learn==1.8.0", "xgboost==3.1.3", "shap==0.52.0", "numpy==2.2.6",
        "pandas==2.3.2", "scipy==1.16.3", "statsmodels==0.14.6", "imbalanced-learn==0.14.2"]
WORK = "/kaggle/working"


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    assert hits, f"{pattern} not found"
    return hits[0]


def pack():
    stage = f"{WORK}/pack"
    shutil.rmtree(stage, ignore_errors=True)
    for src in glob.glob(f"{WORK}/results*"):
        if os.path.isdir(src):
            shutil.copytree(src, f"{stage}/{os.path.basename(src)}", ignore=shutil.ignore_patterns("*.joblib"))
    for f in glob.glob(f"{WORK}/*.log") + glob.glob(f"{WORK}/*.err"):
        os.makedirs(f"{stage}/logs", exist_ok=True)
        shutil.copy(f, f"{stage}/logs")
    if os.path.isdir(stage):
        shutil.make_archive(f"{WORK}/out_{NAME}", "zip", stage)
        shutil.rmtree(stage, ignore_errors=True)


SRC = os.path.dirname(find("common.py"))
DATA_DIR = os.path.dirname(find("framingham.csv"))
subprocess.run(f"{sys.executable} -m pip install --quiet " + " ".join(PINS), shell=True)
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
os.makedirs(f"{WORK}/results", exist_ok=True)
if COPY_SEED_RESULTS:
    shutil.copytree(os.path.dirname(os.path.dirname(find("framingham/hyperparameters.json"))),
                    f"{WORK}/results", dirs_exist_ok=True)
os.chdir(WORK)
BASE = dict(os.environ, XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results", XAI_BUDGET="1000000")
subprocess.run(f"{sys.executable} -c \"import sklearn,xgboost,shap,numpy,torch,sys;"
               "print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'xgboost',xgboost.__version__,"
               "'shap',shap.__version__,'numpy',numpy.__version__,'torch',torch.__version__)\"", shell=True)
for i, (env, cmd) in enumerate(STEPS):
    cmd = cmd.replace("PY", f"{sys.executable} -u")
    print(f"\n$ [{i + 1}/{len(STEPS)}] {env} {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True, env=dict(BASE, **env))
    print(f"[exit {r.returncode}, {(time.time() - t) / 60:.1f} min]", flush=True)
    pack()
for src in glob.glob(f"{WORK}/results*"):
    shutil.rmtree(src, ignore_errors=True)
shutil.rmtree(f"{WORK}/src", ignore_errors=True)
print("done", flush=True)
