"""Kaggle runner: Audit 4 settings checks on BRFSS against the main-run (seed 42) explanations."""
import glob, os, shutil, subprocess, sys
PINS = ["scikit-learn==1.8.0", "xgboost==3.1.3", "shap==0.52.0", "numpy==2.2.6",
        "pandas==2.3.2", "scipy==1.16.3", "statsmodels==0.14.6"]
WORK = "/kaggle/working"
find = lambda p: glob.glob(f"/kaggle/input/**/{p}", recursive=True)[0]
SRC = os.path.dirname(find("common.py"))
DATA_DIR = os.path.dirname(find("framingham.csv"))
SEEDRES = os.path.dirname(os.path.dirname(find("multiplicity_per_patient.npz")))
subprocess.run(f"{sys.executable} -m pip install --quiet " + " ".join(PINS), shell=True)
shutil.copytree(SRC, f"{WORK}/src", dirs_exist_ok=True)
shutil.copytree(SEEDRES, f"{WORK}/results", dirs_exist_ok=True)
os.environ.update(XAI_DATA=DATA_DIR, XAI_OUT=f"{WORK}/results")
os.chdir(WORK)
r = subprocess.run(f"{sys.executable} -u src/audit4_settings.py brfss 2> audit4.err", shell=True)
shutil.copy(f"{WORK}/results/brfss/audit4_report.txt", f"{WORK}/audit4_brfss_report.txt")
shutil.rmtree(f"{WORK}/results", ignore_errors=True)
print("exit", r.returncode)
