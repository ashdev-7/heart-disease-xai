"""Build the folder that is uploaded to Kaggle as one private dataset.

  python kaggle/build_bundle.py <kaggle-username>

Creates kaggle/bundle/ (code, the two CSVs, tuned BRFSS hyperparameters) with
dataset-metadata.json, and kaggle/kernel/ with the runner and kernel-metadata.json.
"""
import json
import shutil
import sys
from pathlib import Path

user = sys.argv[1]
root = Path(__file__).resolve().parents[1]
k = root / "kaggle"
b = k / "bundle"
shutil.rmtree(b, ignore_errors=True)
(b / "seed_results" / "brfss").mkdir(parents=True)
shutil.copytree(root / "src", b / "src", ignore=shutil.ignore_patterns("__pycache__"))
(b / "data").mkdir()
for f in (root / "data").glob("*.csv"):
    shutil.copy(f, b / "data" / f.name)
shutil.copy(root / "results" / "brfss" / "hyperparameters.json", b / "seed_results" / "brfss")
(b / "dataset-metadata.json").write_text(json.dumps({
    "title": "xai-heart-bundle", "id": f"{user}/xai-heart-bundle",
    "licenses": [{"name": "other"}]}, indent=2))

kern = k / "kernel"
kern.mkdir(exist_ok=True)
shutil.copy(k / "run_brfss.py", kern / "run_brfss.py")
(kern / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/xai-heart-brfss-run", "title": "xai-heart-brfss-run",
    "code_file": "run_brfss.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": False, "enable_internet": True,
    "dataset_sources": [f"{user}/xai-heart-bundle"]}, indent=2))
print("bundle:", b, "| kernel:", kern)
