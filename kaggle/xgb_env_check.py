"""Does XGBoost give the same fit on two platforms? With and without row/column subsampling."""
import glob, hashlib, os, platform, subprocess, sys
if os.path.isdir("/kaggle/input"):
    subprocess.run(f"{sys.executable} -m pip install --quiet xgboost==3.1.3 scikit-learn==1.8.0 numpy==2.2.6 pandas==2.3.2", shell=True)
    path = glob.glob("/kaggle/input/**/framingham.csv", recursive=True)[0]
else:
    path = "data/framingham.csv"
import numpy as np, pandas as pd, xgboost
from sklearn.model_selection import train_test_split
df = pd.read_csv(path)
y, X = df.TenYearCHD.astype(int), df.drop(columns="TenYearCHD").astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
print(platform.platform(), "xgboost", xgboost.__version__, "threads", os.cpu_count())
for sub, nj in ((1.0, -1), (0.8, -1), (0.8, 1)):
    m = xgboost.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=sub, colsample_bytree=sub,
                              tree_method="hist", n_jobs=nj, random_state=42).fit(Xtr, ytr)
    p = m.predict(Xte, output_margin=True).astype(np.float64)
    print(f"subsample {sub} n_jobs {nj}: sum of margins {p.sum():.6f}  md5 of rounded margins "
          f"{hashlib.md5(np.round(p, 4).tobytes()).hexdigest()[:12]}")
