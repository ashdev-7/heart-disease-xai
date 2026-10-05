"""Provenance check: rebuild the BRFSS 2015 heart-disease file from the CDC raw file and
compare it, row for row, with the public file used in the study.

The cleaning steps are those documented by the file's author (A. Teboul, Kaggle notebook
"Heart Disease Health Indicators Dataset Notebook"), re-implemented here independently.
Writes cdc_check_report.txt
"""
import glob

import numpy as np
import pandas as pd

raw_path = [p for p in glob.glob("/kaggle/input/**/2015.csv", recursive=True)][0]
pub_path = glob.glob("/kaggle/input/**/heart_disease_health_indicators_BRFSS2015.csv", recursive=True)[0]
COLS = ["_MICHD", "_RFHYPE5", "TOLDHI2", "_CHOLCHK", "_BMI5", "SMOKE100", "CVDSTRK3", "DIABETE3", "_TOTINDA",
        "_FRTLT1", "_VEGLT1", "_RFDRHV5", "HLTHPLN1", "MEDCOST", "GENHLTH", "MENTHLTH", "PHYSHLTH", "DIFFWALK",
        "SEX", "_AGEG5YR", "EDUCA", "INCOME2"]
out = []


def log(s=""):
    print(s, flush=True)
    out.append(str(s))


raw = pd.read_csv(raw_path, usecols=COLS)[COLS]
log(f"CDC raw file: {raw_path}")
log(f"raw rows {len(raw)}; columns selected {raw.shape[1]}")
df = raw.dropna().copy()
log(f"after dropping rows with any missing value: {len(df)}")
# (column, recodes, codes removed)
STEPS = [
    ("_MICHD", {2: 0}, []), ("_RFHYPE5", {1: 0, 2: 1}, [9]), ("TOLDHI2", {2: 0}, [7, 9]),
    ("_CHOLCHK", {3: 0, 2: 0}, [9]), ("SMOKE100", {2: 0}, [7, 9]), ("CVDSTRK3", {2: 0}, [7, 9]),
    ("DIABETE3", {2: 0, 3: 0, 1: 2, 4: 1}, [7, 9]), ("_TOTINDA", {2: 0}, [9]), ("_FRTLT1", {2: 0}, [9]),
    ("_VEGLT1", {2: 0}, [9]), ("_RFDRHV5", {1: 0, 2: 1}, [9]), ("HLTHPLN1", {2: 0}, [7, 9]),
    ("MEDCOST", {2: 0}, [7, 9]), ("GENHLTH", {}, [7, 9]), ("MENTHLTH", {88: 0}, [77, 99]),
    ("PHYSHLTH", {88: 0}, [77, 99]), ("DIFFWALK", {2: 0}, [7, 9]), ("SEX", {2: 0}, []),
    ("_AGEG5YR", {}, [14]), ("EDUCA", {}, [9]), ("INCOME2", {}, [77, 99]),
]
df["_BMI5"] = (df["_BMI5"] / 100).round(0)
for col, rec, drop in STEPS:
    df = df[~df[col].isin(drop)]
    if rec:
        df[col] = df[col].replace(rec)   # simultaneous mapping
log(f"after recoding and removing 'do not know' / 'refused' codes: {len(df)}")

pub = pd.read_csv(pub_path)
log(f"public file: rows {len(pub)}, columns {pub.shape[1]}: {list(pub.columns)}")
df.columns = pub.columns
a = df.astype(float).sort_values(list(pub.columns)).reset_index(drop=True)
b = pub.astype(float).sort_values(list(pub.columns)).reset_index(drop=True)
same_shape = a.shape == b.shape
log(f"same shape: {same_shape}")
if same_shape:
    eq = (a.to_numpy() == b.to_numpy())
    log(f"rows identical after sorting: {int(eq.all(1).sum())} of {len(a)}; cells differing: {int((~eq).sum())}")
log("\nPer-column means (rebuilt | public):")
for c in pub.columns:
    log(f"  {c:<22} {df[c].astype(float).mean():.6f} | {pub[c].astype(float).mean():.6f}")
log(f"\noutcome cases: rebuilt {int(df.iloc[:, 0].sum())} | public {int(pub.iloc[:, 0].sum())}")
log(f"duplicate rows: rebuilt {int(a.duplicated().sum())} | public {int(b.duplicated().sum())}")
open("/kaggle/working/cdc_check_report.txt", "w").write("\n".join(out))
