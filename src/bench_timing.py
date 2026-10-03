"""Timing benchmark used to set the number of training variants (Decision Register D-06)."""
import sys
import time

import numpy as np

from common import NATIVE, Prep, load, make_model

ds = sys.argv[1]
d = load(ds)
prep = Prep().fit(d["Xtr"])
Xtr, Xex, Xbg = prep.transform(d["Xtr"]), prep.transform(d["Xex"]), prep.transform(d["Xbg"])
ytr = d["ytr"].to_numpy()
big = ds == "brfss"
hps = {
    "lr": dict(C=1.0),
    "rf": dict(n_estimators=300, min_samples_leaf=50 if big else 10, max_depth=12 if big else None),
    "xgb": dict(n_estimators=300, max_depth=3, learning_rate=0.05),
    "mlp": dict(epochs=20, batch=1024 if big else 64),
}
for name, hp in hps.items():
    t = time.time()
    m = make_model(name, hp, 0).fit(Xtr, ytr)
    tf = time.time() - t
    line = f"{ds:<11}{name:<5} fit {tf:6.1f}s"
    for ex_name, fn in NATIVE[name].items():
        t = time.time()
        phi = fn(m, Xex, Xbg)
        te = time.time() - t
        gap = np.abs(phi.sum(1)).mean()
        line += f" | {ex_name} {te:6.1f}s shape {phi.shape}"
    if name == "rf":
        leaves = np.mean([e.get_n_leaves() for e in m.est.estimators_])
        line += f" | mean leaves {leaves:.0f}"
    print(line, flush=True)
