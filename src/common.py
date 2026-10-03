"""Shared code for Phases 3-6: data, preprocessing, models, native explainers.

Every model is wrapped so that `margin(X)` returns the log-odds of the
positive class for standardised inputs X. Explanations are computed on the
standardised inputs; interventional Shapley values are unchanged by the
affine rescaling, so attributions refer to the original features.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
SEED = 42
N_EXPLAIN = 500
N_BACKGROUND = 100

DATASETS = {
    "framingham": dict(file="framingham.csv", target="TenYearCHD"),
    "brfss": dict(file="heart_disease_health_indicators_BRFSS2015.csv", target="HeartDiseaseorAttack"),
}
FRAM_BINARY = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]

torch.set_num_threads(4)


def load(dataset):
    """Fixed stratified 80/20 split, fixed explained set and background rows (raw values)."""
    cfg = DATASETS[dataset]
    df = pd.read_csv(DATA / cfg["file"])
    y = df[cfg["target"]].astype(int)
    X = df.drop(columns=cfg["target"]).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    _, Xex, _, yex = train_test_split(Xte, yte, test_size=N_EXPLAIN, stratify=yte, random_state=SEED)
    bg_idx = np.random.default_rng(SEED).choice(len(Xtr), N_BACKGROUND, replace=False)
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xex=Xex, yex=yex,
                Xbg=Xtr.iloc[bg_idx], features=list(X.columns))


class Prep:
    """Train-only imputation (median; mode for binary columns) and standardisation."""

    def fit(self, X):
        self.fill = X.median()
        for c in X.columns:
            if c in FRAM_BINARY:
                self.fill[c] = X[c].mode().iloc[0]
        Xi = X.fillna(self.fill)
        self.mu, self.sd = Xi.mean(), Xi.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X):
        return ((X.fillna(self.fill) - self.mu) / self.sd).to_numpy(np.float32)


# ----------------------------------------------------------------------- models
class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLP:
    """64-32 ReLU network (architecture of the draft paper). Fixed number of epochs."""

    def __init__(self, epochs, batch, seed, lr=5e-4, weight_decay=5e-3):
        self.epochs, self.batch, self.seed, self.lr, self.wd = epochs, batch, seed, lr, weight_decay

    def fit(self, X, y, X_val=None, y_val=None, patience=10):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.model = Net(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.BCEWithLogitsLoss()
        best, best_state, wait, self.best_epoch = np.inf, None, 0, self.epochs
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.model(Xt[idx]), yt[idx]).backward()
                opt.step()
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    v = lossf(self.model(torch.from_numpy(X_val)),
                              torch.from_numpy(np.asarray(y_val, dtype=np.float32))).item()
                if v < best - 1e-5:
                    best, wait, self.best_epoch = v, 0, ep + 1
                    best_state = {k: t.clone() for k, t in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def margin(self, X):
        with torch.no_grad():
            return self.model(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()


class SkWrap:
    def __init__(self, est, kind):
        self.est, self.kind = est, kind

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def margin(self, X):
        if self.kind == "lr":
            return self.est.decision_function(X)
        if self.kind == "xgb":
            return self.est.predict(X, output_margin=True)
        p = np.clip(self.est.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)     # rf
        return np.log(p / (1 - p))


def make_model(name, hp, seed):
    if name == "lr":
        return SkWrap(LogisticRegression(C=hp["C"], max_iter=5000), "lr")
    if name == "rf":
        return SkWrap(RandomForestClassifier(
            n_estimators=hp["n_estimators"], min_samples_leaf=hp["min_samples_leaf"],
            max_depth=hp["max_depth"], max_features="sqrt", n_jobs=-1, random_state=seed), "rf")
    if name == "xgb":
        return SkWrap(XGBClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
            subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
            random_state=seed, eval_metric="logloss"), "xgb")
    if name == "mlp":
        return MLP(epochs=hp["epochs"], batch=hp["batch"], seed=seed)
    raise ValueError(name)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------ native explainers
def explain_linear(model, X, Xbg):
    """Exact interventional Shapley values of a linear margin: w_j (x_j - mean_bg_j)."""
    w = model.est.coef_.ravel()
    return (X - Xbg.mean(axis=0)) * w


def explain_tree(model, X, Xbg):
    """Interventional TreeSHAP. XGBoost: log-odds margin. Random forest: probability
    (the scale on which a forest is additive; see Decision Register D-12)."""
    import shap
    ex = shap.TreeExplainer(model.est, data=Xbg, feature_perturbation="interventional",
                            model_output="raw")
    sv = ex.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def explain_ig(model, X, Xbg, steps=256):
    """Integrated Gradients from the background-mean baseline (Sundararajan et al. 2017)."""
    x = torch.from_numpy(np.asarray(X, dtype=np.float32))
    base = torch.from_numpy(Xbg.mean(axis=0, keepdims=True).astype(np.float32))
    total = torch.zeros_like(x)
    for a in (np.arange(steps) + 0.5) / steps:                      # midpoint rule
        pt = (base + float(a) * (x - base)).requires_grad_(True)
        total += torch.autograd.grad(model.model(pt).sum(), pt)[0]
    return ((x - base) * total / steps).numpy()


def explain_eg(model, X, Xbg, steps=16):
    """Expected Gradients (Erion et al. 2021): Integrated Gradients averaged over every
    background row as baseline. Converges to interventional Shapley-style attributions
    whose sum is f(x) - mean f(background)."""
    x = torch.from_numpy(np.asarray(X, dtype=np.float32))
    bg = torch.from_numpy(np.asarray(Xbg, dtype=np.float32))
    out = torch.zeros_like(x)
    for b in bg:
        total = torch.zeros_like(x)
        for a in (np.arange(steps) + 0.5) / steps:
            pt = (b + float(a) * (x - b)).requires_grad_(True)
            total += torch.autograd.grad(model.model(pt).sum(), pt)[0]
        out += (x - b) * total / steps
    return (out / len(bg)).numpy()


NATIVE = {"lr": {"linear_shap": explain_linear},
          "rf": {"tree_shap": explain_tree},
          "xgb": {"tree_shap": explain_tree},
          "mlp": {"ig": explain_ig, "eg": explain_eg}}
