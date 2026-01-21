# src/model.py
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# -----------------------------
# Builders
# -----------------------------
def build_model(name: str = "hgbt", random_state: int = 42) -> Pipeline:
    """
    Return a sklearn Pipeline that takes NUMERIC features only.
    - 'logreg': Imputer -> Scaler -> LogisticRegression(class_weight='balanced')
    - 'hgbt'  : Imputer -> HistGradientBoostingClassifier (sample_weight-balanced at fit time)
    """
    name = name.lower()
    if name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            C=0.5,
            class_weight="balanced",
            n_jobs=None,
            random_state=random_state
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly & safe for many cols
            ("clf", clf),
        ])
    elif name == "hgbt":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.0,
            scoring="average_precision",
            random_state=random_state,
        )
        # NOTE: HGBT doesn't take class_weight; we pass sample_weight at fit time (see train()).
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])
    elif name == "xgb":
        clf = xgb.XGBClassifier(
            learning_rate=0.05,
            n_estimators=1000,
            max_depth=6,  # Tunable
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="aucpr",  # For imbalance
            early_stopping_rounds=30,
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])
    else:
        raise ValueError(f"Unknown model name: {name}. Try 'hgbt' or 'logreg'.")


# -----------------------------
# Training
# -----------------------------
def train(model: Pipeline, X: pd.DataFrame, y: pd.Series, balance: bool = True) -> Pipeline:
    """
    Fit the pipeline. If balance=True and model is 'hgbt', pass sample_weight='balanced'.
    Works for both pipelines returned by build_model().
    """
    sw = None
    if balance:
        try:
            sw = compute_sample_weight("balanced", y)
        except Exception:
            sw = None
    try:
        model.fit(X, y, clf__sample_weight=sw)  # works if last step is named 'clf'
    except TypeError:
        # Pipelines without named param or estimators that don't accept sample_weight:
        if sw is not None:
            model.fit(X, y, sample_weight=sw)
        else:
            model.fit(X, y)
    return model


# -----------------------------
# Inference
# -----------------------------
def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability for positive class. Falls back to decision_function if needed.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # Min-max to 0..1 for comparability
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s
    # worst case, use predictions
    return model.predict(X).astype(float)


# -----------------------------
# Metrics
# -----------------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision among the top-k by score."""
    k = int(min(max(k, 1), len(y_score)))
    idx = np.argsort(-y_score)[:k]
    return float((y_true[idx] == 1).mean()) if k > 0 else np.nan

def evaluate(y_true: np.ndarray, y_score: np.ndarray, ks=None) -> Dict[str, float]:
    """
    Compute common ranking metrics. Handles single-class edge cases gracefully.
    """
    if ks is None:
        ks = [10, 20, 50]
    out: Dict[str, float] = {}
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # ROC-AUC (skip if single class)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = np.nan

    # Average Precision (PR-AUC)
    try:
        out["avg_precision"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["avg_precision"] = np.nan

    for k in ks:
        out[f"precision@{k}"] = precision_at_k(y_true, y_score, k)

    # base rate
    out["positive_rate"] = float(y_true.mean()) if len(y_true) else np.nan
    return out

def calibrate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, method: str = "sigmoid", cv: int = 5):
    """
    Calibrate the *entire pipeline* using CV on the validation split.
    - Avoids 'prefit' deprecation and 'feature names' warnings.
    - 'sigmoid' (Platt) is robust with few positives; 'isotonic' needs more data.
    - Returns a fitted CalibratedClassifierCV wrapping the pipeline.
    """
    if len(np.unique(y_val)) < 2:
        # Can't calibrate with a single class; just skip
        return None
    try:
        # Wrap the trained model in FrozenEstimator to replace cv="prefit"
        frozen = FrozenEstimator(model)
        cal = CalibratedClassifierCV(frozen, cv="prefit", method="sigmoid")
        cal.fit(X_val, y_val)
        return cal
    except Exception:
        return None

def compute_shap_values(model: Pipeline, X: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer (works for XGB/HGBT).
    Returns shap_values (array) and explainer.
    """
    clf = model.named_steps['clf'] if 'clf' in model.named_steps else model
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer

def get_top_shap_explanations(shap_values: np.ndarray, X: pd.DataFrame, player_index: int, top_n: int = 5):
    """
    Get top contributing features for a specific player.
    Returns dict of feature: contribution.
    """
    player_shap = shap_values[player_index]
    top_indices = np.argsort(np.abs(player_shap))[-top_n:]
    explanations = {X.columns[i]: player_shap[i] for i in top_indices}
    return dict(sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True))