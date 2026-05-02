"""
fraud_model.py
--------------
Trains, evaluates, and persists the XGBoost-based fraud detection model.

Pipeline:
  1. Load processed feature parquet
  2. Temporal train/test split
  3. Handle class imbalance (SMOTE + scale_pos_weight)
  4. Hyperparameter tuning (cross-validation)
  5. Final model training
  6. Evaluation: AUC-ROC, Precision, Recall, F1, PR curve
  7. SHAP explainability
  8. Model persistence (joblib + metadata JSON)

Usage:
    python src/models/fraud_model.py --data data/processed/features_dev.parquet
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("fraud_model")

MODEL_DIR = "models/saved"
REPORT_DIR = "models/reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Columns NOT used as model features
ID_COLS = ["claim_id", "member_id", "provider_npi"]
TARGET_COL = "fraud_label"
DROP_COLS = ID_COLS + [TARGET_COL]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SPLITTING
# ─────────────────────────────────────────────────────────────────────────────


def load_features(path: str) -> pd.DataFrame:
    log.info(f"Loading features from {path}")
    df = pd.read_parquet(path)
    log.info(f"  Shape: {df.shape} | Fraud rate: {df[TARGET_COL].mean()*100:.2f}%")
    return df


def temporal_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Temporal split: earlier claims → train, most recent → test.
    Prevents data leakage from future events.
    """
    if "claim_month" in df.columns and "claim_year" in df.columns:
        df["_sort_key"] = df["claim_year"] * 100 + df["claim_month"]
        df = df.sort_values("_sort_key")
        df = df.drop(columns=["_sort_key"])
    cutoff = int(len(df) * (1 - test_frac))
    train, test = df.iloc[:cutoff], df.iloc[cutoff:]
    log.info(f"Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def prepare_xy(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].values
    return X, y, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────


def get_model_params(scale_pos_weight: float) -> dict:
    """Return XGBoost hyperparameters (pre-tuned via Bayesian optimization)."""
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def train_model(
    X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray
) -> xgb.XGBClassifier:
    """Train XGBoost with early stopping on validation AUC-PR."""

    # Handle class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info(
        f"Class balance — Neg: {n_neg:,} | Pos: {n_pos:,} | scale_pos_weight: {scale_pos_weight:.1f}"
    )

    # Optionally apply SMOTE for severe imbalance
    if HAS_IMBLEARN and (scale_pos_weight > 20):
        log.info("Applying SMOTE for severe class imbalance...")
        smote = SMOTE(sampling_strategy=0.15, random_state=42)
        X_train_arr, y_train = smote.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        log.info(f"Post-SMOTE — Pos: {y_train.sum():,} | Total: {len(y_train):,}")

    params = get_model_params(scale_pos_weight)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train_imp,
        y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=50,
    )
    log.info(f"Best iteration: {model.best_iteration}")
    return model, imputer


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    model, imputer, X_test: pd.DataFrame, y_test: np.ndarray, threshold: float = 0.5
) -> dict:
    X_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    y_proba = model.predict_proba(X_imp)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fp / max(fp + tn, 1), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "threshold": threshold,
    }

    log.info("=" * 55)
    log.info("MODEL EVALUATION RESULTS")
    log.info("=" * 55)
    for k, v in metrics.items():
        log.info(f"  {k:<30} {v}")
    log.info("=" * 55)
    log.info(
        "\n" + classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    )

    return metrics, y_proba


def find_optimal_threshold(
    y_test: np.ndarray, y_proba: np.ndarray, target_fpr: float = 0.08
) -> float:
    """Find threshold that achieves target false-positive rate."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    for threshold in sorted(thresholds, reverse=True):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / max(fp + tn, 1)
        if fpr <= target_fpr:
            log.info(f"Optimal threshold: {threshold:.4f} (FPR={fpr:.4f})")
            return threshold
    return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────


def generate_shap(model, imputer, X_sample: pd.DataFrame, n_samples: int = 500) -> None:
    """Generate and save SHAP summary plot."""
    if not HAS_SHAP:
        log.warning("SHAP not available — skipping explainability plots")
        return

    log.info("Generating SHAP explanations...")
    X_imp = pd.DataFrame(
        imputer.transform(X_sample.head(n_samples)), columns=X_sample.columns
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_imp, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/shap_feature_importance.png", dpi=150)
    plt.close()
    log.info(f"SHAP plot saved → {REPORT_DIR}/shap_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────


def cross_validate(X: pd.DataFrame, y: np.ndarray, n_folds: int = 5) -> dict:
    log.info(f"Running {n_folds}-fold stratified cross-validation...")
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    params = get_model_params(n_neg / max(n_pos, 1))
    params.pop("early_stopping_rounds", None)

    cv_model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    auc_scores = cross_val_score(
        cv_model, X_imp, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    log.info(f"CV AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    return {
        "cv_auc_mean": round(auc_scores.mean(), 4),
        "cv_auc_std": round(auc_scores.std(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────


def save_model(model, imputer, metrics: dict, feature_cols: list, version: str) -> str:
    """Persist model artifacts and metadata."""
    artifact = {
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
    }
    model_path = f"{MODEL_DIR}/fraud_model_{version}.joblib"
    joblib.dump(artifact, model_path)

    metadata = {
        "version": version,
        "trained_at": datetime.now().isoformat(),
        "model_type": "XGBClassifier",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_path": model_path,
    }
    meta_path = f"{MODEL_DIR}/fraud_model_{version}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            default=lambda o: float(o) if hasattr(o, "item") else str(o),
        )

    log.info(f"Model saved → {model_path}")
    log.info(f"Metadata    → {meta_path}")
    return model_path


def load_model(version: str = "latest"):
    """Load model artifact from disk."""
    if version == "latest":
        files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
        if not files:
            raise FileNotFoundError("No saved models found")
        version_tag = (
            sorted(files)[-1].replace("fraud_model_", "").replace(".joblib", "")
        )
    else:
        version_tag = version

    path = f"{MODEL_DIR}/fraud_model_{version_tag}.joblib"
    artifact = joblib.load(path)
    log.info(f"Model loaded from {path}")
    return artifact["model"], artifact["imputer"], artifact["feature_cols"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main(data_path: str, run_cv: bool = True):
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"=== Fraud Model Training — Version {version} ===")

    # 1. Load
    df = load_features(data_path)

    # 2. Split
    train_df, test_df = temporal_split(df, test_frac=0.20)
    train_df, val_df = temporal_split(train_df, test_frac=0.15)

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_val, y_val, _ = prepare_xy(val_df)
    X_test, y_test, _ = prepare_xy(test_df)

    # 3. Cross-validation
    cv_metrics = {}
    if run_cv:
        X_cv = pd.concat([X_train, X_val])
        y_cv = np.concatenate([y_train, y_val])
        cv_metrics = cross_validate(X_cv, y_cv)

    # 4. Train
    model, imputer = train_model(X_train, y_train, X_val, y_val)

    # 5. Find optimal threshold
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    y_val_proba = model.predict_proba(X_val_imp)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, y_val_proba, target_fpr=0.08)

    # 6. Evaluate on test set
    metrics, y_test_proba = evaluate(
        model, imputer, X_test, y_test, threshold=optimal_threshold
    )
    metrics.update(cv_metrics)

    # 7. SHAP
    generate_shap(model, imputer, X_test)

    # 8. Save
    save_model(model, imputer, metrics, feature_cols, version)

    log.info(
        f"=== Training complete. AUC-ROC: {metrics['auc_roc']} | F1: {metrics['f1']} ==="
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", default="data/processed/features_dev.parquet")
    parser.add_argument("--cv", action="store_true", default=True)
    args = parser.parse_args()
    main(data_path=args.data, run_cv=args.cv)
