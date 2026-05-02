"""
claim_approval_model.py
-----------------------
Supervised ML model for automated claim approval / risk scoring.

Models:
  - Logistic Regression   (baseline)
  - Random Forest         (interpretable ensemble)
  - XGBoost               (production champion)

Output per claim:
  - approval_probability  (0.0 – 1.0)
  - risk_band             (LOW / MEDIUM / HIGH / VERY_HIGH)
  - top_risk_factors      (SHAP top-3 features)
  - recommendation        (AUTO_APPROVE / REVIEW / DENY)
"""

import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

log = logging.getLogger(__name__)

MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# RISK BANDING
# ─────────────────────────────────────────────────────────────────────────────


def get_risk_band(proba: float) -> str:
    if proba < 0.20:
        return "LOW"
    if proba < 0.50:
        return "MEDIUM"
    if proba < 0.80:
        return "HIGH"
    return "VERY_HIGH"


def get_recommendation(risk_band: str, fraud_score: float = 0.0) -> str:
    if fraud_score > 0.70:
        return "FLAG_FOR_FRAUD_REVIEW"
    if risk_band == "LOW":
        return "AUTO_APPROVE"
    if risk_band in ("MEDIUM",):
        return "EXPEDITED_REVIEW"
    if risk_band == "HIGH":
        return "FULL_REVIEW"
    return "DENY_PENDING_INVESTIGATION"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────


class ClaimApprovalModel:
    """
    Champion/challenger claim approval classifier.
    Default champion: XGBoost.
    """

    FEATURE_COLS = [
        "age",
        "chronic_conditions",
        "prior_claims_12m",
        "billed_amount",
        "n_procedures",
        "has_high_risk_cpt",
        "out_of_network",
        "prior_auth",
        "oig_excluded",
        "peer_billing_percentile",
        "specialty_risk_score",
        "plan_risk_score",
        "age_risk",
        "duplicate_flag",
        "allowed_ratio",
        "billed_per_procedure",
        "is_weekend",
        "suspicious_lag",
        "high_controlled_rx",
        "high_prior_claims",
    ]

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.explainer = None
        self._is_fitted = False

    def _build_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "logistic":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=42,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, df: pd.DataFrame, target_col: str = "fraud_label") -> dict:
        """Train model. Returns validation metrics."""
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available]
        # Invert fraud label: 1 = approved (not fraud), 0 = problem claim
        y = (df[target_col] == 0).astype(int)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train_imp = pd.DataFrame(
            self.imputer.fit_transform(X_train), columns=available
        )
        X_val_imp = pd.DataFrame(self.imputer.transform(X_val), columns=available)

        self.model = self._build_model()
        if self.model_type == "xgboost":
            self.model.fit(
                X_train_imp, y_train, eval_set=[(X_val_imp, y_val)], verbose=False
            )
        else:
            self.model.fit(X_train_imp, y_train)

        y_proba = self.model.predict_proba(X_val_imp)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        metrics = {
            "auc_roc": round(roc_auc_score(y_val, y_proba), 4),
            "f1": round(f1_score(y_val, y_pred), 4),
        }
        log.info(
            f"Claim Approval Model ({self.model_type}) — AUC: {metrics['auc_roc']} | F1: {metrics['f1']}"
        )

        if HAS_SHAP and self.model_type == "xgboost":
            self.explainer = shap.TreeExplainer(self.model)

        self._is_fitted = True
        self._feature_cols = available
        return metrics

    def predict(self, claim_features: dict, fraud_score: float = 0.0) -> dict:
        """
        Score a single claim dict.

        Returns:
          approval_probability, risk_band, recommendation, top_risk_factors
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call .fit() or .load() first.")

        X = pd.DataFrame([claim_features])[self._feature_cols]
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self._feature_cols)

        approval_proba = self.model.predict_proba(X_imp)[0, 1]
        risk_score = 1.0 - approval_proba
        risk_band = get_risk_band(risk_score)
        recommendation = get_recommendation(risk_band, fraud_score)

        # SHAP top factors
        top_factors = []
        if self.explainer is not None:
            sv = self.explainer.shap_values(X_imp)[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:3]
            for i in top_idx:
                col = self._feature_cols[i]
                val = float(X_imp.iloc[0, i])
                shap_v = float(sv[i])
                direction = "increases" if shap_v > 0 else "decreases"
                top_factors.append(
                    {
                        "feature": col,
                        "value": round(val, 4),
                        "direction": direction,
                        "shap": round(shap_v, 4),
                    }
                )

        return {
            "approval_probability": round(float(approval_proba), 4),
            "risk_score": round(float(risk_score), 4),
            "risk_band": risk_band,
            "recommendation": recommendation,
            "top_risk_factors": top_factors,
        }

    def save(self, path: str = None) -> str:
        path = path or f"{MODEL_DIR}/claim_approval_{self.model_type}.joblib"
        joblib.dump(
            {
                "model": self.model,
                "imputer": self.imputer,
                "feature_cols": self._feature_cols,
                "explainer": self.explainer,
            },
            path,
        )
        log.info(f"Claim approval model saved → {path}")
        return path

    def load(self, path: str) -> None:
        art = joblib.load(path)
        self.model = art["model"]
        self.imputer = art["imputer"]
        self._feature_cols = art["feature_cols"]
        self.explainer = art.get("explainer")
        self._is_fitted = True
        log.info(f"Claim approval model loaded ← {path}")
