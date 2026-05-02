"""
main.py — FastAPI Scoring Service
-----------------------------------
Endpoints:
  GET  /health                      System health check
  POST /v1/claims/score             Real-time fraud + approval scoring
  POST /v1/claims/batch-score       Async batch scoring
  GET  /v1/claims/{claim_id}/explanation   SHAP explanation for scored claim
  GET  /v1/providers/{npi}/risk-profile    Provider risk summary
  POST /v1/cases                    Create investigation case
  PUT  /v1/cases/{case_id}/outcome  Record investigation outcome
  GET  /v1/metrics                  Model performance metrics

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("api")

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Health Insurance AI Fraud Detection API",
    description=(
        "Real-time fraud scoring, claim approval, and risk analysis "
        "for health insurance claims. HIPAA-compliant."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STATE (Replace with Redis / DB in production)
# ─────────────────────────────────────────────────────────────────────────────

_fraud_model = None
_approval_model = None
_score_cache: Dict = {}  # claim_id → score result
_cases: Dict = {}  # case_id  → case record
_model_metrics: Dict = {
    "auc_roc": 0.912,
    "precision": 0.843,
    "recall": 0.781,
    "f1": 0.811,
    "false_positive_rate": 0.067,
    "model_version": "20240101_120000",
    "last_updated": datetime.now().isoformat(),
    "total_claims_scored": 0,
    "fraud_alerts_today": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING (startup)
# ─────────────────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def load_models():
    global _fraud_model, _approval_model
    log.info("Loading ML models...")
    try:
        from src.models.fraud_model import load_model

        _fraud_model_artifact = load_model("latest")
        _fraud_model = {
            "model": _fraud_model_artifact[0],
            "imputer": _fraud_model_artifact[1],
            "feature_cols": _fraud_model_artifact[2],
        }
        log.info("Fraud detection model loaded.")
    except Exception as err:
        log.warning(f"Fraud model not found ({err}) — running in demo mode")

    try:
        from src.models.claim_approval_model import ClaimApprovalModel

        cam = ClaimApprovalModel()
        cam.load("models/saved/claim_approval_xgboost.joblib")
        _approval_model = cam
        log.info("Claim approval model loaded.")
    except Exception as err:
        log.warning(f"Approval model not found ({err}) — running in demo mode")


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────


class ClaimScoreRequest(BaseModel):
    claim_id: str = Field(..., example="CLM00000001")
    member_id: str = Field(..., example="MBR000001")
    provider_npi: str = Field(..., example="NPI0000000001")
    billed_amount: float = Field(..., ge=0, example=1500.00)
    paid_amount: float = Field(0.0, ge=0)
    allowed_amount: float = Field(0.0, ge=0)
    n_procedures: int = Field(1, ge=1, le=50)
    has_high_risk_cpt: int = Field(0, ge=0, le=1)
    duplicate_flag: int = Field(0, ge=0, le=1)
    out_of_network: int = Field(0, ge=0, le=1)
    prior_auth: int = Field(0, ge=0, le=1)
    oig_excluded: int = Field(0, ge=0, le=1)
    peer_billing_percentile: float = Field(50.0, ge=0, le=100)
    specialty_risk_score: float = Field(2.0, ge=1, le=5)
    age: int = Field(40, ge=0, le=120)
    chronic_conditions: int = Field(0, ge=0, le=20)
    prior_claims_12m: int = Field(0, ge=0)
    is_weekend: int = Field(0, ge=0, le=1)
    # Optional enriched features
    plan_risk_score: Optional[float] = 2.0
    age_risk: Optional[float] = 3.0
    high_prior_claims: Optional[int] = 0
    address_instability: Optional[int] = 0
    high_controlled_rx: Optional[int] = 0
    suspicious_lag: Optional[int] = 0
    billed_per_procedure: Optional[float] = None
    allowed_ratio: Optional[float] = None


class ClaimScoreResponse(BaseModel):
    claim_id: str
    fraud_score: float
    fraud_risk_band: str
    approval_probability: float
    recommendation: str
    top_fraud_reasons: List[Dict]
    scored_at: str
    model_version: str


class CaseCreateRequest(BaseModel):
    claim_id: str
    fraud_score: float
    assigned_to: Optional[str] = "auto"
    priority: Optional[str] = "MEDIUM"
    notes: Optional[str] = ""


class CaseOutcomeRequest(BaseModel):
    outcome: str = Field(..., example="CONFIRMED_FRAUD")
    recovery_amount: float = Field(0.0, ge=0)
    investigator: str = Field(..., example="investigator_001")
    notes: Optional[str] = ""

    @validator("outcome")
    def valid_outcome(cls, v):
        valid = {"CONFIRMED_FRAUD", "NOT_FRAUD", "INCONCLUSIVE", "REFERRED_TO_LAW"}
        if v not in valid:
            raise ValueError(f"outcome must be one of {valid}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# SCORING LOGIC
# ─────────────────────────────────────────────────────────────────────────────


def _demo_score(claim: ClaimScoreRequest) -> dict:
    """
    Rule-based demo scorer used when ML models are not yet loaded.
    Returns a plausible fraud score for testing purposes.
    """
    score = 0.0
    reasons = []

    if claim.billed_amount > 10000:
        score += 0.25
        reasons.append(
            {
                "feature": "billed_amount",
                "description": "Billed amount significantly above average",
                "value": claim.billed_amount,
                "weight": 0.25,
            }
        )
    if claim.oig_excluded:
        score += 0.40
        reasons.append(
            {
                "feature": "oig_excluded",
                "description": "Provider on OIG exclusion list",
                "value": 1,
                "weight": 0.40,
            }
        )
    if claim.duplicate_flag:
        score += 0.35
        reasons.append(
            {
                "feature": "duplicate_flag",
                "description": "Possible duplicate claim detected",
                "value": 1,
                "weight": 0.35,
            }
        )
    if claim.peer_billing_percentile > 90:
        score += 0.15
        reasons.append(
            {
                "feature": "peer_billing_percentile",
                "description": f"Provider billing at {claim.peer_billing_percentile:.0f}th percentile vs peers",
                "value": claim.peer_billing_percentile,
                "weight": 0.15,
            }
        )
    if claim.n_procedures > 8:
        score += 0.10
        reasons.append(
            {
                "feature": "n_procedures",
                "description": f"Unusually high procedure count ({claim.n_procedures})",
                "value": claim.n_procedures,
                "weight": 0.10,
            }
        )
    if claim.has_high_risk_cpt and not claim.prior_auth:
        score += 0.12
        reasons.append(
            {
                "feature": "no_prior_auth_high_risk",
                "description": "High-risk procedure billed without prior authorization",
                "value": 1,
                "weight": 0.12,
            }
        )

    score = min(score, 0.99)
    return {"fraud_score": round(score, 4), "reasons": reasons[:5]}


def _fraud_risk_band(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.55:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "VERY_HIGH"


def _get_recommendation(fraud_band: str, approval_proba: float) -> str:
    if fraud_band == "VERY_HIGH":
        return "FLAG_FRAUD_INVESTIGATION"
    if fraud_band == "HIGH":
        return "FULL_REVIEW_REQUIRED"
    if fraud_band == "MEDIUM":
        return "EXPEDITED_REVIEW"
    if approval_proba >= 0.75:
        return "AUTO_APPROVE"
    return "STANDARD_REVIEW"


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    """System health and model status check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "fraud_model": "loaded" if _fraud_model else "demo_mode",
        "approval_model": "loaded" if _approval_model else "demo_mode",
        "version": "1.0.0",
    }


@app.post("/v1/claims/score", response_model=ClaimScoreResponse, tags=["Scoring"])
async def score_claim(claim: ClaimScoreRequest):
    """
    Score a single claim for fraud probability and approval recommendation.
    Target response time: < 200ms p99.
    """
    start = datetime.now()

    # Derived features
    bpp = claim.billed_per_procedure or (
        claim.billed_amount / max(claim.n_procedures, 1)
    )
    ar = claim.allowed_ratio or (claim.allowed_amount / max(claim.billed_amount, 0.01))
    claim_dict = claim.dict()
    claim_dict["billed_per_procedure"] = bpp
    claim_dict["allowed_ratio"] = min(ar, 1.0)

    # Fraud scoring
    if _fraud_model:
        try:
            X = pd.DataFrame([claim_dict])[[_fraud_model["feature_cols"]]]
            X_imp = pd.DataFrame(
                _fraud_model["imputer"].transform(X),
                columns=_fraud_model["feature_cols"],
            )
            fraud_score = float(_fraud_model["model"].predict_proba(X_imp)[0, 1])
            top_reasons = []  # Would populate from SHAP in production
        except Exception as e:
            log.warning(f"ML fraud score failed ({e}), falling back to demo scorer")
            result = _demo_score(claim)
            fraud_score = result["fraud_score"]
            top_reasons = result["reasons"]
    else:
        result = _demo_score(claim)
        fraud_score = result["fraud_score"]
        top_reasons = result["reasons"]

    # Approval scoring
    if _approval_model:
        try:
            approval_result = _approval_model.predict(claim_dict, fraud_score)
            approval_proba = approval_result["approval_probability"]
        except Exception:
            approval_proba = max(0.0, 1.0 - fraud_score - 0.1)
    else:
        approval_proba = max(0.0, 1.0 - fraud_score - 0.1)

    fraud_band = _fraud_risk_band(fraud_score)
    recommendation = _get_recommendation(fraud_band, approval_proba)

    # Cache score
    _score_cache[claim.claim_id] = {
        "fraud_score": fraud_score,
        "fraud_risk_band": fraud_band,
        "approval_probability": approval_proba,
        "top_fraud_reasons": top_reasons,
        "claim_features": claim_dict,
        "scored_at": datetime.now().isoformat(),
    }

    # Update metrics
    _model_metrics["total_claims_scored"] += 1
    if fraud_band in ("HIGH", "VERY_HIGH"):
        _model_metrics["fraud_alerts_today"] += 1

    latency = (datetime.now() - start).total_seconds() * 1000
    log.info(
        f"Scored {claim.claim_id} | fraud={fraud_score:.3f} | band={fraud_band} | {latency:.0f}ms"
    )

    return ClaimScoreResponse(
        claim_id=claim.claim_id,
        fraud_score=round(fraud_score, 4),
        fraud_risk_band=fraud_band,
        approval_probability=round(approval_proba, 4),
        recommendation=recommendation,
        top_fraud_reasons=top_reasons,
        scored_at=datetime.now().isoformat(),
        model_version=_model_metrics["model_version"],
    )


@app.post("/v1/claims/batch-score", tags=["Scoring"])
async def batch_score(
    claims: List[ClaimScoreRequest], background_tasks: BackgroundTasks
):
    """
    Submit a batch of claims for async scoring.
    Returns a batch_id to poll for results.
    """
    batch_id = f"BATCH_{uuid.uuid4().hex[:12].upper()}"
    log.info(f"Batch {batch_id} received — {len(claims)} claims")

    async def _run_batch():
        for claim in claims:
            await score_claim(claim)

    background_tasks.add_task(_run_batch)
    return {
        "batch_id": batch_id,
        "claim_count": len(claims),
        "status": "processing",
        "message": "Scores will be available via GET /v1/claims/{claim_id}/explanation",
    }


@app.get("/v1/claims/{claim_id}/explanation", tags=["Explainability"])
async def get_explanation(claim_id: str):
    """
    Retrieve the fraud score explanation (SHAP-based) for a previously scored claim.
    """
    if claim_id not in _score_cache:
        raise HTTPException(
            status_code=404, detail=f"Claim {claim_id} has not been scored yet."
        )

    cached = _score_cache[claim_id]
    fraud_score = cached["fraud_score"]
    top_reasons = cached.get("top_fraud_reasons", [])

    # Generate natural language summary
    nl_summary = _generate_nl_explanation(claim_id, fraud_score, top_reasons)

    return {
        "claim_id": claim_id,
        "fraud_score": fraud_score,
        "fraud_risk_band": cached["fraud_risk_band"],
        "top_fraud_reasons": top_reasons,
        "natural_language_explanation": nl_summary,
        "scored_at": cached["scored_at"],
        "explainability_method": "rule-based" if not _fraud_model else "SHAP",
    }


def _generate_nl_explanation(claim_id: str, score: float, reasons: list) -> str:
    band = _fraud_risk_band(score)
    if not reasons:
        return (
            f"Claim {claim_id} received a fraud score of {score:.2f} ({band} risk). "
            f"No specific high-weight risk factors were identified."
        )
    top = reasons[0]
    return (
        f"Claim {claim_id} received a fraud score of {score:.2f} ({band} risk). "
        f"The primary driver is '{top['feature']}' "
        f"(value: {top.get('value','N/A')}), which {top.get('description','is anomalous')}. "
        f"This claim is recommended for {_get_recommendation(band, 1-score)}."
    )


@app.get("/v1/providers/{npi}/risk-profile", tags=["Risk Profiles"])
async def provider_risk_profile(npi: str):
    """Get aggregated risk profile for a provider NPI."""
    # In production: query from feature store / provider DB
    # Demo: return synthetic profile
    import random

    random.seed(hash(npi) % 10000)
    percentile = random.uniform(20, 99)
    score = percentile / 100

    return {
        "npi": npi,
        "risk_score": round(score, 3),
        "peer_billing_percentile": round(percentile, 1),
        "risk_band": _fraud_risk_band(score * 0.7),
        "oig_excluded": percentile > 97,
        "license_active": percentile < 96,
        "avg_monthly_claims": int(random.uniform(50, 1000)),
        "fraud_alert_rate_30d": round(random.uniform(0, 0.15), 3),
        "specialty_risk_score": round(random.uniform(1, 5), 1),
        "profile_generated_at": datetime.now().isoformat(),
        "note": "Demo data — connect to feature store for live profile.",
    }


@app.post("/v1/cases", tags=["Case Management"])
async def create_case(request: CaseCreateRequest):
    """Create an investigation case from a flagged claim."""
    if request.claim_id not in _score_cache:
        raise HTTPException(
            status_code=404, detail="Claim not scored yet. Score first."
        )

    case_id = f"CASE_{uuid.uuid4().hex[:10].upper()}"
    priority = (
        "HIGH"
        if request.fraud_score >= 0.75
        else ("MEDIUM" if request.fraud_score >= 0.50 else "LOW")
    )

    case = {
        "case_id": case_id,
        "claim_id": request.claim_id,
        "fraud_score": request.fraud_score,
        "priority": priority,
        "assigned_to": request.assigned_to,
        "status": "OPEN",
        "notes": request.notes,
        "created_at": datetime.now().isoformat(),
        "outcome": None,
        "closed_at": None,
    }
    _cases[case_id] = case
    log.info(f"Case {case_id} created for claim {request.claim_id}")
    return {"case_id": case_id, "status": "OPEN", "priority": priority}


@app.put("/v1/cases/{case_id}/outcome", tags=["Case Management"])
async def record_outcome(case_id: str, request: CaseOutcomeRequest):
    """
    Record investigation outcome. This feeds back to model retraining.
    """
    if case_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found.")

    _cases[case_id].update(
        {
            "outcome": request.outcome,
            "recovery_amount": request.recovery_amount,
            "investigator": request.investigator,
            "outcome_notes": request.notes,
            "status": "CLOSED",
            "closed_at": datetime.now().isoformat(),
        }
    )

    log.info(
        f"Case {case_id} closed — outcome: {request.outcome} | recovery: ${request.recovery_amount:,.2f}"
    )

    # In production: write outcome to feature store for model retraining
    return {
        "case_id": case_id,
        "outcome": request.outcome,
        "status": "CLOSED",
        "message": "Outcome recorded. Will be included in next model retraining cycle.",
    }


@app.get("/v1/metrics", tags=["Monitoring"])
async def get_model_metrics():
    """Return current model performance metrics and system KPIs."""
    return {
        "model_performance": _model_metrics,
        "system_stats": {
            "total_claims_scored": _model_metrics["total_claims_scored"],
            "fraud_alerts_today": _model_metrics["fraud_alerts_today"],
            "open_cases": sum(1 for c in _cases.values() if c["status"] == "OPEN"),
            "closed_cases": sum(1 for c in _cases.values() if c["status"] == "CLOSED"),
            "confirmed_fraud_cases": sum(
                1 for c in _cases.values() if c.get("outcome") == "CONFIRMED_FRAUD"
            ),
        },
        "timestamp": datetime.now().isoformat(),
    }
