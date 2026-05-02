import httpx
import json

BASE_URL = "http://localhost:8000"

# ── Test 1: Health Check ──────────────────────────────────────
print("=" * 50)
print("TEST 1 - HEALTH CHECK")
print("=" * 50)
r = httpx.get(f"{BASE_URL}/health")
print(json.dumps(r.json(), indent=2))

# ── Test 2: Low Risk Claim ────────────────────────────────────
print()
print("=" * 50)
print("TEST 2 - LOW RISK CLAIM (should be AUTO APPROVE)")
print("=" * 50)
low_risk = {
    "claim_id": "CLM_LOW_001",
    "member_id": "MBR_001",
    "provider_npi": "NPI_001",
    "billed_amount": 150.00,
    "paid_amount": 120.00,
    "allowed_amount": 130.00,
    "n_procedures": 1,
    "has_high_risk_cpt": 0,
    "duplicate_flag": 0,
    "out_of_network": 0,
    "prior_auth": 1,
    "oig_excluded": 0,
    "peer_billing_percentile": 45.0,
    "specialty_risk_score": 1.0,
    "age": 35,
    "chronic_conditions": 0,
    "prior_claims_12m": 2,
    "is_weekend": 0
}
r = httpx.post(f"{BASE_URL}/v1/claims/score", json=low_risk)
print(json.dumps(r.json(), indent=2))

# ── Test 3: High Risk Fraud Claim ─────────────────────────────
print()
print("=" * 50)
print("TEST 3 - HIGH RISK FRAUD CLAIM (should be FLAGGED)")
print("=" * 50)
high_risk = {
    "claim_id": "CLM_FRAUD_001",
    "member_id": "MBR_999",
    "provider_npi": "NPI_999",
    "billed_amount": 45000.00,
    "paid_amount": 0.00,
    "allowed_amount": 0.00,
    "n_procedures": 15,
    "has_high_risk_cpt": 1,
    "duplicate_flag": 1,
    "out_of_network": 1,
    "prior_auth": 0,
    "oig_excluded": 1,
    "peer_billing_percentile": 98.0,
    "specialty_risk_score": 5.0,
    "age": 45,
    "chronic_conditions": 0,
    "prior_claims_12m": 30,
    "is_weekend": 1
}
r = httpx.post(f"{BASE_URL}/v1/claims/score", json=high_risk)
print(json.dumps(r.json(), indent=2))

# ── Test 4: Get Metrics ───────────────────────────────────────
print()
print("=" * 50)
print("TEST 4 - MODEL METRICS")
print("=" * 50)
r = httpx.get(f"{BASE_URL}/v1/metrics")
print(json.dumps(r.json(), indent=2))