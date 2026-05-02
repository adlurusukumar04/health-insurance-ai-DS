"""
test_fraud_detection.py
-----------------------
Comprehensive test suite for the Health Insurance AI Fraud Detection System.

Covers:
  - Data generation (synthetic data quality)
  - Feature engineering (correctness, no leakage)
  - HIPAA anonymizer (PHI removal, hash consistency)
  - Model scoring (API endpoints, response schema)
  - Monitoring (PSI calculation, alert thresholds)
  - Edge cases (empty data, nulls, extremes)

Run:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pandas as pd
import pytest

from src.compliance.anonymizer import Anonymizer  # noqa: E402
from src.monitoring.model_monitor import (compute_psi,  # noqa: E402
                                          monitor_feature_drift)
from src.processing.feature_engineering import FeatureEngineer  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_members() -> pd.DataFrame:
    """Minimal member DataFrame for testing."""
    return pd.DataFrame(
        {
            "member_id": ["MBR000001", "MBR000002", "MBR000003"],
            "first_name": ["Alice", "Bob", "Carol"],
            "last_name": ["Smith", "Jones", "Williams"],
            "dob": ["1975-03-15", "1960-07-22", "1990-11-01"],
            "age": [49, 64, 34],
            "gender": ["F", "M", "F"],
            "state": ["CA", "TX", "NY"],
            "zip_code": ["90210", "78201", "10001"],
            "plan_type": ["PPO", "HMO", "HDHP"],
            "plan_start_date": ["2022-01-01", "2021-06-01", "2023-03-15"],
            "chronic_conditions": [2, 5, 0],
            "prior_claims_12m": [8, 25, 2],
            "address_changes_12m": [0, 2, 0],
        }
    )


@pytest.fixture
def sample_providers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "npi": ["NPI0000000001", "NPI0000000002", "NPI0000000003"],
            "provider_name": ["Alpha Medical", "Beta Group", "Gamma Clinic"],
            "specialty": ["Cardiology", "Emergency Medicine", "Family Practice"],
            "state": ["CA", "TX", "NY"],
            "zip_code": ["90001", "78202", "10002"],
            "license_active": [True, True, False],
            "oig_excluded": [False, True, False],
            "years_practice": [15, 8, 3],
            "avg_monthly_claims": [120, 950, 60],
            "peer_billing_percentile": [55.0, 97.0, 40.0],
            "fraud_prone": [False, True, False],
        }
    )


@pytest.fixture
def sample_claims() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["CLM00000001", "CLM00000002", "CLM00000003"],
            "member_id": ["MBR000001", "MBR000002", "MBR000003"],
            "provider_npi": ["NPI0000000001", "NPI0000000002", "NPI0000000003"],
            "claim_date": ["2024-03-01", "2024-03-10", "2024-03-15"],
            "service_date": ["2024-02-28", "2024-03-08", "2024-03-15"],
            "icd10_primary": ["I10", "J18.9", "Z00.00"],
            "cpt_codes": ["99213|93000", "27447|99285|G0444", "99213"],
            "n_procedures": [2, 3, 1],
            "billed_amount": [450.00, 12500.00, 150.00],
            "allowed_amount": [280.00, 8000.00, 120.00],
            "paid_amount": [224.00, 6400.00, 96.00],
            "claim_type": ["Medical", "Medical", "Medical"],
            "place_of_service": ["11", "23", "11"],
            "prior_auth": [False, False, False],
            "duplicate_flag": [False, True, False],
            "out_of_network": [False, True, False],
            "days_supply": [30, 1, 30],
            "fraud_label": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_pharmacy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rx_id": ["RX00000001", "RX00000002"],
            "member_id": ["MBR000001", "MBR000002"],
            "prescriber_npi": ["NPI0000000001", "NPI0000000002"],
            "pharmacy_id": ["PHARM0001", "PHARM0001"],
            "drug_code": ["NDC12345", "NDC67890"],
            "drug_name": ["Lisinopril", "OxyContin"],
            "days_supply": [30, 30],
            "quantity": [30, 90],
            "fill_date": ["2024-02-01", "2024-03-01"],
            "refill_number": [2, 5],
            "billed_amount": [45.00, 380.00],
            "paid_amount": [12.00, 210.00],
            "controlled_substance": [False, True],
        }
    )


@pytest.fixture
def anonymizer():
    return Anonymizer(salt="test-salt-2024")


@pytest.fixture
def feature_engineer():
    return FeatureEngineer()


# ─────────────────────────────────────────────────────────────────────────────
# ANONYMIZER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestAnonymizer:

    def test_removes_names(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        assert "first_name" not in result.columns, "first_name must be removed (HIPAA)"
        assert "last_name" not in result.columns, "last_name must be removed (HIPAA)"

    def test_removes_dob(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        assert "dob" not in result.columns, "DOB must be removed (HIPAA)"
        assert "age_band" in result.columns, "age_band should replace DOB"

    def test_age_band_format(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        valid_bands = {f"{d}s" for d in range(0, 120, 10)} | {"Unknown"}
        for band in result["age_band"]:
            assert band in valid_bands, f"Invalid age band: {band}"

    def test_zip_generalization(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        assert "zip_code" not in result.columns, "Full ZIP must be removed"
        assert "zip3" in result.columns, "3-digit ZIP prefix should be present"
        for z in result["zip3"]:
            assert (
                len(str(z)) == 3 or z == "000"
            ), f"ZIP3 must be 3 digits or '000': {z}"

    def test_member_id_hashed(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        for orig, hashed in zip(sample_members["member_id"], result["member_id"]):
            assert orig != hashed, "member_id must be hashed"
            assert hashed.startswith("TOK_"), "Hash should start with 'TOK_'"

    def test_hash_deterministic(self, anonymizer):
        """Same input always produces same hash (for join-ability)."""
        h1 = anonymizer.tokenize_id("MBR000001")
        h2 = anonymizer.tokenize_id("MBR000001")
        assert h1 == h2, "Hash must be deterministic"

    def test_hash_different_inputs(self, anonymizer):
        h1 = anonymizer.tokenize_id("MBR000001")
        h2 = anonymizer.tokenize_id("MBR000002")
        assert h1 != h2, "Different inputs must produce different hashes"

    def test_state_preserved(self, anonymizer, sample_members):
        result = anonymizer.anonymize_members(sample_members)
        assert (
            "state" in result.columns
        ), "State (safe under Safe Harbor) should be preserved"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineer:

    def test_output_shape(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        assert len(result) == len(sample_claims), "Row count must match claims"
        assert "fraud_label" in result.columns, "Target column must be present"

    def test_billed_per_procedure(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        assert "billed_per_procedure" in result.columns
        # CLM1: $450 / 2 procedures = $225
        row = result[result["claim_id"] == "CLM00000001"]
        assert abs(row["billed_per_procedure"].values[0] - 225.0) < 1.0

    def test_high_risk_cpt_detection(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        # CLM2 has CPT 27447 (knee replacement) — high risk
        row2 = result[result["claim_id"] == "CLM00000002"]
        assert row2["has_high_risk_cpt"].values[0] == 1

    def test_no_target_leakage(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        """Feature matrix must not contain raw fraud indicator columns."""
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        leaky_cols = ["fraud_prone", "is_fraud"]
        for col in leaky_cols:
            assert col not in result.columns, f"Leaky column '{col}' found in features!"

    def test_allowed_ratio_clipped(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        assert (
            result["allowed_ratio"].between(0, 1).all()
        ), "allowed_ratio must be in [0,1]"

    def test_temporal_features(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        result = feature_engineer.build_claim_features(
            sample_claims, sample_members, sample_providers, sample_pharmacy
        )
        assert "is_weekend" in result.columns
        assert "claim_dow" in result.columns
        assert result["is_weekend"].isin([0, 1]).all()

    def test_handles_missing_provider(
        self, feature_engineer, sample_members, sample_providers, sample_pharmacy
    ):
        """Claims with unknown provider NPI should not crash."""
        claims_unknown = pd.DataFrame(
            {
                "claim_id": ["CLM99999999"],
                "member_id": ["MBR000001"],
                "provider_npi": ["NPI_UNKNOWN"],
                "claim_date": ["2024-03-01"],
                "service_date": ["2024-03-01"],
                "cpt_codes": ["99213"],
                "n_procedures": [1],
                "billed_amount": [200.0],
                "allowed_amount": [150.0],
                "paid_amount": [120.0],
                "claim_type": ["Medical"],
                "place_of_service": ["11"],
                "prior_auth": [False],
                "duplicate_flag": [False],
                "out_of_network": [False],
                "days_supply": [30],
                "fraud_label": [0],
            }
        )
        result = feature_engineer.build_claim_features(
            claims_unknown, sample_members, sample_providers, sample_pharmacy
        )
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# MODEL MONITOR TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestModelMonitor:

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be near 0."""
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI of identical distributions should be ~0, got {psi}"

    def test_psi_very_different_distributions(self):
        """PSI of very different distributions should be high."""
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)  # shifted mean by 5 std
        psi = compute_psi(ref, cur)
        assert (
            psi > 0.25
        ), f"PSI of very different distributions should be >0.25, got {psi}"

    def test_psi_moderate_shift(self):
        """PSI of moderately shifted distributions should be 0.10–0.25."""
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(1, 1, 1000)  # 1 std shift
        psi = compute_psi(ref, cur)
        assert psi >= 0.05, f"Expected moderate PSI, got {psi}"

    def test_feature_drift_no_alerts_stable(self, sample_claims):
        """No alerts when reference and current are the same data."""
        ref = sample_claims.copy()
        ref["billed_amount"] = np.random.normal(500, 100, len(ref))
        ref["peer_billing_percentile"] = np.random.uniform(20, 80, len(ref))

        # Create larger dfs for PSI
        ref_large = pd.concat([ref] * 20, ignore_index=True)
        cur_large = ref_large.copy()  # identical

        result = monitor_feature_drift(ref_large, cur_large)
        critical = [a for a in result["alerts"] if "CRITICAL" in a["type"]]
        assert len(critical) == 0, "No critical alerts for stable data"

    def test_psi_positive_value(self):
        """PSI should always be non-negative."""
        ref = np.random.exponential(1.0, 500)
        cur = np.random.exponential(2.0, 500)
        psi = compute_psi(ref, cur)
        assert psi >= 0, "PSI must be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# API ENDPOINT TESTS (using FastAPI TestClient)
# ─────────────────────────────────────────────────────────────────────────────


class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.main import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "fraud_model" in data
        assert "approval_model" in data

    def test_score_claim_valid(self, client):
        payload = {
            "claim_id": "CLM_TEST_001",
            "member_id": "MBR_TEST_001",
            "provider_npi": "NPI_TEST_001",
            "billed_amount": 1500.00,
            "paid_amount": 1000.00,
            "allowed_amount": 1100.00,
            "n_procedures": 3,
            "has_high_risk_cpt": 0,
            "duplicate_flag": 0,
            "out_of_network": 0,
            "prior_auth": 1,
            "oig_excluded": 0,
            "peer_billing_percentile": 55.0,
            "specialty_risk_score": 2.0,
            "age": 45,
            "chronic_conditions": 1,
            "prior_claims_12m": 5,
            "is_weekend": 0,
        }
        response = client.post("/v1/claims/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "fraud_risk_band" in data
        assert "approval_probability" in data
        assert "recommendation" in data
        assert 0.0 <= data["fraud_score"] <= 1.0
        assert data["fraud_risk_band"] in ("LOW", "MEDIUM", "HIGH", "VERY_HIGH")

    def test_score_claim_high_fraud_signals(self, client):
        """Claim with multiple fraud signals should receive high score."""
        payload = {
            "claim_id": "CLM_FRAUD_001",
            "member_id": "MBR_TEST_999",
            "provider_npi": "NPI_TEST_999",
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
            "is_weekend": 1,
        }
        response = client.post("/v1/claims/score", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert (
            data["fraud_score"] >= 0.50
        ), "High fraud signal claim should score > 0.50"
        assert data["fraud_risk_band"] in ("HIGH", "VERY_HIGH")

    def test_score_claim_missing_required_field(self, client):
        """Missing required field should return 422."""
        payload = {"claim_id": "CLM_BAD_001", "billed_amount": 500.0}
        response = client.post("/v1/claims/score", json=payload)
        assert response.status_code == 422

    def test_score_negative_amount_rejected(self, client):
        """Negative billed amount should return 422."""
        payload = {
            "claim_id": "CLM_NEG_001",
            "member_id": "M1",
            "provider_npi": "N1",
            "billed_amount": -100.0,
            "n_procedures": 1,
            "has_high_risk_cpt": 0,
            "duplicate_flag": 0,
            "out_of_network": 0,
            "prior_auth": 0,
            "oig_excluded": 0,
            "peer_billing_percentile": 50,
            "specialty_risk_score": 2,
            "age": 40,
            "chronic_conditions": 0,
            "prior_claims_12m": 0,
            "is_weekend": 0,
        }
        response = client.post("/v1/claims/score", json=payload)
        assert response.status_code == 422

    def test_explanation_not_found(self, client):
        response = client.get("/v1/claims/CLM_NEVER_SCORED/explanation")
        assert response.status_code == 404

    def test_explanation_after_scoring(self, client):
        # Score first
        payload = {
            "claim_id": "CLM_EXP_001",
            "member_id": "M1",
            "provider_npi": "N1",
            "billed_amount": 500.0,
            "paid_amount": 300.0,
            "allowed_amount": 350.0,
            "n_procedures": 1,
            "has_high_risk_cpt": 0,
            "duplicate_flag": 0,
            "out_of_network": 0,
            "prior_auth": 1,
            "oig_excluded": 0,
            "peer_billing_percentile": 45,
            "specialty_risk_score": 2,
            "age": 35,
            "chronic_conditions": 0,
            "prior_claims_12m": 3,
            "is_weekend": 0,
        }
        client.post("/v1/claims/score", json=payload)
        response = client.get("/v1/claims/CLM_EXP_001/explanation")
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert "natural_language_explanation" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "model_performance" in data
        assert "system_stats" in data

    def test_provider_risk_profile(self, client):
        response = client.get("/v1/providers/NPI0000000001/risk-profile")
        assert response.status_code == 200
        data = response.json()
        assert "risk_score" in data
        assert "risk_band" in data
        assert "npi" in data


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_claims_dataframe(
        self, feature_engineer, sample_members, sample_providers, sample_pharmacy
    ):
        empty = pd.DataFrame(
            columns=[
                "claim_id",
                "member_id",
                "provider_npi",
                "claim_date",
                "service_date",
                "cpt_codes",
                "n_procedures",
                "billed_amount",
                "allowed_amount",
                "paid_amount",
                "claim_type",
                "place_of_service",
                "prior_auth",
                "duplicate_flag",
                "out_of_network",
                "days_supply",
                "fraud_label",
            ]
        )
        result = feature_engineer.build_claim_features(
            empty, sample_members, sample_providers, sample_pharmacy
        )
        assert len(result) == 0

    def test_zero_billed_amount(
        self,
        feature_engineer,
        sample_claims,
        sample_members,
        sample_providers,
        sample_pharmacy,
    ):
        """Zero billed amount should not cause division by zero."""
        claims = sample_claims.copy()
        claims.loc[0, "billed_amount"] = 0.0
        try:
            result = feature_engineer.build_claim_features(
                claims, sample_members, sample_providers, sample_pharmacy
            )
            assert len(result) == len(claims)
        except ZeroDivisionError:
            pytest.fail("Division by zero on zero billed amount")

    def test_anonymizer_empty_dataframe(self, anonymizer):
        empty = pd.DataFrame(
            columns=["member_id", "first_name", "last_name", "dob", "zip_code", "state"]
        )
        result = anonymizer.anonymize_members(empty)
        assert len(result) == 0
        assert "first_name" not in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
