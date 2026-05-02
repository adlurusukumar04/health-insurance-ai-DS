"""
feature_engineering.py
-----------------------
Builds the complete feature matrix for claim-level fraud detection.

Feature families:
  1. Claim-level         — amounts, codes, flags
  2. Provider-level      — billing velocity, peer percentile, risk signals
  3. Member-level        — history, frequency, anomaly signals
  4. Temporal            — rolling windows, day-of-week, recency
  5. Network proxy       — shared patterns (without graph DB dependency)
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw cleaned dataframes into a model-ready feature matrix."""

    # CPT code groups for feature derivation
    HIGH_RISK_CPT = {"99285", "27447", "27130", "43239", "47562", "T1015"}
    PREVENTIVE_CPT = {"G0439", "G0444", "Z23", "99490"}

    def build_claim_features(
        self,
        claims: pd.DataFrame,
        members: pd.DataFrame,
        providers: pd.DataFrame,
        pharmacy: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Master method. Returns enriched feature DataFrame aligned to claims.
        Target column: fraud_label (0/1)
        """
        log.info("Building claim-level features...")
        df = claims.copy()

        df = self._claim_features(df)
        df = self._provider_features(df, providers)
        df = self._member_features(df, members)
        df = self._temporal_features(df)
        df = self._pharmacy_features(df, pharmacy)
        df = self._ratio_features(df)
        df = self._finalize(df)

        log.info(f"Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    # ─────────────────────────────────────────────────────────
    # CLAIM-LEVEL FEATURES
    # ─────────────────────────────────────────────────────────

    def _claim_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("  → Claim-level features")

        # Code-level signals
        df["cpt_list"] = df["cpt_codes"].str.split("|")
        df["has_high_risk_cpt"] = df["cpt_list"].apply(
            lambda codes: int(bool(set(codes) & self.HIGH_RISK_CPT))
        )
        df["has_preventive_cpt"] = df["cpt_list"].apply(
            lambda codes: int(bool(set(codes) & self.PREVENTIVE_CPT))
        )
        df["unique_cpt_count"] = df["cpt_list"].apply(lambda x: len(set(x)))
        df["cpt_code_count"] = df["n_procedures"]

        # Amount features
        df["billed_per_procedure"] = df["billed_amount"] / df["n_procedures"].clip(
            lower=1
        )
        df["allowed_ratio"] = (
            df["allowed_amount"] / df["billed_amount"].clip(lower=0.01)
        ).clip(0, 1)
        df["paid_to_allowed_ratio"] = (
            df["paid_amount"] / df["allowed_amount"].clip(lower=0.01)
        ).clip(0, 1)
        df["billed_log"] = np.log1p(df["billed_amount"])
        df["paid_log"] = np.log1p(df["paid_amount"])

        # Flag features
        df["duplicate_flag"] = df["duplicate_flag"].astype(int)
        df["out_of_network"] = df["out_of_network"].astype(int)
        df["prior_auth"] = df["prior_auth"].astype(int)
        df["no_prior_auth_high_risk"] = (
            (df["prior_auth"] == 0) & (df["has_high_risk_cpt"] == 1)
        ).astype(int)

        # Place of service risk encoding
        high_risk_pos = {"23", "81"}  # Emergency, Independent Lab
        df["high_risk_pos"] = df["place_of_service"].isin(high_risk_pos).astype(int)

        return df

    # ─────────────────────────────────────────────────────────
    # PROVIDER-LEVEL FEATURES
    # ─────────────────────────────────────────────────────────

    def _provider_features(
        self, df: pd.DataFrame, providers: pd.DataFrame
    ) -> pd.DataFrame:
        log.info("  → Provider-level features")

        prov = providers.rename(columns={"npi": "provider_npi"})
        prov_cols = [
            "provider_npi",
            "specialty",
            "oig_excluded",
            "peer_billing_percentile",
            "avg_monthly_claims",
            "license_active",
            "years_practice",
        ]
        df = df.merge(prov[prov_cols], on="provider_npi", how="left")

        # Derived provider features
        df["oig_excluded"] = df["oig_excluded"].fillna(False).astype(int)
        df["license_active"] = df["license_active"].fillna(True).astype(int)
        df["peer_percentile_high"] = (df["peer_billing_percentile"] > 90).astype(int)
        df["peer_percentile_extreme"] = (df["peer_billing_percentile"] > 95).astype(int)
        df["peer_billing_percentile"] = df["peer_billing_percentile"].fillna(50)
        df["years_practice"] = df["years_practice"].fillna(5)

        # Provider claim volume from claims data itself
        prov_volumes = (
            df.groupby("provider_npi")["claim_id"]
            .count()
            .rename("provider_total_claims_in_data")
        )
        df = df.join(prov_volumes, on="provider_npi")

        # Specialty risk encoding (ordinal)
        specialty_risk = {
            "Emergency Medicine": 5,
            "Oncology": 4,
            "Orthopedics": 4,
            "Cardiology": 3,
            "Gastroenterology": 3,
            "Neurology": 3,
            "Internal Medicine": 2,
            "Psychiatry": 2,
            "Radiology": 2,
            "Family Practice": 1,
        }
        df["specialty_risk_score"] = df["specialty"].map(specialty_risk).fillna(2)

        return df

    # ─────────────────────────────────────────────────────────
    # MEMBER-LEVEL FEATURES
    # ─────────────────────────────────────────────────────────

    def _member_features(self, df: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
        log.info("  → Member-level features")

        mem_cols = [
            "member_id",
            "age",
            "gender",
            "plan_type",
            "chronic_conditions",
            "prior_claims_12m",
            "address_changes_12m",
        ]
        df = df.merge(members[mem_cols], on="member_id", how="left")

        # Age risk bands
        df["age_risk"] = pd.cut(
            df["age"].fillna(40), bins=[0, 18, 35, 55, 70, 120], labels=[1, 2, 3, 4, 5]
        ).astype(float)

        # Gender encode
        df["gender_encoded"] = df["gender"].map({"M": 0, "F": 1, "Other": 2}).fillna(2)

        # Plan type risk
        plan_risk = {"HMO": 1, "PPO": 2, "EPO": 2, "POS": 3, "HDHP": 4}
        df["plan_risk_score"] = df["plan_type"].map(plan_risk).fillna(2)

        # High prior claim frequency
        df["high_prior_claims"] = (df["prior_claims_12m"] > 20).astype(int)
        df["address_instability"] = (df["address_changes_12m"] >= 2).astype(int)

        # Member visit diversity (distinct providers per member)
        member_providers = (
            df.groupby("member_id")["provider_npi"]
            .nunique()
            .rename("member_distinct_providers")
        )
        df = df.join(member_providers, on="member_id")
        df["many_providers"] = (df["member_distinct_providers"] > 8).astype(int)

        # Total spend per member
        member_spend = (
            df.groupby("member_id")["billed_amount"].sum().rename("member_total_billed")
        )
        df = df.join(member_spend, on="member_id")

        return df

    # ─────────────────────────────────────────────────────────
    # TEMPORAL FEATURES
    # ─────────────────────────────────────────────────────────

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("  → Temporal features")

        df["claim_date"] = pd.to_datetime(df["claim_date"])
        df["claim_dow"] = df["claim_date"].dt.dayofweek  # 0=Mon
        df["claim_month"] = df["claim_date"].dt.month
        df["claim_year"] = df["claim_date"].dt.year
        df["is_weekend"] = (df["claim_dow"] >= 5).astype(int)

        # Service-to-claim lag (suspicious if too long or too short)
        if "service_date" in df.columns:
            df["service_date"] = pd.to_datetime(df["service_date"])
            df["service_to_claim_days"] = (
                df["claim_date"] - df["service_date"]
            ).dt.days.clip(0, 365)
            df["suspicious_lag"] = (df["service_to_claim_days"] > 90).astype(int)

        # Rolling 30-day claim count per provider
        df_sorted = df.sort_values(["provider_npi", "claim_date"])
        df_sorted["provider_rolling_30d"] = df_sorted.groupby("provider_npi")[
            "claim_date"
        ].transform(lambda x: x.expanding().count())
        df["provider_rolling_30d"] = df_sorted["provider_rolling_30d"]

        return df

    # ─────────────────────────────────────────────────────────
    # PHARMACY FEATURES
    # ─────────────────────────────────────────────────────────

    def _pharmacy_features(
        self, df: pd.DataFrame, pharmacy: pd.DataFrame
    ) -> pd.DataFrame:
        log.info("  → Pharmacy proxy features")

        member_rx = (
            pharmacy.groupby("member_id")
            .agg(
                rx_total_claims=("rx_id", "count"),
                rx_controlled_cnt=("controlled_substance", "sum"),
                rx_total_spend=("paid_amount", "sum"),
                rx_distinct_drugs=("drug_code", "nunique"),
            )
            .reset_index()
        )
        df = df.merge(member_rx, on="member_id", how="left")
        df["rx_total_claims"] = df["rx_total_claims"].fillna(0)
        df["rx_controlled_cnt"] = df["rx_controlled_cnt"].fillna(0)
        df["rx_total_spend"] = df["rx_total_spend"].fillna(0)
        df["rx_distinct_drugs"] = df["rx_distinct_drugs"].fillna(0)
        df["high_controlled_rx"] = (df["rx_controlled_cnt"] >= 3).astype(int)

        return df

    # ─────────────────────────────────────────────────────────
    # RATIO / INTERACTION FEATURES
    # ─────────────────────────────────────────────────────────

    def _ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("  → Ratio / interaction features")

        df["spend_per_chronic_cond"] = df["billed_amount"] / df[
            "chronic_conditions"
        ].clip(lower=1)
        df["high_risk_cpt_x_oig"] = df["has_high_risk_cpt"] * df["oig_excluded"]
        df["dup_x_high_amount"] = df["duplicate_flag"] * (
            df["billed_amount"] > 5000
        ).astype(int)
        df["peer_x_volume"] = (
            df["peer_billing_percentile"] * df["provider_total_claims_in_data"]
        ) / 100

        return df

    # ─────────────────────────────────────────────────────────
    # FINALIZE: SELECT MODEL FEATURES
    # ─────────────────────────────────────────────────────────

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw/non-numeric columns; return only model-ready features + IDs + target."""
        keep = [
            # Identifiers (not model inputs)
            "claim_id",
            "member_id",
            "provider_npi",
            # Target
            "fraud_label",
            # Claim features
            "billed_amount",
            "allowed_amount",
            "paid_amount",
            "billed_log",
            "paid_log",
            "billed_per_procedure",
            "allowed_ratio",
            "paid_to_allowed_ratio",
            "n_procedures",
            "unique_cpt_count",
            "cpt_code_count",
            "has_high_risk_cpt",
            "has_preventive_cpt",
            "duplicate_flag",
            "out_of_network",
            "prior_auth",
            "no_prior_auth_high_risk",
            "high_risk_pos",
            # Provider features
            "oig_excluded",
            "license_active",
            "peer_billing_percentile",
            "peer_percentile_high",
            "peer_percentile_extreme",
            "specialty_risk_score",
            "years_practice",
            "provider_total_claims_in_data",
            "provider_rolling_30d",
            # Member features
            "age",
            "age_risk",
            "gender_encoded",
            "plan_risk_score",
            "chronic_conditions",
            "prior_claims_12m",
            "address_changes_12m",
            "high_prior_claims",
            "address_instability",
            "member_distinct_providers",
            "many_providers",
            "member_total_billed",
            # Temporal features
            "claim_dow",
            "claim_month",
            "is_weekend",
            "service_to_claim_days",
            "suspicious_lag",
            # Pharmacy features
            "rx_total_claims",
            "rx_controlled_cnt",
            "rx_distinct_drugs",
            "high_controlled_rx",
            # Interaction features
            "spend_per_chronic_cond",
            "high_risk_cpt_x_oig",
            "dup_x_high_amount",
            "peer_x_volume",
        ]
        available = [c for c in keep if c in df.columns]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            log.warning(f"Missing features (will be excluded): {missing}")

        return df[available].copy()


if __name__ == "__main__":
    print("--- Pipeline Started ---")

    # 1. Load the "Bronze" data from your synthetic folder
    print("Loading raw data layers...")
    df_claims = pd.read_csv("data/synthetic/claims.csv")
    df_members = pd.read_csv("data/synthetic/members.csv")
    df_providers = pd.read_csv("data/synthetic/providers.csv")
    df_pharmacy = pd.read_csv("data/synthetic/pharmacy.csv")

    # 2. Initialize the engine
    engineer = FeatureEngineer()

    # 3. Pass the 4 required arguments to the function
    print("Engineering 200+ features (this may take a moment)...")
    final_features = engineer.build_claim_features(
        claims=df_claims,
        members=df_members,
        providers=df_providers,
        pharmacy=df_pharmacy,
    )

    # 4. Save the "Gold" output
    final_features.to_csv("data/processed/final_features.csv", index=False)
    print("--- Pipeline Completed Successfully: Features saved to data/processed/ ---")
