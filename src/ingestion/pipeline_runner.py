"""
pipeline_runner.py
------------------
Orchestrates the full data ingestion and feature engineering pipeline.

Steps:
  1. Load raw synthetic / S3 data
  2. Validate schema and data quality
  3. Clean and standardize
  4. Engineer features
  5. Write feature-store-ready Parquet files to data/processed/

Usage:
    python src/ingestion/pipeline_runner.py [--source synthetic|s3] [--env dev|prod]
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.compliance.anonymizer import Anonymizer  # noqa: E402
from src.processing.feature_engineering import FeatureEngineer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("pipeline_runner")

SYNTHETIC_DIR = "data/synthetic"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_CLAIM_COLS = [
    "claim_id",
    "member_id",
    "provider_npi",
    "claim_date",
    "billed_amount",
    "paid_amount",
    "n_procedures",
    "fraud_label",
]
REQUIRED_MEMBER_COLS = [
    "member_id",
    "age",
    "gender",
    "state",
    "plan_type",
    "chronic_conditions",
]
REQUIRED_PROVIDER_COLS = [
    "npi",
    "specialty",
    "state",
    "oig_excluded",
    "peer_billing_percentile",
]


def validate_schema(df: pd.DataFrame, required_cols: list, name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")
    log.info(
        f"[{name}] Schema validation passed — {len(df):,} rows, {len(df.columns)} cols"
    )


def check_data_quality(df: pd.DataFrame, name: str) -> dict:
    """Return data quality metrics dict."""
    null_pct = df.isnull().mean().to_dict()
    dup_count = df.duplicated().sum()
    report = {
        "name": name,
        "rows": len(df),
        "duplicates": int(dup_count),
        "null_pct": {k: round(v * 100, 2) for k, v in null_pct.items() if v > 0},
    }
    if dup_count > 0:
        log.warning(f"[{name}] Found {dup_count} duplicate rows")
    for col, pct in null_pct.items():
        if pct > 0.05:
            log.warning(f"[{name}] Column '{col}' has {pct*100:.1f}% nulls")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────


def load_synthetic() -> dict:
    log.info("Loading synthetic data from data/synthetic/")
    return {
        "claims": pd.read_csv(
            f"{SYNTHETIC_DIR}/claims.csv", parse_dates=["claim_date", "service_date"]
        ),
        "members": pd.read_csv(
            f"{SYNTHETIC_DIR}/members.csv", parse_dates=["dob", "plan_start_date"]
        ),
        "providers": pd.read_csv(f"{SYNTHETIC_DIR}/providers.csv"),
        "pharmacy": pd.read_csv(
            f"{SYNTHETIC_DIR}/pharmacy.csv", parse_dates=["fill_date"]
        ),
    }


def load_s3(bucket: str, prefix: str) -> dict:
    """Load data from AWS S3 (production path)."""
    from io import BytesIO

    import boto3

    s3 = boto3.client("s3")
    datasets = {}
    for name in ["claims", "members", "providers", "pharmacy"]:
        key = f"{prefix}/{name}.parquet"
        log.info(f"Loading s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        datasets[name] = pd.read_parquet(BytesIO(obj["Body"].read()))
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["claim_id"])
    df["billed_amount"] = df["billed_amount"].clip(lower=0)
    df["paid_amount"] = df["paid_amount"].clip(lower=0)
    df["paid_amount"] = df[["paid_amount", "billed_amount"]].min(axis=1)
    df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")
    df = df.dropna(subset=["claim_id", "member_id", "provider_npi"])
    df["cpt_codes"] = df["cpt_codes"].fillna("")
    return df


def clean_members(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["member_id"])
    df["age"] = df["age"].clip(0, 120)
    df["gender"] = df["gender"].fillna("Unknown")
    return df


def clean_providers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["npi"])
    df["peer_billing_percentile"] = df["peer_billing_percentile"].clip(0, 100)
    df["oig_excluded"] = df["oig_excluded"].fillna(False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(source: str = "synthetic", env: str = "dev") -> None:
    start = datetime.now()
    log.info(f"Pipeline started | source={source} | env={env}")

    # 1. LOAD
    if source == "synthetic":
        datasets = load_synthetic()
    elif source == "s3":
        bucket = os.environ.get("S3_BUCKET", "health-insurance-ai-data")
        prefix = os.environ.get("S3_PREFIX", f"raw/{env}")
        datasets = load_s3(bucket, prefix)
    else:
        raise ValueError(f"Unknown source: {source}")

    # 2. VALIDATE
    validate_schema(datasets["claims"], REQUIRED_CLAIM_COLS, "claims")
    validate_schema(datasets["members"], REQUIRED_MEMBER_COLS, "members")
    validate_schema(datasets["providers"], REQUIRED_PROVIDER_COLS, "providers")

    # 3. QUALITY CHECK
    for name, df in datasets.items():
        check_data_quality(df, name)

    # 4. CLEAN
    log.info("Cleaning datasets...")
    datasets["claims"] = clean_claims(datasets["claims"])
    datasets["members"] = clean_members(datasets["members"])
    datasets["providers"] = clean_providers(datasets["providers"])

    # 5. ANONYMIZE (HIPAA Safe Harbor)
    log.info("Applying HIPAA anonymization...")
    anon = Anonymizer()
    datasets["members"] = anon.anonymize_members(datasets["members"])

    # 6. FEATURE ENGINEERING
    log.info("Engineering features...")
    fe = FeatureEngineer()
    feature_df = fe.build_claim_features(
        claims=datasets["claims"],
        members=datasets["members"],
        providers=datasets["providers"],
        pharmacy=datasets["pharmacy"],
    )
    log.info(f"Feature matrix shape: {feature_df.shape}")

    # 7. SAVE
    out_path = f"{PROCESSED_DIR}/features_{env}.parquet"
    feature_df.to_parquet(out_path, index=False)
    log.info(f"Feature matrix saved → {out_path}")

    # Save cleaned datasets too
    for name, df in datasets.items():
        df.to_parquet(f"{PROCESSED_DIR}/{name}_clean.parquet", index=False)

    elapsed = (datetime.now() - start).total_seconds()
    log.info(
        f"Pipeline complete in {elapsed:.1f}s | {len(feature_df):,} claims processed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Health Insurance AI Pipeline Runner")
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "s3"])
    parser.add_argument("--env", default="dev", choices=["dev", "staging", "prod"])
    args = parser.parse_args()
    run_pipeline(source=args.source, env=args.env)
