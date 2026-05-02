# config/settings.py
"""
Central configuration for the Health Insurance AI Platform.
All environment-specific values are read from environment variables.
Never hardcode secrets here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatabaseConfig:
    host:     str = os.getenv("DB_HOST",     "localhost")
    port:     int = int(os.getenv("DB_PORT", "5432"))
    name:     str = os.getenv("DB_NAME",     "health_insurance_ai")
    user:     str = os.getenv("DB_USER",     "postgres")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class AWSConfig:
    region:            str = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    s3_bucket:         str = os.getenv("S3_BUCKET",          "health-insurance-ai-data")
    s3_prefix:         str = os.getenv("S3_PREFIX",          "raw")
    sagemaker_role:    str = os.getenv("SAGEMAKER_ROLE",     "")
    account_id:        str = os.getenv("AWS_ACCOUNT_ID",     "")


@dataclass
class ModelConfig:
    fraud_model_version:    str   = os.getenv("FRAUD_MODEL_VERSION",    "latest")
    approval_model_version: str   = os.getenv("APPROVAL_MODEL_VERSION", "latest")
    fraud_threshold:        float = float(os.getenv("FRAUD_THRESHOLD",   "0.55"))
    fpr_alert_threshold:    float = float(os.getenv("FPR_ALERT",         "0.10"))
    psi_warning_threshold:  float = float(os.getenv("PSI_WARNING",       "0.10"))
    psi_critical_threshold: float = float(os.getenv("PSI_CRITICAL",      "0.25"))
    retraining_auc_drop:    float = float(os.getenv("RETRAIN_AUC_DROP",  "0.03"))
    model_dir:              str   = os.getenv("MODEL_DIR",               "models/saved")


@dataclass
class APIConfig:
    host:         str = os.getenv("API_HOST",    "0.0.0.0")
    port:         int = int(os.getenv("API_PORT","8000"))
    workers:      int = int(os.getenv("API_WORKERS","2"))
    log_level:    str = os.getenv("LOG_LEVEL",   "info")
    cors_origins: str = os.getenv("CORS_ORIGINS","*")


@dataclass
class ComplianceConfig:
    anonymization_salt: str = os.getenv("ANON_SALT", "change-me-in-production")
    phi_encryption_key: str = os.getenv("PHI_ENC_KEY","")
    audit_log_path:     str = os.getenv("AUDIT_LOG",  "logs/audit.jsonl")
    hipaa_mode:         bool= os.getenv("HIPAA_MODE","true").lower() == "true"


@dataclass
class AppConfig:
    env:        str             = os.getenv("APP_ENV", "dev")
    debug:      bool            = os.getenv("DEBUG","false").lower() == "true"
    database:   DatabaseConfig  = field(default_factory=DatabaseConfig)
    aws:        AWSConfig       = field(default_factory=AWSConfig)
    model:      ModelConfig     = field(default_factory=ModelConfig)
    api:        APIConfig       = field(default_factory=APIConfig)
    compliance: ComplianceConfig= field(default_factory=ComplianceConfig)


# Singleton
config = AppConfig()
