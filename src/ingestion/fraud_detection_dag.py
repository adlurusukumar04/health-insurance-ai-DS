"""
fraud_detection_dag.py
----------------------
Apache Airflow DAG for the full Health Insurance AI pipeline.

Schedule: Daily at 02:00 UTC
Tasks:
  1. ingest_data          — Pull from S3 / CMS
  2. validate_quality     — Schema + null checks
  3. anonymize_phi        — HIPAA Safe Harbor
  4. engineer_features    — Build feature matrix
  5. score_batch_claims   — Run fraud model on overnight claims
  6. monitor_model        — Check for drift / performance degradation
  7. generate_report      — Daily fraud summary report
  8. notify_team          — Slack / email alerts on high-priority findings

Conditional branches:
  - If monitor detects drift → trigger_retraining_dag
  - If critical fraud found  → notify_siu_immediately
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT ARGS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["ds-alerts@your-org.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(hours=2),
}


# ─────────────────────────────────────────────────────────────────────────────
# PYTHON CALLABLES
# ─────────────────────────────────────────────────────────────────────────────


def ingest_data(**context):
    """Pull yesterday's claims from S3 data lake."""
    import logging
    from io import BytesIO

    import boto3
    import pandas as pd

    log = logging.getLogger(__name__)
    execution_date = context["ds"]  # e.g. "2024-06-15"
    bucket = "health-insurance-ai-data"
    prefix = f"raw/claims/dt={execution_date}"

    log.info(f"Ingesting claims for {execution_date} from s3://{bucket}/{prefix}")
    # s3 = boto3.client("s3")
    # ... actual S3 pull logic here ...
    log.info("Ingestion complete (demo mode — skipping actual S3)")

    # Push partition date to XCom for downstream tasks
    context["task_instance"].xcom_push(key="partition_date", value=execution_date)
    return execution_date


def validate_quality(**context):
    """Run schema validation and data quality checks."""
    import logging

    log = logging.getLogger(__name__)
    partition_date = context["task_instance"].xcom_pull(key="partition_date")
    log.info(f"Validating data quality for {partition_date}")

    # Simulate quality check
    quality_ok = True
    null_pct = 0.02  # 2% nulls — within tolerance

    if null_pct > 0.10:
        raise ValueError(
            f"Data quality FAILED: null rate {null_pct:.1%} exceeds 10% threshold"
        )

    log.info(f"Data quality OK — null rate: {null_pct:.1%}")
    return {"quality_ok": quality_ok, "null_pct": null_pct}


def anonymize_phi(**context):
    """Apply HIPAA Safe Harbor anonymization."""
    import logging
    import os
    import sys

    sys.path.insert(0, "/app")

    from src.compliance.anonymizer import Anonymizer

    log = logging.getLogger(__name__)
    log.info("Applying HIPAA anonymization to member data")
    # In production: read from S3, anonymize, write back
    log.info("Anonymization complete")


def engineer_features(**context):
    """Build feature matrix from cleaned data."""
    import logging

    log = logging.getLogger(__name__)
    partition_date = context["task_instance"].xcom_pull(key="partition_date")
    log.info(f"Engineering features for {partition_date}")
    # In production: call feature_engineering.py on partition
    log.info("Feature engineering complete")
    return {"feature_count": 42, "claim_count": 5000}


def score_batch_claims(**context):
    """Run fraud model on overnight claim batch."""
    import logging

    log = logging.getLogger(__name__)
    log.info("Scoring overnight claim batch")

    # In production: call FastAPI /v1/claims/batch-score
    high_risk_count = 37
    total_claims = 5000
    fraud_alert_rate = high_risk_count / total_claims

    log.info(
        f"Batch scoring complete — {total_claims} claims | {high_risk_count} high-risk alerts"
    )

    context["task_instance"].xcom_push(key="high_risk_count", value=high_risk_count)
    context["task_instance"].xcom_push(key="fraud_alert_rate", value=fraud_alert_rate)
    return {"high_risk": high_risk_count, "total": total_claims}


def run_model_monitor(**context):
    """Check for model drift and performance degradation."""
    import logging

    log = logging.getLogger(__name__)
    log.info("Running model monitoring checks")

    # Simulated PSI check
    max_psi = 0.08  # within threshold

    context["task_instance"].xcom_push(key="max_psi", value=max_psi)
    context["task_instance"].xcom_push(key="needs_retrain", value=(max_psi >= 0.25))
    log.info(
        f"Model monitoring complete — max PSI: {max_psi} | retrain needed: {max_psi >= 0.25}"
    )
    return {"max_psi": max_psi, "needs_retrain": max_psi >= 0.25}


def branch_on_drift(**context):
    """Branch: trigger retraining if drift detected."""
    needs_retrain = context["task_instance"].xcom_pull(key="needs_retrain")
    if needs_retrain:
        return "trigger_retraining"
    return "generate_report"


def trigger_retraining(**context):
    """Trigger model retraining DAG via Airflow API."""
    import logging

    log = logging.getLogger(__name__)
    log.info("DRIFT DETECTED — Triggering model retraining DAG")
    # In production: trigger via Airflow REST API or TriggerDagRunOperator
    # requests.post("http://airflow:8080/api/v1/dags/fraud_model_retraining/dagRuns", ...)
    log.info("Retraining DAG triggered successfully")


def generate_report(**context):
    """Generate daily fraud summary report."""
    import logging

    log = logging.getLogger(__name__)
    partition_date = context["task_instance"].xcom_pull(key="partition_date")
    high_risk_count = context["task_instance"].xcom_pull(key="high_risk_count")
    fraud_alert_rate = context["task_instance"].xcom_pull(key="fraud_alert_rate")

    report = {
        "date": partition_date,
        "total_claims": 5000,
        "high_risk_alerts": high_risk_count,
        "alert_rate": f"{(fraud_alert_rate or 0)*100:.2f}%",
        "auto_approved": 4500,
        "sent_to_review": 463,
        "flagged_for_siu": high_risk_count,
    }
    log.info(f"Daily Report: {report}")
    return report


def notify_siu_if_critical(**context):
    """Send immediate SIU notification for very high risk claims."""
    import logging

    log = logging.getLogger(__name__)
    high_risk = context["task_instance"].xcom_pull(key="high_risk_count") or 0
    if high_risk > 50:
        log.warning(
            f"SIU ALERT: {high_risk} high-risk claims require immediate review!"
        )
        # In production: send Slack / PagerDuty / email
    log.info("SIU notification check complete")


# ─────────────────────────────────────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="health_insurance_fraud_detection_daily",
    default_args=DEFAULT_ARGS,
    description="Daily health insurance fraud detection pipeline",
    schedule_interval="0 2 * * *",  # 02:00 UTC daily
    catchup=False,
    max_active_runs=1,
    tags=["fraud-detection", "health-insurance", "ml", "daily"],
) as dag:

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )
    t_validate = PythonOperator(
        task_id="validate_quality",
        python_callable=validate_quality,
    )
    t_anonymize = PythonOperator(
        task_id="anonymize_phi",
        python_callable=anonymize_phi,
    )
    t_features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )
    t_score = PythonOperator(
        task_id="score_batch_claims",
        python_callable=score_batch_claims,
    )
    t_monitor = PythonOperator(
        task_id="run_model_monitor",
        python_callable=run_model_monitor,
    )
    t_branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
    )
    t_retrain = PythonOperator(
        task_id="trigger_retraining",
        python_callable=trigger_retraining,
    )
    t_report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    t_notify_siu = PythonOperator(
        task_id="notify_siu_if_critical",
        python_callable=notify_siu_if_critical,
    )

    # ── DAG Dependencies ───────────────────────────────────────────
    (
        start
        >> t_ingest
        >> t_validate
        >> t_anonymize
        >> t_features
        >> t_score
        >> t_monitor
        >> t_branch
    )

    t_branch >> t_retrain >> t_report
    t_branch >> t_report

    t_report >> t_notify_siu >> end
