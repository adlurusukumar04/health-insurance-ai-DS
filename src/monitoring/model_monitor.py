"""
model_monitor.py
----------------
Monitors production model health:
  - Data drift   (PSI — Population Stability Index)
  - Model drift  (AUC-ROC degradation over time)
  - Score distribution shift
  - False positive rate tracking
  - Auto-alert when thresholds breached

Usage:
    python src/monitoring/model_monitor.py --reference data/processed/features_dev.parquet
                                           --current   data/processed/features_current.parquet
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

log = logging.getLogger(__name__)
ALERT_LOG = "models/reports/monitoring_alerts.jsonl"
os.makedirs("models/reports", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI < 0.10  → No significant change
    PSI 0.10–0.25 → Moderate shift — monitor
    PSI > 0.25  → Major shift — investigate / retrain
    """
    def _safe_pct(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        pct = counts / len(arr)
        return np.where(pct == 0, 0.0001, pct)

    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0]  -= 1e-6
    breakpoints[-1] += 1e-6

    pct_exp = _safe_pct(expected, breakpoints)
    pct_act = _safe_pct(actual,   breakpoints)
    psi = np.sum((pct_act - pct_exp) * np.log(pct_act / pct_exp))
    return round(float(psi), 4)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DRIFT MONITOR
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES_TO_MONITOR = [
    "billed_amount", "n_procedures", "peer_billing_percentile",
    "age", "chronic_conditions", "prior_claims_12m",
    "billed_per_procedure", "allowed_ratio",
]

PSI_THRESHOLDS = {"warning": 0.10, "critical": 0.25}


def monitor_feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compute PSI for each monitored feature."""
    results = {}
    alerts  = []

    for feat in NUMERIC_FEATURES_TO_MONITOR:
        if feat not in reference.columns or feat not in current.columns:
            continue

        ref_vals = reference[feat].dropna().values
        cur_vals = current[feat].dropna().values
        if len(ref_vals) < 30 or len(cur_vals) < 30:
            continue

        psi = compute_psi(ref_vals, cur_vals)
        status = "OK"
        if psi >= PSI_THRESHOLDS["critical"]:
            status = "CRITICAL"
            alerts.append({"type": "DATA_DRIFT_CRITICAL", "feature": feat, "psi": psi})
        elif psi >= PSI_THRESHOLDS["warning"]:
            status = "WARNING"
            alerts.append({"type": "DATA_DRIFT_WARNING", "feature": feat, "psi": psi})

        results[feat] = {"psi": psi, "status": status}

    return {"feature_drift": results, "alerts": alerts}


# ─────────────────────────────────────────────────────────────────────────────
# SCORE DISTRIBUTION MONITOR
# ─────────────────────────────────────────────────────────────────────────────

def monitor_score_distribution(reference_scores: np.ndarray,
                                 current_scores:   np.ndarray) -> dict:
    """Check if fraud score distribution has shifted significantly."""
    psi = compute_psi(reference_scores, current_scores)

    ref_mean = float(reference_scores.mean())
    cur_mean = float(current_scores.mean())
    mean_shift = abs(cur_mean - ref_mean) / max(ref_mean, 0.001)

    alerts = []
    if psi >= 0.25:
        alerts.append({"type": "SCORE_DIST_CRITICAL", "psi": psi,
                        "message": "Fraud score distribution has shifted critically — model may be stale"})
    elif psi >= 0.10:
        alerts.append({"type": "SCORE_DIST_WARNING", "psi": psi,
                        "message": "Moderate shift in score distribution — monitor closely"})
    if mean_shift > 0.30:
        alerts.append({"type": "SCORE_MEAN_SHIFT", "mean_shift_pct": round(mean_shift*100, 1),
                        "message": f"Fraud score mean shifted by {mean_shift*100:.1f}%"})

    return {
        "score_psi":      psi,
        "reference_mean": round(ref_mean, 4),
        "current_mean":   round(cur_mean, 4),
        "mean_shift_pct": round(mean_shift * 100, 1),
        "alerts":         alerts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE MONITOR
# ─────────────────────────────────────────────────────────────────────────────

PERF_THRESHOLDS = {
    "auc_roc_drop_warning":  0.02,
    "auc_roc_drop_critical": 0.05,
    "fpr_warning":           0.10,
    "fpr_critical":          0.15,
}


def monitor_model_performance(baseline_metrics: dict, current_metrics: dict) -> dict:
    """Compare current performance to approved baseline. Alert on degradation."""
    alerts = []
    comparisons = {}

    for metric in ["auc_roc", "precision", "recall", "f1"]:
        baseline = baseline_metrics.get(metric, 0)
        current  = current_metrics.get(metric, 0)
        delta    = current - baseline
        comparisons[metric] = {"baseline": baseline, "current": current, "delta": round(delta, 4)}

        if metric == "auc_roc":
            drop = -delta
            if drop >= PERF_THRESHOLDS["auc_roc_drop_critical"]:
                alerts.append({"type": "AUC_CRITICAL",
                                "message": f"AUC-ROC dropped {drop:.3f} — RETRAIN REQUIRED",
                                "action": "TRIGGER_RETRAINING"})
            elif drop >= PERF_THRESHOLDS["auc_roc_drop_warning"]:
                alerts.append({"type": "AUC_WARNING",
                                "message": f"AUC-ROC dropped {drop:.3f} — monitor",
                                "action": "INCREASE_MONITORING_FREQUENCY"})

    fpr = current_metrics.get("false_positive_rate", 0)
    comparisons["false_positive_rate"] = {"current": fpr}
    if fpr >= PERF_THRESHOLDS["fpr_critical"]:
        alerts.append({"type": "FPR_CRITICAL",
                        "message": f"False positive rate {fpr:.3f} exceeds critical threshold",
                        "action": "RECALIBRATE_THRESHOLD"})
    elif fpr >= PERF_THRESHOLDS["fpr_warning"]:
        alerts.append({"type": "FPR_WARNING",
                        "message": f"False positive rate {fpr:.3f} elevated — review threshold",
                        "action": "MONITOR_THRESHOLD"})

    return {"performance_comparison": comparisons, "alerts": alerts}


# ─────────────────────────────────────────────────────────────────────────────
# ALERT DISPATCH
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_alerts(alerts: list, run_id: str) -> None:
    """Write alerts to log file. In production: send to PagerDuty / Slack / SNS."""
    if not alerts:
        log.info("No alerts triggered. Model health: OK")
        return

    for alert in alerts:
        alert["run_id"] = run_id
        alert["ts"]     = datetime.now().isoformat()
        log.warning(f"ALERT [{alert['type']}]: {alert.get('message','')}")
        # Append to JSONL file for audit trail
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(alert) + "\n")

    # In production: integrate with PagerDuty / SNS / Slack
    critical = [a for a in alerts if "CRITICAL" in a["type"]]
    if critical:
        log.error(f"CRITICAL alerts detected: {len(critical)}. Immediate action required.")


# ─────────────────────────────────────────────────────────────────────────────
# FULL MONITORING RUN
# ─────────────────────────────────────────────────────────────────────────────

class ModelMonitor:
    def __init__(self, baseline_metrics: dict):
        self.baseline_metrics = baseline_metrics

    def run(self, reference_df: pd.DataFrame, current_df: pd.DataFrame,
            current_metrics: dict = None) -> dict:

        run_id = f"MON_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log.info(f"=== Model Monitoring Run {run_id} ===")
        all_alerts = []

        # 1. Feature drift
        drift_results = monitor_feature_drift(reference_df, current_df)
        all_alerts.extend(drift_results["alerts"])

        # 2. Performance (if labels available)
        perf_results = {}
        if current_metrics:
            perf_results = monitor_model_performance(self.baseline_metrics, current_metrics)
            all_alerts.extend(perf_results["alerts"])

        # 3. Dispatch
        dispatch_alerts(all_alerts, run_id)

        report = {
            "run_id":           run_id,
            "timestamp":        datetime.now().isoformat(),
            "total_alerts":     len(all_alerts),
            "critical_alerts":  sum(1 for a in all_alerts if "CRITICAL" in a.get("type","")),
            "feature_drift":    drift_results,
            "performance":      perf_results,
            "recommendation":   "RETRAIN" if any("RETRAIN" in str(a) for a in all_alerts) else "OK",
        }

        report_path = f"models/reports/monitoring_{run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Monitoring report saved → {report_path}")

        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/processed/features_dev.parquet")
    parser.add_argument("--current",   default="data/processed/features_dev.parquet")
    args = parser.parse_args()

    ref_df = pd.read_parquet(args.reference)
    cur_df = pd.read_parquet(args.current)

    baseline = {"auc_roc": 0.912, "precision": 0.843, "recall": 0.781,
                 "f1": 0.811, "false_positive_rate": 0.067}
    monitor = ModelMonitor(baseline_metrics=baseline)
    report  = monitor.run(ref_df, cur_df)
    print(json.dumps(report, indent=2))
