import pandas as pd
import sys
sys.path.insert(0, '.')
from src.models.claim_approval_model import ClaimApprovalModel

print("CLAIM APPROVAL MODEL TRAINING")
print("-" * 40)

# Load features
df = pd.read_parquet('data/processed/features_dev.parquet')
print("Shape:", df.shape)

# Train model
print("Training...")
model = ClaimApprovalModel(model_type='xgboost')
metrics = model.fit(df, target_col='fraud_label')
print("AUC-ROC:", metrics['auc_roc'])
print("F1 Score:", metrics['f1'])

# Save model
path = model.save()
print("Model saved:", path)

# Low risk claim test
print("\nLow Risk Claim Test:")
low_risk = {
    'age': 35,
    'chronic_conditions': 0,
    'prior_claims_12m': 2,
    'billed_amount': 150.0,
    'n_procedures': 1,
    'has_high_risk_cpt': 0,
    'out_of_network': 0,
    'prior_auth': 1,
    'oig_excluded': 0,
    'peer_billing_percentile': 45.0,
    'specialty_risk_score': 1.0,
    'plan_risk_score': 2.0,
    'age_risk': 2.0,
    'duplicate_flag': 0,
    'allowed_ratio': 0.85,
    'billed_per_procedure': 150.0,
    'is_weekend': 0,
    'suspicious_lag': 0,
    'high_controlled_rx': 0,
    'high_prior_claims': 0
}
result = model.predict(low_risk, fraud_score=0.05)
print("Approval Probability:", result['approval_probability'])
print("Risk Band:", result['risk_band'])
print("Recommendation:", result['recommendation'])

# High risk claim test
print("\nHigh Risk Claim Test:")
high_risk = {
    'age': 45,
    'chronic_conditions': 0,
    'prior_claims_12m': 30,
    'billed_amount': 45000.0,
    'n_procedures': 15,
    'has_high_risk_cpt': 1,
    'out_of_network': 1,
    'prior_auth': 0,
    'oig_excluded': 1,
    'peer_billing_percentile': 98.0,
    'specialty_risk_score': 5.0,
    'plan_risk_score': 4.0,
    'age_risk': 3.0,
    'duplicate_flag': 1,
    'allowed_ratio': 0.10,
    'billed_per_procedure': 3000.0,
    'is_weekend': 1,
    'suspicious_lag': 1,
    'high_controlled_rx': 1,
    'high_prior_claims': 1
}
result = model.predict(high_risk, fraud_score=0.90)
print("Approval Probability:", result['approval_probability'])
print("Risk Band:", result['risk_band'])
print("Recommendation:", result['recommendation'])

print("\nDone!")