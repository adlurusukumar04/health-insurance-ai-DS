import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="AI Fraud Detection Dashboard", layout="wide")
st.title("🛡️ Health Insurance Fraud Investigation Portal")
st.markdown("---")


# 2. Load Model and Data
@st.cache_resource
def load_assets():
    model = joblib.load("models/fraud_detection_model.pkl")
    data = pd.read_csv("data/processed/final_features.csv")
    return model, data


model, df = load_assets()

# 3. Sidebar - Filter Claims
st.sidebar.header("Filter Claims")
risk_threshold = st.sidebar.slider("Risk Score Threshold", 0.0, 1.0, 0.8)

# 4. Processing Predictions
# Get probability for the fraud class (1)
X = df.drop(
    columns=["fraud_label", "claim_id", "member_id", "provider_npi"], errors="ignore"
)
probs = model.predict_proba(X)[:, 1]
df["fraud_probability"] = probs
high_risk_df = df[df["fraud_probability"] >= risk_threshold].sort_values(
    by="fraud_probability", ascending=False
)

# 5. Top Level Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Claims Scored", len(df))
col2.metric("High Risk Flags", len(high_risk_df))
col3.metric("Avg Risk Probability", f"{df['fraud_probability'].mean():.2%}")

# 6. Visualization - Fraud by State/Specialty (Mocking specialty for viz)
st.subheader("Geographic Risk Distribution")
fig = px.histogram(
    high_risk_df,
    x="fraud_probability",
    nbins=20,
    title="Distribution of High Risk Scores",
)
st.plotly_chart(fig, use_container_width=True)

# 7. Investigation Table
st.subheader("🚩 High Risk Claims for Review")
st.write(
    f"The following claims exceed the {risk_threshold} risk threshold and require immediate manual audit."
)

# Display key columns for the investigator
display_cols = [
    "claim_id",
    "fraud_probability",
    "billed_amount",
    "provider_npi",
    "peer_billing_percentile",
    "specialty_risk_score",
]
st.dataframe(
    high_risk_df[display_cols].style.background_gradient(
        subset=["fraud_probability"], cmap="Reds"
    )
)

# 8. Individual Claim Drill-down
st.markdown("---")
st.subheader("🔍 Deep Dive Analysis")
selected_claim = st.selectbox(
    "Select a Claim ID to investigate specific risk factors:", high_risk_df["claim_id"]
)

if selected_claim:
    claim_data = high_risk_df[high_risk_df["claim_id"] == selected_claim].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Provider Details**")
        st.write(f"NPI: {claim_data['provider_npi']}")
        st.write(
            f"Peer Billing Percentile: {claim_data['peer_billing_percentile']:.2f}%"
        )
        st.write(f"License Active: {claim_data['license_active']}")
    with c2:
        st.write("**Clinical Risk Factors**")
        st.write(f"Billed Amount: ${claim_data['billed_amount']:,.2f}")
        st.write(f"High Risk CPT Present: {claim_data['has_high_risk_cpt']}")
        st.write(f"Prior Auth Missing: {claim_data['no_prior_auth_high_risk']}")
