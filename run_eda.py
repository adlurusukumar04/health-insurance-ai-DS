import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder for charts
os.makedirs('data/eda_charts', exist_ok=True)

print("EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Load data
claims    = pd.read_csv('data/synthetic/claims.csv')
members   = pd.read_csv('data/synthetic/members.csv')
providers = pd.read_csv('data/synthetic/providers.csv')
pharmacy  = pd.read_csv('data/synthetic/pharmacy.csv')

print("Claims shape:   ", claims.shape)
print("Members shape:  ", members.shape)
print("Providers shape:", providers.shape)
print("Pharmacy shape: ", pharmacy.shape)

# ── Chart 1: Fraud Distribution ──────────────────────────────
print("\nGenerating Chart 1 - Fraud Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

claims['fraud_label'].value_counts().plot(
    kind='bar', ax=axes[0],
    color=['#2ecc71', '#e74c3c'],
    edgecolor='black'
)
axes[0].set_title('Claim Label Distribution', fontsize=14, fontweight='bold')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[0].set_ylabel('Count')

for label, color, name in [(0, '#2ecc71', 'Legitimate'), (1, '#e74c3c', 'Fraud')]:
    subset = claims[claims['fraud_label'] == label]['billed_amount']
    axes[1].hist(subset.clip(0, 20000), bins=50,
                 alpha=0.6, label=name, color=color)
axes[1].set_title('Billed Amount by Label', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Billed Amount ($)')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/eda_charts/01_fraud_distribution.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/01_fraud_distribution.png")

# ── Chart 2: Fraud by Claim Type ─────────────────────────────
print("Generating Chart 2 - Fraud by Claim Type...")
fig, ax = plt.subplots(figsize=(10, 5))
fraud_by_type = claims.groupby('claim_type')['fraud_label'].mean() * 100
fraud_by_type.sort_values(ascending=False).plot(
    kind='bar', ax=ax,
    color='#e74c3c', edgecolor='black'
)
ax.set_title('Fraud Rate by Claim Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Claim Type')
ax.set_ylabel('Fraud Rate (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('data/eda_charts/02_fraud_by_type.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/02_fraud_by_type.png")

# ── Chart 3: Billed Amount Distribution ──────────────────────
print("Generating Chart 3 - Billed Amount Stats...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

claims['billed_amount'].clip(0, 20000).hist(
    bins=50, ax=axes[0], color='#3498db', edgecolor='black'
)
axes[0].set_title('Billed Amount Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Billed Amount ($)')
axes[0].set_ylabel('Count')

claims.groupby('fraud_label')['billed_amount'].mean().plot(
    kind='bar', ax=axes[1],
    color=['#2ecc71', '#e74c3c'], edgecolor='black'
)
axes[1].set_title('Average Billed Amount by Label', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[1].set_ylabel('Average Amount ($)')
plt.tight_layout()
plt.savefig('data/eda_charts/03_billed_amounts.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/03_billed_amounts.png")

# ── Chart 4: Provider Analysis ───────────────────────────────
print("Generating Chart 4 - Provider Analysis...")
df = claims.merge(
    providers[['npi', 'specialty', 'peer_billing_percentile', 'oig_excluded']],
    left_on='provider_npi', right_on='npi', how='left'
)
fraud_by_spec = df.groupby('specialty')['fraud_label'].mean() * 100

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#e74c3c' if v > 3 else '#3498db' for v in fraud_by_spec.values]
fraud_by_spec.sort_values(ascending=False).plot(
    kind='barh', ax=ax, color=colors
)
ax.set_title('Fraud Rate by Provider Specialty', fontsize=14, fontweight='bold')
ax.set_xlabel('Fraud Rate (%)')
plt.tight_layout()
plt.savefig('data/eda_charts/04_fraud_by_specialty.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/04_fraud_by_specialty.png")

# ── Chart 5: Member Analysis ─────────────────────────────────
print("Generating Chart 5 - Member Analysis...")
df2 = claims.merge(
    members[['member_id', 'age', 'chronic_conditions', 'plan_type']],
    on='member_id', how='left'
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for label, color, name in [(0, '#2ecc71', 'Legit'), (1, '#e74c3c', 'Fraud')]:
    df2[df2['fraud_label'] == label]['age'].dropna().plot(
        kind='hist', bins=20, ax=axes[0],
        alpha=0.6, color=color, label=name
    )
axes[0].set_title('Member Age by Label', fontweight='bold')
axes[0].set_xlabel('Age')
axes[0].legend()

chron_fraud = df2.groupby('chronic_conditions')['fraud_label'].mean() * 100
chron_fraud.plot(
    kind='bar', ax=axes[1],
    color='#e74c3c', edgecolor='black'
)
axes[1].set_title('Fraud Rate by Chronic Conditions', fontweight='bold')
axes[1].set_xlabel('Chronic Conditions Count')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plan_fraud = df2.groupby('plan_type')['fraud_label'].mean() * 100
plan_fraud.sort_values(ascending=False).plot(
    kind='bar', ax=axes[2],
    color='#3498db', edgecolor='black'
)
axes[2].set_title('Fraud Rate by Plan Type', fontweight='bold')
axes[2].set_xlabel('Plan Type')
axes[2].set_ylabel('Fraud Rate (%)')
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('data/eda_charts/05_member_analysis.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/05_member_analysis.png")

# ── Chart 6: Correlation Matrix ──────────────────────────────
print("Generating Chart 6 - Correlation Matrix...")
numeric_cols = [
    'billed_amount', 'paid_amount', 'n_procedures',
    'duplicate_flag', 'out_of_network', 'fraud_label'
]
corr = claims[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f',
    cmap='RdYlGn_r', center=0, ax=ax, linewidths=0.5
)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data/eda_charts/06_correlation_matrix.png', dpi=150)
plt.close()
print("Saved: data/eda_charts/06_correlation_matrix.png")

# ── Summary Statistics ───────────────────────────────────────
print("\n" + "=" * 40)
print("EDA SUMMARY")
print("=" * 40)
print("Total Claims:     ", len(claims))
print("Fraud Claims:     ", claims['fraud_label'].sum())
print("Fraud Rate:       ", str(round(claims['fraud_label'].mean() * 100, 2)) + "%")
print("Avg Billed (Legit):", round(claims[claims['fraud_label']==0]['billed_amount'].mean(), 2))
print("Avg Billed (Fraud):", round(claims[claims['fraud_label']==1]['billed_amount'].mean(), 2))
print("Total Members:    ", len(members))
print("Total Providers:  ", len(providers))
print("OIG Excluded:     ", providers['oig_excluded'].sum())
print("\nAll charts saved to: data/eda_charts/")
print("\nEDA Complete!")