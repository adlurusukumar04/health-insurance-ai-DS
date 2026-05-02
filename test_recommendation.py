import sys
import pandas as pd
sys.path.insert(0, '.')
from src.models.recommendation_engine import PlanRecommender

print("PLAN RECOMMENDATION ENGINE TEST")
print("-" * 40)

# Create sample members data
members_data = {
    'member_id': ['MBR001', 'MBR002', 'MBR003', 'MBR004', 'MBR005'],
    'age': [25, 65, 45, 35, 55],
    'chronic_conditions': [0, 5, 2, 1, 4],
    'prior_claims_12m': [2, 25, 10, 5, 20],
    'gender_encoded': [1, 0, 1, 0, 1],
    'plan_risk_score': [1, 4, 2, 2, 3]
}
members_df = pd.DataFrame(members_data)

# Create sample interactions
interactions_data = {
    'member_id': ['MBR001', 'MBR002', 'MBR003', 'MBR004', 'MBR005'],
    'plan_id': ['PLN_HDHP_HSA', 'PLN_HMO_SENIOR', 'PLN_PPO_STD',
                'PLN_EPO_MID', 'PLN_PPO_PREM']
}
interactions_df = pd.DataFrame(interactions_data)

# Initialize and fit recommender
print("\nFitting recommender...")
recommender = PlanRecommender()
recommender.fit(members_df, interactions_df=interactions_df)
print("Recommender fitted!")

# Test 1 - Young healthy member
print("\nTest 1 - Young Healthy Member (age=25, no chronic conditions):")
features_young = {
    'age': 25,
    'chronic_conditions': 0,
    'prior_claims_12m': 1,
    'gender_encoded': 1,
    'plan_risk_score': 1,
    'budget_priority': True,
    'needs_dental': False,
    'needs_vision': False,
    'income_band': 'medium'
}
recs = recommender.recommend(
    member_features=features_young,
    top_n=3
)
for r in recs:
    print(f"  Rank {r['rank']}: {r['plan_id']}")
    print(f"    Type:    {r['plan_type']}")
    print(f"    Premium: ${r['monthly_premium']}/month")
    print(f"    Score:   {r['hybrid_score']}")
    print(f"    Reason:  {r['reason']}")

# Test 2 - Senior member with chronic conditions
print("\nTest 2 - Senior Member (age=68, chronic conditions=5):")
features_senior = {
    'age': 68,
    'chronic_conditions': 5,
    'prior_claims_12m': 28,
    'gender_encoded': 0,
    'plan_risk_score': 4,
    'budget_priority': False,
    'needs_dental': True,
    'needs_vision': True,
    'income_band': 'medium'
}
recs = recommender.recommend(
    member_features=features_senior,
    top_n=3
)
for r in recs:
    print(f"  Rank {r['rank']}: {r['plan_id']}")
    print(f"    Type:    {r['plan_type']}")
    print(f"    Premium: ${r['monthly_premium']}/month")
    print(f"    Score:   {r['hybrid_score']}")
    print(f"    Reason:  {r['reason']}")

# Test 3 - Budget conscious member
print("\nTest 3 - Budget Conscious Member (low income):")
features_budget = {
    'age': 30,
    'chronic_conditions': 1,
    'prior_claims_12m': 3,
    'gender_encoded': 1,
    'plan_risk_score': 2,
    'budget_priority': True,
    'needs_dental': False,
    'needs_vision': False,
    'income_band': 'low'
}
recs = recommender.recommend(
    member_features=features_budget,
    top_n=3
)
for r in recs:
    print(f"  Rank {r['rank']}: {r['plan_id']}")
    print(f"    Type:    {r['plan_type']}")
    print(f"    Premium: ${r['monthly_premium']}/month")
    print(f"    Score:   {r['hybrid_score']}")
    print(f"    Reason:  {r['reason']}")

print("\nRecommendation Engine Done!")