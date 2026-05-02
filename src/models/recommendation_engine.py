"""
recommendation_engine.py
-------------------------
Personalized health insurance plan recommendation system.

Algorithms:
  - Collaborative Filtering (KNN-based) — "members like you chose..."
  - Content-Based Filtering             — plan features vs member needs
  - Hybrid Ensemble                     — combines both signals

Features used:
  - Member: age, chronic conditions, prior utilization, family size, income band
  - Plan:   premium, deductible, copays, network size, drug coverage, specialist access

Output:
  - Top-N recommended plan IDs
  - Confidence scores
  - Human-readable reason for each recommendation

Usage:
    from src.models.recommendation_engine import PlanRecommender
    rec = PlanRecommender()
    rec.fit(members_df, plans_df, interactions_df)
    recs = rec.recommend(member_id="MBR000001", top_n=3)
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

log = logging.getLogger(__name__)

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("scikit-learn not available — recommendation engine in demo mode")


# ─────────────────────────────────────────────────────────────────────────────
# PLAN CATALOG (demo plans — replace with real plan DB in production)
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PLANS = pd.DataFrame(
    [
        {
            "plan_id": "PLN_HMO_BASIC",
            "plan_type": "HMO",
            "monthly_premium": 280,
            "deductible": 1500,
            "copay_primary": 20,
            "copay_specialist": 50,
            "drug_coverage_tier": 3,
            "network_size": "small",
            "mental_health": True,
            "dental": False,
            "vision": False,
            "out_of_pocket_max": 6000,
        },
        {
            "plan_id": "PLN_PPO_STD",
            "plan_type": "PPO",
            "monthly_premium": 420,
            "deductible": 2000,
            "copay_primary": 30,
            "copay_specialist": 70,
            "drug_coverage_tier": 3,
            "network_size": "large",
            "mental_health": True,
            "dental": False,
            "vision": True,
            "out_of_pocket_max": 7000,
        },
        {
            "plan_id": "PLN_PPO_PREM",
            "plan_type": "PPO",
            "monthly_premium": 680,
            "deductible": 500,
            "copay_primary": 15,
            "copay_specialist": 35,
            "drug_coverage_tier": 4,
            "network_size": "large",
            "mental_health": True,
            "dental": True,
            "vision": True,
            "out_of_pocket_max": 4000,
        },
        {
            "plan_id": "PLN_HDHP_HSA",
            "plan_type": "HDHP",
            "monthly_premium": 180,
            "deductible": 4000,
            "copay_primary": 0,
            "copay_specialist": 0,
            "drug_coverage_tier": 2,
            "network_size": "medium",
            "mental_health": True,
            "dental": False,
            "vision": False,
            "out_of_pocket_max": 8000,
        },
        {
            "plan_id": "PLN_EPO_MID",
            "plan_type": "EPO",
            "monthly_premium": 360,
            "deductible": 1000,
            "copay_primary": 25,
            "copay_specialist": 60,
            "drug_coverage_tier": 3,
            "network_size": "medium",
            "mental_health": True,
            "dental": False,
            "vision": True,
            "out_of_pocket_max": 5500,
        },
        {
            "plan_id": "PLN_HMO_SENIOR",
            "plan_type": "HMO",
            "monthly_premium": 320,
            "deductible": 800,
            "copay_primary": 10,
            "copay_specialist": 30,
            "drug_coverage_tier": 4,
            "network_size": "medium",
            "mental_health": True,
            "dental": True,
            "vision": True,
            "out_of_pocket_max": 3500,
        },
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT-BASED SCORER
# ─────────────────────────────────────────────────────────────────────────────


class ContentBasedScorer:
    """Score plans against member profile based on feature alignment."""

    def score(self, member: Dict, plans: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of scores (0–1) for each plan, indexed by plan_id.
        Higher = better fit for this member.
        """
        scores = {}
        age = member.get("age", 40)
        chronic = member.get("chronic_conditions", 0)
        prior_claims = member.get("prior_claims_12m", 5)
        income_band = member.get("income_band", "medium")  # low/medium/high
        needs_mental = member.get("needs_mental_health", False)
        needs_dental = member.get("needs_dental", False)
        needs_vision = member.get("needs_vision", False)
        budget_priority = member.get("budget_priority", False)  # prefers low premium
        low_deductible = (
            chronic > 2 or prior_claims > 15
        )  # high utilizer → low deductible

        for _, plan in plans.iterrows():
            score = 0.0

            # Budget sensitivity
            if budget_priority or income_band == "low":
                score += max(0, (700 - plan["monthly_premium"]) / 700) * 0.25
            else:
                score += 0.15  # neutral

            # Deductible preference
            if low_deductible:
                score += max(0, (5000 - plan["deductible"]) / 5000) * 0.25
            else:
                # Healthy member benefits from HDHP / high deductible + HSA
                score += (plan["deductible"] / 5000) * 0.10

            # Age-based: seniors benefit from low copay
            if age >= 60:
                score += max(0, (80 - plan["copay_specialist"]) / 80) * 0.20
            else:
                score += 0.10

            # Coverage needs
            if needs_mental and plan["mental_health"]:
                score += 0.10
            if needs_dental and plan["dental"]:
                score += 0.10
            if needs_vision and plan["vision"]:
                score += 0.05

            # Drug coverage for chronic conditions
            if chronic >= 3:
                score += (plan["drug_coverage_tier"] / 4) * 0.15

            scores[plan["plan_id"]] = min(score, 1.0)

        return pd.Series(scores)


# ─────────────────────────────────────────────────────────────────────────────
# COLLABORATIVE FILTERING (KNN)
# ─────────────────────────────────────────────────────────────────────────────


class CollaborativeFilteringScorer:
    """
    KNN collaborative filtering:
    Find similar members and recommend what they chose.
    """

    MEMBER_FEATURES = [
        "age",
        "chronic_conditions",
        "prior_claims_12m",
        "gender_encoded",
        "plan_risk_score",
    ]

    def __init__(self, k: int = 10):
        self.k = k
        self.knn: Optional[NearestNeighbors] = None
        self.scaler = MinMaxScaler()
        self.member_matrix: Optional[np.ndarray] = None
        self.member_ids: List[str] = []
        self.member_plans: Dict[str, str] = {}  # member_id → current plan_id
        self._fitted = False

    def fit(self, members_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        """
        Fit KNN on member feature matrix.
        interactions_df: member_id, plan_id (plan chosen / renewed)
        """
        if not HAS_SKLEARN:
            log.warning("scikit-learn not available — CF scorer not fitted")
            return

        available_features = [
            f for f in self.MEMBER_FEATURES if f in members_df.columns
        ]
        X = members_df[available_features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(X)), metric="euclidean", algorithm="auto"
        )
        self.knn.fit(X_scaled)
        self.member_matrix = X_scaled
        self.member_ids = members_df["member_id"].tolist()
        self._feature_cols = available_features

        # Build member → plan lookup
        for _, row in interactions_df.iterrows():
            self.member_plans[row["member_id"]] = row["plan_id"]

        self._fitted = True
        log.info(f"CF scorer fitted on {len(self.member_ids)} members")

    def recommend(self, member_features: Dict, plans: pd.DataFrame) -> pd.Series:
        """Score plans based on similar members' choices."""
        if not self._fitted or not HAS_SKLEARN:
            return pd.Series({pid: 0.5 for pid in plans["plan_id"]})

        available = [f for f in self._feature_cols if f in member_features]
        X_member = np.array([[member_features.get(f, 0) for f in self._feature_cols]])
        X_scaled = self.scaler.transform(X_member)

        distances, indices = self.knn.kneighbors(X_scaled)
        plan_votes: Dict[str, float] = {}

        for dist, idx in zip(distances[0], indices[0]):
            neighbor_id = self.member_ids[idx]
            neighbor_plan = self.member_plans.get(neighbor_id)
            if neighbor_plan:
                weight = 1.0 / max(dist + 0.01, 0.001)
                plan_votes[neighbor_plan] = plan_votes.get(neighbor_plan, 0) + weight

        # Normalize to [0, 1]
        if plan_votes:
            max_votes = max(plan_votes.values())
            plan_votes = {k: v / max_votes for k, v in plan_votes.items()}

        return pd.Series({pid: plan_votes.get(pid, 0.0) for pid in plans["plan_id"]})


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────


class PlanRecommender:
    """
    Hybrid plan recommender combining content-based and collaborative filtering.

    Weights:
      - content_weight   = 0.6  (plan feature alignment)
      - collab_weight    = 0.4  (similar members' choices)
    """

    def __init__(self, content_weight: float = 0.6, collab_weight: float = 0.4):
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.content_scorer = ContentBasedScorer()
        self.cf_scorer = CollaborativeFilteringScorer()
        self.plans = DEMO_PLANS.copy()
        self._member_lookup: Dict[str, Dict] = {}
        self._fitted = False

    def fit(
        self,
        members_df: pd.DataFrame,
        plans_df: Optional[pd.DataFrame] = None,
        interactions_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fit the recommender.
        members_df:      member features
        plans_df:        plan catalog (defaults to DEMO_PLANS)
        interactions_df: member→plan choices history
        """
        if plans_df is not None:
            self.plans = plans_df.copy()

        # Build member lookup
        for _, row in members_df.iterrows():
            self._member_lookup[row["member_id"]] = row.to_dict()

        # Fit CF scorer
        if interactions_df is not None:
            self.cf_scorer.fit(members_df, interactions_df)

        self._fitted = True
        log.info(
            f"PlanRecommender fitted — {len(members_df)} members | {len(self.plans)} plans"
        )

    def recommend(
        self, member_id: str = None, member_features: Dict = None, top_n: int = 3
    ) -> List[Dict]:
        """
        Get top-N plan recommendations.

        Args:
            member_id:       Look up features from fitted member data
            member_features: Provide features directly (overrides member_id lookup)
            top_n:           Number of recommendations to return

        Returns:
            List of dicts with plan details, score, and reason
        """
        if member_features is None:
            if member_id and member_id in self._member_lookup:
                member_features = self._member_lookup[member_id]
            else:
                member_features = {}

        # Score all plans
        content_scores = self.content_scorer.score(member_features, self.plans)
        cf_scores = self.cf_scorer.recommend(member_features, self.plans)

        # Hybrid score
        hybrid_scores = (
            self.content_weight * content_scores + self.collab_weight * cf_scores
        ).sort_values(ascending=False)

        results = []
        for plan_id in hybrid_scores.head(top_n).index:
            plan_row = self.plans[self.plans["plan_id"] == plan_id]
            if plan_row.empty:
                continue
            plan = plan_row.iloc[0].to_dict()
            reason = self._generate_reason(
                member_features,
                plan,
                content_scores.get(plan_id, 0),
                cf_scores.get(plan_id, 0),
            )
            results.append(
                {
                    "plan_id": plan_id,
                    "plan_type": plan["plan_type"],
                    "monthly_premium": plan["monthly_premium"],
                    "deductible": plan["deductible"],
                    "hybrid_score": round(float(hybrid_scores[plan_id]), 4),
                    "content_score": round(float(content_scores.get(plan_id, 0)), 4),
                    "cf_score": round(float(cf_scores.get(plan_id, 0)), 4),
                    "reason": reason,
                    "rank": len(results) + 1,
                }
            )

        return results

    def _generate_reason(
        self, member: Dict, plan: Dict, content_score: float, cf_score: float
    ) -> str:
        """Generate a human-readable explanation for the recommendation."""
        reasons = []
        age = member.get("age", 40)
        chronic = member.get("chronic_conditions", 0)
        budget_pref = member.get("budget_priority", False)

        if budget_pref and plan["monthly_premium"] < 300:
            reasons.append(f"low monthly premium (${plan['monthly_premium']})")
        if chronic >= 3 and plan["deductible"] < 1000:
            reasons.append(
                f"low deductible ideal for frequent care (${plan['deductible']})"
            )
        if age >= 60 and plan["copay_specialist"] < 40:
            reasons.append(
                f"low specialist copay suited for your age (${plan['copay_specialist']})"
            )
        if plan["dental"] and member.get("needs_dental"):
            reasons.append("includes dental coverage")
        if plan["vision"] and member.get("needs_vision"):
            reasons.append("includes vision coverage")
        if cf_score > 0.6:
            reasons.append("popular with members who have a similar profile")
        if not reasons:
            reasons.append(f"strong overall match for your health profile")

        return "Recommended because: " + "; ".join(reasons) + "."

    def get_ndcg(self, recommendations: List[Dict], relevant_plan: str) -> float:
        """
        Compute NDCG@N — relevance metric for recommendation quality.
        relevant_plan: the plan the member actually chose (ground truth).
        """
        plan_ids = [r["plan_id"] for r in recommendations]
        if relevant_plan not in plan_ids:
            return 0.0
        rank = plan_ids.index(relevant_plan) + 1
        dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(2)  # best possible: relevant at rank 1
        return round(dcg / idcg, 4)
