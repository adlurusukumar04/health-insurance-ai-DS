"""
generate_synthetic_data.py
--------------------------
Generates realistic synthetic health insurance data for:
  - Claims
  - Members (beneficiaries)
  - Providers
  - Pharmacy records
  - Fraud labels

All data is fully synthetic. No real PHI is used or stored.
Output: data/synthetic/*.csv
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
N_MEMBERS = 5_000
N_PROVIDERS = 500
N_CLAIMS = 50_000
N_PHARMACY = 15_000
FRAUD_RATE = 0.03  # 3% fraud rate

CPT_CODES = [
    "99213",
    "99214",
    "99232",
    "99233",
    "99285",  # E&M
    "93000",
    "93306",
    "71046",
    "72148",
    "70553",  # Diagnostics
    "27447",
    "27130",
    "29827",
    "43239",
    "47562",  # Surgery
    "90834",
    "90837",
    "90847",
    "96127",
    "99490",  # Behavioral/Wellness
    "G0439",
    "G0444",
    "G0463",
    "T1015",
    "S9083",  # Preventive/Home
]
ICD10_CODES = [
    "I10",
    "E11.9",
    "J18.9",
    "M54.5",
    "F32.9",
    "K21.0",
    "Z23",
    "I25.10",
    "N39.0",
    "J06.9",
    "E78.5",
    "G47.33",
    "M79.3",
    "Z00.00",
    "R05",
]
SPECIALTIES = [
    "Internal Medicine",
    "Family Practice",
    "Cardiology",
    "Orthopedics",
    "Psychiatry",
    "Gastroenterology",
    "Radiology",
    "Emergency Medicine",
    "Neurology",
    "Oncology",
]
PLAN_TYPES = ["HMO", "PPO", "EPO", "HDHP", "POS"]
STATES = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]
DRUG_CODES = [f"NDC{random.randint(10000,99999)}" for _ in range(50)]


def generate_members(n: int) -> pd.DataFrame:
    """Generate member / beneficiary records."""
    records = []
    for i in range(n):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
        age = (datetime.now().date() - dob).days // 365
        records.append(
            {
                "member_id": f"MBR{i+1:06d}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "dob": dob,
                "age": age,
                "gender": random.choice(["M", "F", "Other"]),
                "state": random.choice(STATES),
                "zip_code": fake.zipcode(),
                "plan_type": random.choice(PLAN_TYPES),
                "plan_start_date": fake.date_between(start_date="-5y", end_date="-1y"),
                "chronic_conditions": random.randint(0, 5),
                "prior_claims_12m": random.randint(0, 30),
                "address_changes_12m": random.choices(
                    [0, 1, 2, 3], weights=[70, 20, 7, 3]
                )[0],
            }
        )
    return pd.DataFrame(records)


def generate_providers(n: int) -> pd.DataFrame:
    """Generate provider records."""
    records = []
    for i in range(n):
        is_flagged = random.random() < 0.05  # 5% providers are fraud-prone
        records.append(
            {
                "npi": f"NPI{i+1:010d}",
                "provider_name": fake.company() + " Medical Group",
                "specialty": random.choice(SPECIALTIES),
                "state": random.choice(STATES),
                "zip_code": fake.zipcode(),
                "license_active": random.choices([True, False], weights=[95, 5])[0],
                "oig_excluded": random.choices([False, True], weights=[97, 3])[0],
                "years_practice": random.randint(1, 35),
                "avg_monthly_claims": (
                    random.randint(20, 800)
                    if not is_flagged
                    else random.randint(800, 3000)
                ),
                "peer_billing_percentile": np.clip(
                    random.gauss(50, 20) if not is_flagged else random.gauss(90, 8),
                    1,
                    99,
                ),
                "fraud_prone": is_flagged,  # hidden label for simulation
            }
        )
    return pd.DataFrame(records)


def generate_claims(
    members: pd.DataFrame, providers: pd.DataFrame, n: int
) -> pd.DataFrame:
    """Generate claims with realistic fraud injection."""
    member_ids = members["member_id"].tolist()
    provider_npis = providers["npi"].tolist()
    fraud_prone_npis = providers.loc[providers["fraud_prone"], "npi"].tolist()

    records = []
    for i in range(n):
        is_fraud = random.random() < FRAUD_RATE
        member_id = random.choice(member_ids)
        npi = random.choice(fraud_prone_npis if is_fraud else provider_npis)

        claim_date = fake.date_between(start_date="-3y", end_date="today")
        n_cpt = random.randint(1, 5) if not is_fraud else random.randint(5, 15)
        cpt_codes = random.choices(CPT_CODES, k=n_cpt)
        icd_code = random.choice(ICD10_CODES)

        billed_amt = round(random.uniform(50, 500) * n_cpt, 2)
        if is_fraud:
            billed_amt = round(billed_amt * random.uniform(2.5, 6.0), 2)  # inflated

        allowed_amt = round(billed_amt * random.uniform(0.55, 0.85), 2)
        paid_amt = round(allowed_amt * random.uniform(0.70, 0.95), 2)

        records.append(
            {
                "claim_id": f"CLM{i+1:08d}",
                "member_id": member_id,
                "provider_npi": npi,
                "claim_date": claim_date,
                "service_date": claim_date - timedelta(days=random.randint(0, 7)),
                "icd10_primary": icd_code,
                "cpt_codes": "|".join(cpt_codes),
                "n_procedures": n_cpt,
                "billed_amount": billed_amt,
                "allowed_amount": allowed_amt,
                "paid_amount": paid_amt,
                "claim_type": random.choice(
                    ["Medical", "DME", "Pharmacy", "Behavioral"]
                ),
                "place_of_service": random.choice(["11", "22", "23", "31", "32", "81"]),
                "prior_auth": random.choices([True, False], weights=[30, 70])[0],
                "duplicate_flag": is_fraud and random.random() < 0.3,
                "out_of_network": random.choices([False, True], weights=[80, 20])[0],
                "days_supply": random.randint(1, 90),
                "fraud_label": int(is_fraud),  # ground truth
            }
        )
    return pd.DataFrame(records)


def generate_pharmacy(
    members: pd.DataFrame, providers: pd.DataFrame, n: int
) -> pd.DataFrame:
    """Generate pharmacy claims."""
    records = []
    for i in range(n):
        fill_date = fake.date_between(start_date="-2y", end_date="today")
        records.append(
            {
                "rx_id": f"RX{i+1:08d}",
                "member_id": random.choice(members["member_id"].tolist()),
                "prescriber_npi": random.choice(providers["npi"].tolist()),
                "pharmacy_id": f"PHARM{random.randint(1,200):04d}",
                "drug_code": random.choice(DRUG_CODES),
                "drug_name": fake.word().capitalize()
                + random.choice([" HCL", " SR", " XR", ""]),
                "days_supply": random.choice([7, 14, 30, 60, 90]),
                "quantity": random.randint(10, 360),
                "fill_date": fill_date,
                "refill_number": random.randint(0, 5),
                "billed_amount": round(random.uniform(10, 800), 2),
                "paid_amount": round(random.uniform(5, 600), 2),
                "controlled_substance": random.choices([False, True], weights=[80, 20])[
                    0
                ],
            }
        )
    return pd.DataFrame(records)


def main():
    print("Generating synthetic health insurance data...")

    print("  → Members...")
    members = generate_members(N_MEMBERS)
    members.to_csv(f"{OUTPUT_DIR}/members.csv", index=False)
    print(f"     {len(members):,} members saved")

    print("  → Providers...")
    providers = generate_providers(N_PROVIDERS)
    providers.to_csv(f"{OUTPUT_DIR}/providers.csv", index=False)
    print(f"     {len(providers):,} providers saved")

    print("  → Claims...")
    claims = generate_claims(members, providers, N_CLAIMS)
    claims.to_csv(f"{OUTPUT_DIR}/claims.csv", index=False)
    fraud_pct = claims["fraud_label"].mean() * 100
    print(f"     {len(claims):,} claims saved  |  Fraud rate: {fraud_pct:.2f}%")

    print("  → Pharmacy...")
    pharmacy = generate_pharmacy(members, providers, N_PHARMACY)
    pharmacy.to_csv(f"{OUTPUT_DIR}/pharmacy.csv", index=False)
    print(f"     {len(pharmacy):,} pharmacy records saved")

    print("\nDone! Synthetic data written to data/synthetic/")
    print("WARNING: This data is entirely synthetic. Never use real PHI.")


if __name__ == "__main__":
    main()
