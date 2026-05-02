"""
anonymizer.py
-------------
HIPAA Safe Harbor de-identification for member / patient data.

The HIPAA Safe Harbor method requires removal / transformation of 18 PHI identifiers:
  1. Names                   10. Account numbers
  2. Geographic < state      11. Certificate/license numbers
  3. Dates (except year)     12. VINs
  4. Phone numbers           13. Device identifiers
  5. Fax numbers             14. URLs
  6. Email addresses         15. IP addresses
  7. SSNs                    16. Biometric identifiers
  8. Medical record numbers  17. Full-face photos
  9. Health plan bene IDs    18. Any unique identifying number

This module handles the data-level PHI elements applicable to tabular data.
"""

import hashlib
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Anonymizer:
    """HIPAA Safe Harbor de-identification for tabular health data."""

    def __init__(self, salt: str = "hi-ai-2024-safe"):
        """
        Args:
            salt: Secret salt for one-way hashing of identifiers.
                  In production, load from environment variable / vault.
        """
        self.salt = salt

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def anonymize_members(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply HIPAA Safe Harbor de-identification to member DataFrame.

        Transformations:
          - Names         → removed
          - DOB           → age band (decade)
          - Zip code      → 3-digit prefix only (if pop > 20k; else "000")
          - member_id     → one-way hash token
        """
        df = df.copy()
        log.info("Applying HIPAA Safe Harbor anonymization to members...")

        # 1. Remove direct identifiers (names)
        df = self._drop_if_exists(
            df, ["first_name", "last_name", "email", "phone", "ssn", "address"]
        )

        # 2. Hash member_id → pseudonymous token (preserves join-ability)
        if "member_id" in df.columns:
            df["member_id"] = df["member_id"].apply(self._hash_id)

        # 3. Generalize DOB → age decade band
        if "dob" in df.columns:
            df["age_band"] = df["dob"].apply(self._age_to_decade)
            df = df.drop(columns=["dob"])

        # 4. Generalize zip → 3-digit prefix
        if "zip_code" in df.columns:
            df["zip3"] = df["zip_code"].apply(self._zip_to_3digit)
            df = df.drop(columns=["zip_code"])

        # 5. Keep only state-level geography (not finer)
        # state column is retained — state level is safe under Safe Harbor

        log.info("Member anonymization complete.")
        return df

    def anonymize_claims(self, df: pd.DataFrame) -> pd.DataFrame:
        """Light de-identification for claims data (no direct PHI in standard claims)."""
        df = df.copy()

        # Hash IDs for referential integrity without exposure
        for col in ["claim_id", "member_id", "provider_npi"]:
            if col in df.columns:
                df[col] = df[col].apply(self._hash_id)

        # Generalize dates to year-month only for public reporting
        for col in ["claim_date", "service_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.to_period("M").astype(str)

        return df

    def tokenize_id(self, identifier: str) -> str:
        """One-way hash a single identifier. Safe to use in pipelines."""
        return self._hash_id(identifier)

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _hash_id(self, value: str) -> str:
        """SHA-256 HMAC-style one-way hash with salt."""
        salted = f"{self.salt}:{value}".encode("utf-8")
        return "TOK_" + hashlib.sha256(salted).hexdigest()[:16].upper()

    @staticmethod
    def _age_to_decade(dob) -> str:
        """Convert DOB to decade age band (e.g. '40s')."""
        try:
            from datetime import datetime

            age = (datetime.now().date() - pd.to_datetime(dob).date()).days // 365
            decade = (age // 10) * 10
            return f"{decade}s"
        except Exception:
            return "Unknown"

    @staticmethod
    def _zip_to_3digit(zip_code: str) -> str:
        """
        HIPAA Safe Harbor: retain only 3-digit ZIP prefix.
        For low-population ZIP3s (population < 20k), return '000'.
        Note: In production, cross-reference actual ZIP3 population data.
        """
        try:
            z = str(zip_code).zfill(5)[:3]
            # Known sparse ZIP3 prefixes (simplified — use full table in production)
            sparse_zip3 = {
                "036",
                "059",
                "063",
                "102",
                "203",
                "556",
                "692",
                "790",
                "821",
                "823",
            }
            return "000" if z in sparse_zip3 else z
        except Exception:
            return "000"

    @staticmethod
    def _drop_if_exists(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        existing = [c for c in cols if c in df.columns]
        if existing:
            log.info(f"Dropping PHI columns: {existing}")
            df = df.drop(columns=existing)
        return df
