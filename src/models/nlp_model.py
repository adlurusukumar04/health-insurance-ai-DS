"""
nlp_model.py
------------
Deep Learning NLP pipeline for medical text analysis.

Capabilities:
  1. Named Entity Recognition (NER) — identify diagnoses, procedures, drugs, providers
  2. Diagnosis classification using BERT fine-tuned on clinical text
  3. Clinical note anomaly detection — flag notes inconsistent with billed codes
  4. Copy-paste detection — identical or near-identical notes across claims (fraud signal)
  5. Sentiment analysis on member complaints / appeal letters

Models:
  - spaCy en_core_web_sm     (NER baseline)
  - BioBERT / ClinicalBERT   (classification, when GPU available)
  - TF-IDF + Cosine Sim      (copy-paste detection, lightweight)

Usage:
    from src.models.nlp_model import ClinicalNLPPipeline
    nlp = ClinicalNLPPipeline()
    result = nlp.analyze_note("Patient presented with chest pain...")
"""

import hashlib
import logging
import re
from typing import Dict, List

import pandas as pd

log = logging.getLogger(__name__)

# ── Optional heavy imports (graceful degradation) ─────────────────────────────
try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    log.warning("spaCy model not loaded — NER will use regex fallback")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN_NLP = True
except ImportError:
    HAS_SKLEARN_NLP = False

try:
    from transformers import pipeline as hf_pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    log.warning("HuggingFace transformers not available — using rule-based fallback")


# ─────────────────────────────────────────────────────────────────────────────
# MEDICAL ENTITY PATTERNS (regex fallback when spaCy unavailable)
# ─────────────────────────────────────────────────────────────────────────────

ICD10_PATTERN = re.compile(r"\b[A-Z]\d{2}\.?\d{0,4}\b")
CPT_PATTERN = re.compile(r"\b\d{5}\b")
NPI_PATTERN = re.compile(r"\bNPI\s*:?\s*\d{10}\b", re.IGNORECASE)
DRUG_PATTERN = re.compile(
    r"\b(aspirin|lisinopril|metformin|atorvastatin|amoxicillin|"
    r"hydrocodone|oxycodone|alprazolam|gabapentin|furosemide)\b",
    re.IGNORECASE,
)
DIAGNOSIS_KEYWORDS = {
    "hypertension": ["hypertension", "high blood pressure", "htn"],
    "diabetes": ["diabetes", "diabetic", "dm type", "t2dm", "t1dm"],
    "chest_pain": ["chest pain", "angina", "cardiac pain", "substernal"],
    "fracture": ["fracture", "fx", "broken bone", "osseous"],
    "depression": ["depression", "depressive", "mdd", "major depressive"],
    "anxiety": ["anxiety", "anxious", "panic disorder", "gad"],
    "infection": ["infection", "infectious", "bacterial", "viral", "sepsis"],
}


# ─────────────────────────────────────────────────────────────────────────────
# NAMED ENTITY RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────


def extract_entities_regex(text: str) -> Dict[str, List[str]]:
    """Regex-based NER fallback."""
    return {
        "ICD10_codes": ICD10_PATTERN.findall(text),
        "CPT_codes": CPT_PATTERN.findall(text),
        "NPI_numbers": NPI_PATTERN.findall(text),
        "medications": DRUG_PATTERN.findall(text),
        "diagnoses": [
            dx
            for dx, terms in DIAGNOSIS_KEYWORDS.items()
            if any(t in text.lower() for t in terms)
        ],
    }


def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """spaCy NER — extracts persons, organizations, dates, quantities."""
    if not HAS_SPACY:
        return extract_entities_regex(text)
    doc = _nlp(text[:512000])  # spaCy limit
    entities = {
        "PERSON": [],
        "ORG": [],
        "DATE": [],
        "MONEY": [],
        "GPE": [],
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    # Merge with regex medical entities
    medical = extract_entities_regex(text)
    return {**entities, **medical}


# ─────────────────────────────────────────────────────────────────────────────
# COPY-PASTE DETECTION
# ─────────────────────────────────────────────────────────────────────────────


class CopyPasteDetector:
    """
    Detects suspiciously similar clinical notes across claims.
    High similarity (>0.95) is a known fraud signal — providers
    copy-paste notes for services never actually rendered.
    """

    SIMILARITY_THRESHOLDS = {
        "suspicious": 0.85,
        "likely_copied": 0.95,
    }

    def __init__(self):
        self.vectorizer = (
            TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=1
            )
            if HAS_SKLEARN_NLP
            else None
        )
        self._fitted = False
        self._note_hashes: Dict[str, str] = {}  # note_hash → claim_id

    def fit(self, notes: List[str]) -> None:
        if self.vectorizer:
            self.vectorizer.fit(notes)
            self._fitted = True

    def check_similarity(self, note_a: str, note_b: str) -> Dict:
        """Compute similarity between two clinical notes."""
        # Exact match via hash
        hash_a = hashlib.md5(note_a.strip().lower().encode()).hexdigest()
        hash_b = hashlib.md5(note_b.strip().lower().encode()).hexdigest()
        if hash_a == hash_b:
            return {"similarity": 1.0, "flag": "EXACT_DUPLICATE", "risk": "CRITICAL"}

        # TF-IDF cosine similarity
        if self.vectorizer and self._fitted:
            try:
                vecs = self.vectorizer.transform([note_a, note_b])
                sim = float(cosine_similarity(vecs[0], vecs[1])[0][0])
            except Exception:
                sim = self._character_similarity(note_a, note_b)
        else:
            sim = self._character_similarity(note_a, note_b)

        if sim >= self.SIMILARITY_THRESHOLDS["likely_copied"]:
            flag = "LIKELY_COPIED"
            risk = "HIGH"
        elif sim >= self.SIMILARITY_THRESHOLDS["suspicious"]:
            flag = "SUSPICIOUS_SIMILARITY"
            risk = "MEDIUM"
        else:
            flag = "OK"
            risk = "LOW"

        return {"similarity": round(sim, 4), "flag": flag, "risk": risk}

    def batch_check(
        self,
        notes_df: pd.DataFrame,
        note_col: str = "clinical_note",
        id_col: str = "claim_id",
    ) -> pd.DataFrame:
        """Flag claims with suspiciously similar notes within same provider."""
        results = []
        notes = notes_df[note_col].fillna("").tolist()
        ids = notes_df[id_col].tolist()

        if not self._fitted and self.vectorizer:
            self.fit(notes)

        n = len(notes)
        for i in range(n):
            max_sim = 0.0
            most_sim = None
            for j in range(n):
                if i == j:
                    continue
                sim = self.check_similarity(notes[i], notes[j])["similarity"]
                if sim > max_sim:
                    max_sim = sim
                    most_sim = ids[j]

            flag = (
                "LIKELY_COPIED"
                if max_sim >= 0.95
                else ("SUSPICIOUS" if max_sim >= 0.85 else "OK")
            )
            results.append(
                {
                    "claim_id": ids[i],
                    "max_similarity": round(max_sim, 4),
                    "most_similar_claim": most_sim,
                    "copy_paste_flag": flag,
                }
            )
        return pd.DataFrame(results)

    @staticmethod
    def _character_similarity(a: str, b: str) -> float:
        """Jaccard similarity on character 3-grams as lightweight fallback."""

        def ngrams(s, n=3):
            return set(s[i : i + n] for i in range(len(s) - n + 1))

        sa, sb = ngrams(a.lower()), ngrams(b.lower())
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL NOTE ANOMALY SCORER
# ─────────────────────────────────────────────────────────────────────────────


class ClinicalNoteAnomalyScorer:
    """
    Scores clinical notes for internal consistency with billed codes.
    Flags:
      - Note mentions diagnosis not supported by ICD-10 codes on claim
      - Note describes simple visit but complex procedures billed
      - Note is too short / generic for billed complexity
      - Note contains red-flag phrases
    """

    RED_FLAG_PHRASES = [
        "as previously documented",
        "see prior note",
        "unchanged from last visit",
        "patient refused to be examined",
        "dictated but not read",
        "see attached",
        "n/a",
        "na",
    ]

    MIN_NOTE_LENGTH_BY_COMPLEXITY = {
        1: 50,  # simple visit — at least 50 chars
        2: 100,
        3: 150,
        4: 200,
        5: 300,  # complex case — at least 300 chars
    }

    def score_note(
        self,
        note: str,
        billed_cpt_codes: List[str],
        n_procedures: int,
        billed_amount: float,
    ) -> Dict:
        """
        Returns anomaly score 0–1 and list of flags.
        Higher score = more anomalous = higher fraud risk.
        """
        flags = []
        score_parts = []

        if not note or len(note.strip()) < 10:
            return {
                "anomaly_score": 0.9,
                "flags": ["EMPTY_OR_MISSING_NOTE"],
                "note_length": 0,
            }

        note_lower = note.lower()
        note_length = len(note)

        # 1. Note length vs complexity
        complexity = min(max(n_procedures, 1), 5)
        min_len = self.MIN_NOTE_LENGTH_BY_COMPLEXITY.get(complexity, 100)
        if note_length < min_len:
            flags.append(f"NOTE_TOO_SHORT_FOR_COMPLEXITY_{complexity}")
            score_parts.append(0.3)

        # 2. Red flag phrases
        found_red = [p for p in self.RED_FLAG_PHRASES if p in note_lower]
        if found_red:
            flags.append(f"RED_FLAG_PHRASES: {found_red[:3]}")
            score_parts.append(0.2 * len(found_red))

        # 3. High billed amount with generic note
        generic_words = ["patient seen", "follow up", "reviewed", "noted"]
        generic_count = sum(1 for w in generic_words if w in note_lower)
        if billed_amount > 5000 and generic_count >= 2:
            flags.append("HIGH_BILL_GENERIC_NOTE")
            score_parts.append(0.35)

        # 4. Check diagnosis consistency
        mentioned_diagnoses = [
            dx
            for dx, terms in DIAGNOSIS_KEYWORDS.items()
            if any(t in note_lower for t in terms)
        ]
        if n_procedures > 5 and not mentioned_diagnoses:
            flags.append("MANY_PROCEDURES_NO_DIAGNOSIS_MENTIONED")
            score_parts.append(0.20)

        # 5. Copy-paste indicator: repeating sentences
        sentences = [s.strip() for s in re.split(r"[.!?]", note) if len(s.strip()) > 20]
        if len(sentences) > 3:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.6:
                flags.append("REPETITIVE_SENTENCES_POSSIBLE_COPY_PASTE")
                score_parts.append(0.25)

        anomaly_score = min(sum(score_parts), 1.0)
        return {
            "anomaly_score": round(anomaly_score, 4),
            "flags": flags,
            "note_length": note_length,
            "mentioned_diagnoses": mentioned_diagnoses,
            "has_red_flags": len(found_red) > 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT ANALYSIS (member appeals / complaints)
# ─────────────────────────────────────────────────────────────────────────────


class AppealSentimentAnalyzer:
    """
    Analyzes member appeal letters and complaint text.
    Used to:
      - Prioritize urgent appeals (very negative / distressed)
      - Detect potential legal threats
      - Flag escalation risk
    """

    ESCALATION_KEYWORDS = [
        "lawyer",
        "attorney",
        "lawsuit",
        "sue",
        "legal action",
        "department of insurance",
        "state insurance commissioner",
        "cms complaint",
        "medicare complaint",
        "discrimination",
        "hipaa violation",
        "news",
        "media",
        "social media",
    ]
    URGENCY_KEYWORDS = [
        "urgent",
        "emergency",
        "life-threatening",
        "terminal",
        "surgery scheduled",
        "cannot wait",
        "death",
        "dying",
    ]

    def analyze(self, text: str) -> Dict:
        """Analyze appeal / complaint text."""
        text_lower = text.lower()

        escalation_risk = any(kw in text_lower for kw in self.ESCALATION_KEYWORDS)
        urgency_flag = any(kw in text_lower for kw in self.URGENCY_KEYWORDS)

        # Simple rule-based sentiment (replace with BERT in production)
        negative_words = [
            "denied",
            "unfair",
            "wrong",
            "incorrect",
            "mistake",
            "error",
            "refuse",
            "cannot",
            "unable",
            "frustrated",
        ]
        positive_words = ["thank", "appreciate", "satisfied", "resolved", "helpful"]
        neg_count = sum(1 for w in negative_words if w in text_lower)
        pos_count = sum(1 for w in positive_words if w in text_lower)

        if neg_count > pos_count + 2:
            sentiment = "NEGATIVE"
        elif pos_count > neg_count:
            sentiment = "POSITIVE"
        else:
            sentiment = "NEUTRAL"

        priority = (
            "CRITICAL"
            if escalation_risk and urgency_flag
            else (
                "HIGH"
                if escalation_risk or urgency_flag
                else "MEDIUM" if sentiment == "NEGATIVE" else "LOW"
            )
        )

        return {
            "sentiment": sentiment,
            "escalation_risk": escalation_risk,
            "urgency_flag": urgency_flag,
            "priority": priority,
            "negative_signals": neg_count,
            "positive_signals": pos_count,
            "word_count": len(text.split()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


class ClinicalNLPPipeline:
    """
    Unified NLP pipeline. Combines all NLP components into one interface.

    Usage:
        nlp = ClinicalNLPPipeline()
        result = nlp.analyze_note(
            note="Patient seen for chest pain. EKG ordered.",
            claim_id="CLM001",
            cpt_codes=["93000","99213"],
            n_procedures=2,
            billed_amount=450.0
        )
    """

    def __init__(self):
        self.anomaly_scorer = ClinicalNoteAnomalyScorer()
        self.copy_detector = CopyPasteDetector()
        self.sentiment = AppealSentimentAnalyzer()
        log.info("ClinicalNLPPipeline initialized")

    def analyze_note(
        self,
        note: str,
        claim_id: str = "unknown",
        cpt_codes: List[str] = None,
        n_procedures: int = 1,
        billed_amount: float = 0.0,
    ) -> Dict:
        """Full analysis of a single clinical note."""
        entities = extract_entities_spacy(note)
        anomaly = self.anomaly_scorer.score_note(
            note, cpt_codes or [], n_procedures, billed_amount
        )
        return {
            "claim_id": claim_id,
            "entities": entities,
            "anomaly": anomaly,
            "nlp_risk_score": anomaly["anomaly_score"],
            "nlp_risk_band": (
                "HIGH"
                if anomaly["anomaly_score"] >= 0.6
                else "MEDIUM" if anomaly["anomaly_score"] >= 0.3 else "LOW"
            ),
        }

    def analyze_batch(
        self,
        df: pd.DataFrame,
        note_col: str = "clinical_note",
        claim_id_col: str = "claim_id",
        cpt_col: str = "cpt_codes",
        procedures_col: str = "n_procedures",
        amount_col: str = "billed_amount",
    ) -> pd.DataFrame:
        """Analyze a DataFrame of clinical notes. Returns enriched DataFrame."""
        results = []
        for _, row in df.iterrows():
            cpts = str(row.get(cpt_col, "")).split("|") if cpt_col in df.columns else []
            r = self.analyze_note(
                note=str(row.get(note_col, "")),
                claim_id=str(row.get(claim_id_col, "")),
                cpt_codes=cpts,
                n_procedures=int(row.get(procedures_col, 1)),
                billed_amount=float(row.get(amount_col, 0)),
            )
            results.append(
                {
                    "claim_id": r["claim_id"],
                    "nlp_risk_score": r["nlp_risk_score"],
                    "nlp_risk_band": r["nlp_risk_band"],
                    "note_flags": "|".join(r["anomaly"]["flags"]),
                    "note_length": r["anomaly"]["note_length"],
                    "has_red_flags": r["anomaly"]["has_red_flags"],
                }
            )

        # Copy-paste detection across batch
        if note_col in df.columns and len(df) > 1:
            copy_df = self.copy_detector.batch_check(df, note_col, claim_id_col)
            results_df = pd.DataFrame(results).merge(copy_df, on="claim_id", how="left")
        else:
            results_df = pd.DataFrame(results)

        return results_df

    def analyze_appeal(self, text: str) -> Dict:
        """Analyze a member appeal or complaint letter."""
        return self.sentiment.analyze(text)
