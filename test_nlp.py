import sys
sys.path.insert(0, '.')
from src.models.nlp_model import ClinicalNLPPipeline, CopyPasteDetector

print("NLP PIPELINE TEST")
print("-" * 40)

nlp = ClinicalNLPPipeline()

# Test 1 - Clean normal note
print("\nTest 1 - Clean Normal Note:")
note1 = "Patient presented with chest pain. ECG performed showing normal sinus rhythm. Blood pressure 130/85. Prescribed aspirin and scheduled follow up in 2 weeks."
result = nlp.analyze_note(
    note=note1,
    claim_id="CLM_001",
    cpt_codes=["99213", "93000"],
    n_procedures=2,
    billed_amount=450.0
)
print("NLP Risk Score:", result['nlp_risk_score'])
print("NLP Risk Band:", result['nlp_risk_band'])
print("Flags:", result['anomaly']['flags'])

# Test 2 - Suspicious note
print("\nTest 2 - Suspicious Note:")
note2 = "As previously documented. See prior note."
result = nlp.analyze_note(
    note=note2,
    claim_id="CLM_002",
    cpt_codes=["27447", "99285"],
    n_procedures=8,
    billed_amount=25000.0
)
print("NLP Risk Score:", result['nlp_risk_score'])
print("NLP Risk Band:", result['nlp_risk_band'])
print("Flags:", result['anomaly']['flags'])

# Test 3 - Copy paste detection
print("\nTest 3 - Copy Paste Detection:")
detector = CopyPasteDetector()
note_a = "Patient seen for chest pain. EKG performed. Normal sinus rhythm."
note_b = "Patient seen for chest pain. EKG performed. Normal sinus rhythm."
result = detector.check_similarity(note_a, note_b)
print("Similarity:", result['similarity'])
print("Flag:", result['flag'])
print("Risk:", result['risk'])

# Test 4 - Negative appeal
print("\nTest 4 - Negative Appeal:")
appeal1 = "This denial is wrong and unfair. I am contacting my attorney and the state insurance commissioner."
result = nlp.analyze_appeal(appeal1)
print("Sentiment:", result['sentiment'])
print("Escalation Risk:", result['escalation_risk'])
print("Priority:", result['priority'])

# Test 5 - Positive appeal
print("\nTest 5 - Positive Appeal:")
appeal2 = "Thank you for resolving my claim. I appreciate the helpful service from your team."
result = nlp.analyze_appeal(appeal2)
print("Sentiment:", result['sentiment'])
print("Escalation Risk:", result['escalation_risk'])
print("Priority:", result['priority'])

print("\nNLP Pipeline Done!")