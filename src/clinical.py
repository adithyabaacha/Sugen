"""Small, deterministic note normalizer. Does not use gold labels or external services."""
from __future__ import annotations
import re

# Generic expansions, not mappings from consultation IDs to gold answers.
EXPANSIONS = {
    "urti": "upper respiratory tract infection",
    "uti": "urinary tract infection",
    "copd": "chronic obstructive pulmonary disease",
    "tb": "tuberculosis",
    "t2dm": "type 2 diabetes mellitus",
    "dm": "diabetes mellitus",
    "gord": "gastro oesophageal reflux disease",
    "gerd": "gastro oesophageal reflux disease",
    "pid": "pelvic inflammatory disease",
    "dvt": "deep vein thrombosis",
    "pet": "pre eclampsia",
    "sam": "severe acute malnutrition",
    "mps": "malaria parasites",
    "niddm": "type 2 diabetes mellitus",
    "htn": "hypertension",
    "af": "atrial fibrillation",
    "bph": "benign prostatic hyperplasia",
    "uri": "upper respiratory infection",
    "rti": "respiratory tract infection",
}
# Common clinical synonyms; transparently curated; no code-ID mappings.
SYNONYMS = {
    "jiggers": "tungiasis sand flea infestation",
    "bilharzia": "schistosomiasis",
    "chickenpox": "varicella",
    "vaginal thrush": "vulvovaginal candidiasis candidosis",
    "thrush": "candidiasis candidosis",
    "enteric fever": "typhoid fever",
    "amoebic dysentery": "acute amoebiasis intestinal entamoeba",
    "pink eye": "conjunctivitis",
    "heartburn": "gastro oesophageal reflux",
    "p falciparum": "plasmodium falciparum",
    "malaria due to plasmodium falciparum": "falciparum malaria",
    "pulmonary tb": "respiratory tuberculosis",
    "gastroenteritis": "intestinal infection diarrhoea",
    "eczema": "dermatitis",
    "hives": "urticaria",
    "sinus infection": "sinusitis",
    "hypoglycaemia": "low blood glucose",
    "hypoglycemia": "low blood glucose",
}

DIAG_LINE = re.compile(r"(?im)^\s*(?:a|dx|diagnosis|imp|impression|assessment|working diagnosis|final diagnosis)\s*[:\-]\s*(.+)$")
DIAG_INLINE = re.compile(r"(?i)\b(?:dx|diagnosis|impression|assessment)\s*[:\-]\s*([^.;\n]{3,140})")
SPOKEN_DIAG = re.compile(r"(?i)\b(?:this is|it is|it's|you have|has|most likely|we call it|diagnosis is|i think this is)\s+([^.!?\n]{3,100})")


def clean_text(text: str) -> str:
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"\bp\s*\.\s*falciparum\b", "plasmodium falciparum", text)
    text = re.sub(r"\bx\s*\d+\s*/\s*\d+\b", " ", text)
    for key, value in EXPANSIONS.items():
        text = re.sub(r"\b" + re.escape(key) + r"\b", value, text)
    # Keep original expressions and supplement them with search terms.
    additions = []
    for phrase, expansion in SYNONYMS.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            additions.append(expansion)
    return re.sub(r"\s+", " ", text + " " + " ".join(additions)).strip()


def get_diagnosis_focus(text: str) -> str:
    """Extract likely assessment spans. Never reads style or gold fields."""
    parts = [m.group(1).strip() for m in DIAG_LINE.finditer(text)]
    if not parts:
        parts = [m.group(1).strip() for m in DIAG_INLINE.finditer(text)]
    if not parts:
        # Ambient transcripts often state the assessment near the end.
        spoken = [m.group(1).strip() for m in SPOKEN_DIAG.finditer(text)]
        parts = spoken[-3:]
    if parts:
        # Limit the impact of extra historical diagnoses and long plan text.
        return clean_text(" ".join(parts)[:400])
    return ""


def search_queries(text: str) -> tuple[str, str]:
    return clean_text(text[:6000]), get_diagnosis_focus(text)
