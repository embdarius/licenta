"""Shared text preprocessing utilities used by triage v1, v2, and downstream
training/inference pipelines.

Centralizing this here means triage v1 and v2 are peers — neither has to import
from the other just to get complaint normalization.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Medical abbreviation expansion table for chief-complaint text.
# Used by TF-IDF vectorization in triage models.
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "abd": "abdominal",
    "n/v": "nausea vomiting",
    "n v": "nausea vomiting",
    "s/p": "status post",
    "s p": "status post",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "ha": "headache",
    "ams": "altered mental status",
    "loc": "loss of consciousness",
    "etoh": "alcohol intoxication",
    "uti": "urinary tract infection",
    "uri": "upper respiratory infection",
    "mv": "motor vehicle",
    "mva": "motor vehicle accident",
    "mvc": "motor vehicle collision",
    "htn": "hypertension",
    "dm": "diabetes",
    "chf": "congestive heart failure",
    "gi": "gastrointestinal",
    "r/o": "rule out",
    "w/": "with",
    "w/o": "without",
    "fx": "fracture",
    "lac": "laceration",
    "inj": "injury",
    "sx": "symptoms",
    "dx": "diagnosis",
    "tx": "treatment",
    "hx": "history",
    "bld": "blood",
    "diff": "difficulty",
    "eval": "evaluation",
    "sz": "seizure",
    "ped": "pediatric",
    "psych": "psychiatric",
    "resp": "respiratory",
    "bilat": "bilateral",
    "lt": "left",
    "rt": "right",
    "pos": "positive",
    "neg": "negative",
    "hiv": "hiv",
    "copd": "chronic obstructive pulmonary disease",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident stroke",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "ble": "bleeding",
}


def normalize_complaint_text(text: str) -> str:
    """Normalize chief complaint text for TF-IDF.

    Lowercases, strips punctuation, expands medical abbreviations, drops
    single-character tokens.
    """
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text).lower().strip()
    text = text.replace(",", " ").replace(";", " ").replace("/", " ").replace("-", " ")
    text = text.replace("(", " ").replace(")", " ").replace(".", " ")

    words = text.split()
    expanded = []
    for word in words:
        word = word.strip()
        if word in ABBREVIATIONS:
            expanded.append(ABBREVIATIONS[word])
        elif len(word) > 1:
            expanded.append(word)

    return " ".join(expanded)
