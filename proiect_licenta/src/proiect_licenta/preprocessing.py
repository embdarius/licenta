"""Shared chief-complaint text normalization, so the triage and doctor tiers
stay peers instead of importing normalization from one another.
"""

from __future__ import annotations

import re

import pandas as pd


# Medical abbreviation expansion table for chief-complaint text.
# Used by TF-IDF vectorization in triage models.
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


# Lay -> clinical chief-complaint synonym map. Applied only to free-text parser
# output (via clinicalize_complaint), never to the tabular chief_complaint and
# never at training time, so the shared TF-IDF vectorizer and every trained model
# stay unchanged (no retraining) and the benchmark's tabular reference modes stay
# uncontaminated. Folding this into normalize_complaint_text would change the
# trained vocabulary and require retraining, since triage staff already chart
# many semi-lay terms. Targets are high-frequency clinical terms from the
# MIMIC-IV-ED chiefcomplaint vocabulary; keys are general lay paraphrases.
LAY_TO_CLINICAL = {
    # abdominal pain
    "stomach pain": "abdominal pain",
    "stomach ache": "abdominal pain",
    "stomachache": "abdominal pain",
    "stomach hurts": "abdominal pain",
    "stomach hurting": "abdominal pain",
    "stomach cramps": "abdominal pain",
    "belly pain": "abdominal pain",
    "belly ache": "abdominal pain",
    "bellyache": "abdominal pain",
    "belly hurts": "abdominal pain",
    "tummy pain": "abdominal pain",
    "tummy ache": "abdominal pain",
    "tummy hurts": "abdominal pain",
    "gut pain": "abdominal pain",
    # dyspnea / shortness of breath
    "trouble breathing": "dyspnea",
    "having trouble breathing": "dyspnea",
    "hard to breathe": "dyspnea",
    "having a hard time breathing": "dyspnea",
    "difficulty breathing": "dyspnea",
    "can't breathe": "dyspnea",
    "cant breathe": "dyspnea",
    "cannot breathe": "dyspnea",
    "can't catch my breath": "dyspnea",
    "cant catch my breath": "dyspnea",
    "out of breath": "dyspnea",
    "short of breath": "dyspnea",
    "shortness of breath": "dyspnea",
    "breathing problems": "dyspnea",
    "breathing problem": "dyspnea",
    # nausea / vomiting (lay forms only; "nausea"/"vomiting" stay as-is)
    "throwing up": "nausea vomiting",
    "throw up": "nausea vomiting",
    "threw up": "nausea vomiting",
    "puking": "nausea vomiting",
    "queasy": "nausea vomiting",
    "sick to my stomach": "nausea vomiting",
    "feeling sick to my stomach": "nausea vomiting",
    "nauseous": "nausea vomiting",
    # syncope
    "passed out": "syncope",
    "pass out": "syncope",
    "fainted": "syncope",
    "fainting": "syncope",
    "blacked out": "syncope",
    "black out": "syncope",
    "lost consciousness": "syncope",
    "loss of consciousness": "syncope",
    # dizziness
    "dizzy": "dizziness",
    "feeling dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",
    "woozy": "dizziness",
    "room spinning": "dizziness",
    "the room is spinning": "dizziness",
    "off balance": "dizziness",
    # fall
    "i fell": "fall",
    "fell down": "fall",
    "had a fall": "fall",
    "slipped and fell": "fall",
    "tripped and fell": "fall",
    "took a fall": "fall",
    "fell over": "fall",
    # head injury
    "hit my head": "head injury",
    "banged my head": "head injury",
    "bumped my head": "head injury",
    "knocked my head": "head injury",
    # intracranial hemorrhage
    "brain bleed": "intracranial hemorrhage",
    "bleeding in my brain": "intracranial hemorrhage",
    "bleed in my brain": "intracranial hemorrhage",
    "bleeding on my brain": "intracranial hemorrhage",
    # rectal bleeding (BRBPR)
    "blood in my stool": "brbpr rectal bleeding",
    "blood in stool": "brbpr rectal bleeding",
    "bloody stool": "brbpr rectal bleeding",
    "blood in my poop": "brbpr rectal bleeding",
    "blood when i poop": "brbpr rectal bleeding",
    "bright red blood": "brbpr rectal bleeding",
    "blood in my bowel movement": "brbpr rectal bleeding",
    # chest pain
    "chest hurts": "chest pain",
    "pain in my chest": "chest pain",
    "chest tightness": "chest pain",
    "chest pressure": "chest pain",
    "tightness in my chest": "chest pain",
    "pressure in my chest": "chest pain",
    "chest is tight": "chest pain",
    # palpitations
    "heart racing": "palpitations",
    "racing heart": "palpitations",
    "my heart is racing": "palpitations",
    "heart is racing": "palpitations",
    "heart pounding": "palpitations",
    "heart fluttering": "palpitations",
    "fluttering in my chest": "palpitations",
    "heart skipping": "palpitations",
    # fever
    "feverish": "fever",
    "burning up": "fever",
    "high temperature": "fever",
    "running a temperature": "fever",
    "running a fever": "fever",
    # weakness
    "weak": "weakness",
    "feel weak": "weakness",
    "feeling weak": "weakness",
    "no strength": "weakness",
    "weak all over": "weakness",
    # suicidal ideation (SI)
    "suicidal": "suicidal ideation si",
    "suicidal thoughts": "suicidal ideation si",
    "thoughts of suicide": "suicidal ideation si",
    "thinking of suicide": "suicidal ideation si",
    "want to kill myself": "suicidal ideation si",
    "kill myself": "suicidal ideation si",
    "want to hurt myself": "suicidal ideation si",
    "want to end my life": "suicidal ideation si",
    # cough
    "coughing": "cough",
    "bad cough": "cough",
    "can't stop coughing": "cough",
    # sore throat
    "throat hurts": "sore throat",
    "throat pain": "sore throat",
    "scratchy throat": "sore throat",
    # allergic reaction
    "having an allergic reaction": "allergic reaction",
    "allergic": "allergic reaction",
    "breaking out in hives": "allergic reaction",
    "hives": "allergic reaction",
    # dysuria
    "burning when i pee": "dysuria",
    "burning when i urinate": "dysuria",
    "hurts to pee": "dysuria",
    "hurts when i pee": "dysuria",
    "pain when i pee": "dysuria",
    "painful urination": "dysuria",
    "burning urination": "dysuria",
    "burning pee": "dysuria",
    # hematuria
    "blood in my urine": "hematuria",
    "blood in urine": "hematuria",
    "blood in my pee": "hematuria",
    "bloody urine": "hematuria",
    "blood when i pee": "hematuria",
    # urinary retention
    "can't pee": "urinary retention",
    "cant pee": "urinary retention",
    "can't urinate": "urinary retention",
    "cant urinate": "urinary retention",
    "unable to pee": "urinary retention",
    "unable to urinate": "urinary retention",
    "can't empty my bladder": "urinary retention",
    # epistaxis
    "nosebleed": "epistaxis",
    "nose bleed": "epistaxis",
    "bloody nose": "epistaxis",
    "nose bleeding": "epistaxis",
    "bleeding from my nose": "epistaxis",
    # diarrhea
    "loose stool": "diarrhea",
    "loose stools": "diarrhea",
    "watery stool": "diarrhea",
    "watery stools": "diarrhea",
    "the runs": "diarrhea",
    "runny stool": "diarrhea",
    # vaginal bleeding
    "bleeding down there": "vaginal bleeding",
    # hyperglycemia
    "high blood sugar": "hyperglycemia",
    "blood sugar is high": "hyperglycemia",
    # hypertension
    "high blood pressure": "hypertension",
    # alcohol intoxication
    "drunk": "alcohol intoxication",
    "intoxicated": "alcohol intoxication",
    "been drinking": "alcohol intoxication",
    "too much to drink": "alcohol intoxication",
    "had too much to drink": "alcohol intoxication",
    "drank too much": "alcohol intoxication",
    # seizure (lay forms only)
    "convulsion": "seizure",
    "convulsions": "seizure",
    "had a fit": "seizure",
    "shaking uncontrollably": "seizure",
    # altered mental status
    "confused": "altered mental status",
    "acting confused": "altered mental status",
    "disoriented": "altered mental status",
    "not making sense": "altered mental status",
    "not acting right": "altered mental status",
    # rash
    "skin rash": "rash",
    "itchy rash": "rash",
    "breaking out in a rash": "rash",
    # back pain
    "back hurts": "back pain",
    "backache": "back pain",
    "back ache": "back pain",
    "pain in my back": "back pain",
    "lower back hurts": "back pain",
    # headache
    "head hurts": "headache",
    "head is pounding": "headache",
    "migraine": "headache",
    "bad headache": "headache",
    "pounding headache": "headache",
    "splitting headache": "headache",
}


# Compiled once: longest phrases first so multi-word keys win over any prefix,
# matched on word boundaries and case-insensitively. Curly apostrophes are
# normalized to straight before matching (see `clinicalize_complaint`).
_LAY_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(LAY_TO_CLINICAL, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def clinicalize_complaint(text: str) -> str:
    """Rewrite lay chief-complaint phrases into the clinical terms the models
    were trained on (see `LAY_TO_CLINICAL`).

    Intended for free-text *parser output* only - do NOT apply to the tabular
    `chiefcomplaint` or at training time (that would change the trained TF-IDF
    vocabulary). Non-matching text is left untouched; downstream callers still
    run the result through `normalize_complaint_text` for TF-IDF.
    """
    if text is None:
        return ""
    s = str(text)
    if not s.strip():
        return s
    s = s.replace("'", "'")  # curly -> straight apostrophe for matching
    return _LAY_PATTERN.sub(
        lambda m: LAY_TO_CLINICAL[re.sub(r"\s+", " ", m.group(0).lower())], s
    )


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
