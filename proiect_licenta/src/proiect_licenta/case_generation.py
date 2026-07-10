"""Case Generation Agent (offline/benchmark-only).

Translates real MIMIC-IV tabular ED rows into realistic natural-language patient
descriptions (the free text the NLP Parser sees) plus every other input the crew
collects (vitals, rhythm, medications, PMH). Cases are generated strictly from
the tabular row: the LLM only rephrases the opening narrative, everything else is
deterministic from the row, and a grounding validator enforces that invention is
confined to (and caught in) the narrative. prior_history is reverse-mapped from
the loader's real pmh_* flags to canonical phrases that round-trip through
pmh_vocab.flags_from_text back to the same category. Generated cases drive
benchmark_pipeline_e2e.py and are persisted under data/derived/synthetic_cases/
(gitignored under the MIMIC DUA). generate_and_save_cases runs the disposition
and nurse_v3 loaders once to sample from the held-out test splits.
"""

import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from proiect_licenta.paths import (
    DERIVED_DIR, MEDRECON_CSV, VITALSIGN_CSV,
    DOCTOR_V3_BASE_DIR, DOCTOR_V3_DIR,
)
from proiect_licenta.preprocessing import normalize_complaint_text
from proiect_licenta.pmh_vocab import (
    PMH_CATEGORIES, PMH_KEYWORD_MAP, flags_from_text as pmh_flags_from_text,
)
# Switchable LLM backend - None for the default flash backend (Gemini, unchanged).
from proiect_licenta.llm_config import get_llm

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths / artifacts
SYNTH_DIR = DERIVED_DIR / "synthetic_cases"
CASES_JSON = SYNTH_DIR / "cases.json"
FEATURES_PKL = SYNTH_DIR / "sampled_features.pkl"

VITAL_COLS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

# Default cohort sizes (stratified admit/discharge).
DEFAULT_N_ADMITTED = 13
DEFAULT_N_DISCHARGED = 7
DEFAULT_SEED = 20260529


# PMH category -> canonical patient-speak phrase (deterministic reverse map)
# Curated preferred phrasing per diagnosis group. Every phrase is verified at
# import time to round-trip through pmh_vocab.flags_from_text back to its
# category (so the triage/disposition/v3 tools re-derive the same flag the
# loader set). If a preferred phrase fails the round-trip, we fall back to the
# first single-category keyword in PMH_KEYWORD_MAP that maps to that category.
_PREFERRED_PMH_PHRASE = {
    "Blood and Blood-Forming Organs": "anemia",
    "Circulatory": "congestive heart failure",
    "Digestive": "gerd",
    "Endocrine, Nutritional, Metabolic": "diabetes",
    "Genitourinary": "chronic kidney disease",
    "Infectious and Parasitic": "hiv",
    "Injury and Poisoning": "prior fracture",
    "Mental Disorders": "depression",
    "Musculoskeletal": "arthritis",
    "Nervous System and Sense Organs": "epilepsy",
    "Other": "cancer",
    "Respiratory": "copd",
    "Skin and Subcutaneous Tissue": "psoriasis",
}


def _build_category_to_phrase() -> dict:
    """Return {category: phrase}, each phrase guaranteed to map back to its
    category via pmh_vocab.flags_from_text (round-trip safe)."""
    out = {}
    for cat in PMH_CATEGORIES:
        preferred = _PREFERRED_PMH_PHRASE.get(cat)
        if preferred and pmh_flags_from_text(preferred) == {cat}:
            out[cat] = preferred
            continue
        # Fall back: first keyword that maps SOLELY to this category.
        chosen = None
        for kw, val in PMH_KEYWORD_MAP.items():
            if isinstance(val, str) and val == cat and pmh_flags_from_text(kw) == {cat}:
                chosen = kw
                break
        out[cat] = chosen or (preferred or cat.split(",")[0].lower())
    return out


PMH_CATEGORY_TO_PHRASE = _build_category_to_phrase()


def prior_history_text(fired_categories) -> str:
    """Deterministic reverse-map of fired pmh_<category> flags -> a comma-joined
    free-text PMH string a patient might report. Empty string if nothing fired."""
    phrases = [PMH_CATEGORY_TO_PHRASE[c] for c in PMH_CATEGORIES if c in set(fired_categories)]
    return ", ".join(phrases)


# Small field normalizers (tabular raw -> crew input vocabulary)
def transport_to_input(raw: str) -> str:
    r = str(raw).strip().upper()
    return {
        "AMBULANCE": "ambulance",
        "WALK IN": "walk_in",
        "HELICOPTER": "helicopter",
    }.get(r, "other" if r not in ("", "NAN", "UNKNOWN") else "unknown")


def gender_to_input(raw: str) -> str:
    r = str(raw).strip().upper()
    return {"M": "male", "F": "female"}.get(r, "unknown")


# Targeted raw pulls (medication names, cardiac rhythm) keyed by stay_id
def pull_medrecon_names(stay_ids) -> dict:
    """{stay_id: 'drug a, drug b'} from medrecon.csv (the patient's home-med
    NAMES - what a patient would actually report to the nurse)."""
    stay_set = set(int(s) for s in stay_ids)
    med = pd.read_csv(MEDRECON_CSV, usecols=["stay_id", "name"])
    med = med[med["stay_id"].isin(stay_set)]
    med = med.dropna(subset=["name"])
    out = {}
    for sid, grp in med.groupby("stay_id"):
        names = [str(n).strip() for n in grp["name"].tolist() if str(n).strip()]
        # De-dup, keep order, cap length so the narrative-side list stays sane.
        seen, uniq = set(), []
        for n in names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                uniq.append(n)
        out[int(sid)] = ", ".join(uniq[:15])
    return out


def pull_rhythm_readings(stay_intime: dict) -> dict:
    """{stay_id: ['sinus', 'atrial fibrillation', ...]} - ALL non-null rhythm
    readings within [intime, intime + 4h], chronological (same leakage guard +
    window as training's `_aggregate_vitalsigns`). Returning the full sequence
    (not just the first) lets the doctor tools reproduce training's rhythm
    aggregation - mode bucket + any-non-sinus `rhythm_irregular` - closing the
    single-reading divergence (#3)."""
    stay_set = set(int(s) for s in stay_intime)
    vs = pd.read_csv(VITALSIGN_CSV, usecols=["stay_id", "charttime", "rhythm"])
    vs = vs[vs["stay_id"].isin(stay_set)].dropna(subset=["rhythm"])
    if vs.empty:
        return {}
    vs["charttime"] = pd.to_datetime(vs["charttime"], errors="coerce")
    intime_s = pd.Series(stay_intime)
    vs["intime"] = vs["stay_id"].map(intime_s)
    vs["intime"] = pd.to_datetime(vs["intime"], errors="coerce")
    window_end = vs["intime"] + pd.Timedelta(hours=4)
    vs = vs[(vs["charttime"] >= vs["intime"]) & (vs["charttime"] <= window_end)]
    vs = vs.sort_values(["stay_id", "charttime"])
    out = {}
    for sid, grp in vs.groupby("stay_id"):
        rhythms = [str(r).strip() for r in grp["rhythm"].tolist() if str(r).strip()]
        if rhythms:
            out[int(sid)] = rhythms
    return out


def pull_vital_trajectory(stay_intime: dict) -> dict:
    """{stay_id: {vital: [chronological readings]}} - the REAL multi-reading
    trajectory within [intime, intime + 4h], same window + clip parity as
    train_nurse_v3._aggregate_vitalsigns. This is what a runtime that collected
    several readings during the stay would have; the benchmark feeds it into the
    doctor tools' new `vital_trajectory_json` arg to undo snapshot degradation."""
    stay_set = set(int(s) for s in stay_intime)
    vs = pd.read_csv(
        VITALSIGN_CSV, usecols=["stay_id", "charttime"] + VITAL_COLS,
    )
    vs = vs[vs["stay_id"].isin(stay_set)].copy()
    if vs.empty:
        return {}
    vs["charttime"] = pd.to_datetime(vs["charttime"], errors="coerce")
    intime_s = pd.to_datetime(pd.Series(stay_intime))
    vs["intime"] = vs["stay_id"].map(intime_s)
    window_end = vs["intime"] + pd.Timedelta(hours=4)
    vs = vs[(vs["charttime"] >= vs["intime"]) & (vs["charttime"] <= window_end)]
    vs = vs.sort_values(["stay_id", "charttime"])
    out = {}
    for sid, grp in vs.groupby("stay_id"):
        traj = {}
        for v in VITAL_COLS:
            vals = [round(float(x), 1) for x in grp[v].tolist() if pd.notna(x)]
            if vals:
                traj[v] = vals
        if traj:
            out[int(sid)] = traj
    return out


# Field extraction (one tabular row -> the full input bundle + ground truth)
def extract_fields(
    row: pd.Series,
    diag_gt: Optional[str],
    dept_gt: Optional[str],
    med_names: str,
    rhythm_readings: Optional[list] = None,
    vital_trajectory: Optional[dict] = None,
) -> dict:
    """Assemble all crew inputs + ground truth for a single stay from the
    disposition-loader row. Nothing is invented; vitals reflect what was
    actually measured (missing -> None), PMH reflects the loader's flags.

    `rhythm_readings` is the full chronological rhythm sequence in the 4h window;
    the first is surfaced as the single `rhythm` field (v2 tool / display) and
    the whole list is carried inside `vital_trajectory['rhythm']` so the v3 +
    disposition tools aggregate it the way training does."""
    arrival = transport_to_input(row["arrival_transport"])
    is_ems = arrival in ("ambulance", "helicopter")
    rhythm_list = list(rhythm_readings or [])
    rhythm_raw = rhythm_list[0] if rhythm_list else ""
    traj = dict(vital_trajectory or {})
    if rhythm_list:
        traj["rhythm"] = rhythm_list

    # Vitals: real measured value when present, None when the triage row had it
    # missing (the patient "doesn't know"). _<vital>_missing flags come from
    # _clean_vitals which imputes the value but records the missing flag.
    vitals = {}
    for v in VITAL_COLS:
        missing = int(row.get(f"{v}_missing", 0)) == 1
        vitals[v] = None if missing else round(float(row[v]), 1)

    # PMH (reverse-mapped from the loader's real pmh_* flags)
    fired = [c for c in PMH_CATEGORIES if int(row.get(f"pmh_{c}", 0)) == 1]
    pmh_text = prior_history_text(fired)
    n_prior_adm = int(row["n_prior_admissions"]) if "n_prior_admissions" in row and \
        not pd.isna(row["n_prior_admissions"]) else -1

    pain = int(row["pain"]) if not pd.isna(row["pain"]) else -1

    return {
        "stay_id": int(row["stay_id"]),
        "subject_id": int(row["subject_id"]),
        # Current ED arrival time - the leakage cutoff the PatientHistoryLookupTool
        # filters prior encounters against (only visits strictly before this are
        # used). Stored as ISO so the benchmark can feed it to the lookup tool.
        "intime": str(row["intime"]),
        "ground_truth": {
            "acuity": int(row["acuity"]),
            "disposition": str(row["disposition"]),
            "admitted": bool(int(row["admitted"])),
            "diagnosis_group": diag_gt,
            "service_group": dept_gt,
        },
        # What the NLP parser / triage tool consume (vitals only surfaced for
        # ambulance/helicopter, matching live inference).
        "triage_inputs": {
            "chief_complaints": str(row["chiefcomplaint"]).strip(),
            "pain_score": pain,
            "age": int(row["age"]),
            "gender": gender_to_input(row["gender"]),
            "arrival_transport": arrival,
            "ems_vitals": {v: vitals[v] for v in VITAL_COLS} if is_ems else None,
            "prior_history": pmh_text,
            "n_prior_admissions": n_prior_adm,
        },
        # What the nurse tool collects (all patients). `vital_trajectory` is the
        # real multi-reading sequence within [intime,+4h] - what a runtime that
        # took several readings would have. Snapshot `vitals` is kept too so the
        # snapshot-fallback path stays testable.
        "nurse_inputs": {
            "vitals": {v: vitals[v] for v in VITAL_COLS},
            "vital_trajectory": traj,
            "rhythm": rhythm_raw,
            "medications_raw": med_names or "",
            "prior_history": pmh_text,
            "n_prior_admissions": n_prior_adm,
        },
        # Filled in by the generation step.
        "narrative": None,
        "grounding": None,
    }


# Grounding validator - enforce "no invented clinical facts" on the narrative
_AGE_PAT = re.compile(r"\b(\d{1,3})[\s-]*(?:years?[\s-]*old|y[/.]?o\b|yo\b|year[\s-]*old)", re.I)
_AGE_PAT2 = re.compile(r"\b(?:i'?m|i am|aged)\s+(\d{1,3})\b", re.I)
_PAIN_PAT = re.compile(r"\b(\d{1,2})\s*(?:/\s*10|out of 10)\b", re.I)
_BP_PAT = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\b")          # blood-pressure-like
_VITAL_NUM_PAT = re.compile(r"\b(?:bpm|mmhg|°|℉|spo2|o2 sat|sat of|temp(?:erature)? of)\b", re.I)

# Clinical jargon a layperson would not say. Flagged so the "stay lay" rule is
# enforced - important because a medical-domain generator (e.g. MedGemma) may
# leak clinical vocabulary into the patient voice, which would unfairly inflate
# downstream parser accuracy (pre-clinicalizing the input; see docs/llm-backend
# §7). Deliberately only clearly-technical terms/abbreviations to avoid false
# positives on borderline words a patient might genuinely use.
_CLINICAL_JARGON_PAT = re.compile(
    r"\b(?:dyspnea|dyspnoea|syncope|syncopal|hematuria|haematuria|epistaxis|brbpr|"
    r"melena|haematemesis|hematemesis|emesis|h[ae]morrhage|myocardial|infarction|"
    r"intracranial|subdural|subarachnoid|tachycardi[ac]|bradycardi[ac]|dysuria|"
    r"diaphoresis|cephalgia|paresthesia|paraesthesia|ischemi[ac]|cerebrovascular|"
    r"febrile|afebrile|edema|oedema|n/?v|s/?p|etoh|sob)\b",
    re.I,
)

# Generation-only completeness anchors - DELIBERATELY SEPARATE from
# preprocessing.LAY_TO_CLINICAL (the eval-time lay->clinical map). This detects
# whether a *lay narrative* covers a complaint, and is intentionally broader, so
# enforcing completeness here does NOT bias the generator toward the eval map's
# exact vocabulary (which would quietly inflate the benchmark). Each entry is
# (clinical trigger tokens that identify the concept from the tabular complaint,
#  lay/clinical anchor substrings that count as "the patient mentioned it").
# Conservative by design: complaints whose concept is not listed are never
# flagged, so the check only enforces completeness for common, well-understood
# families (which are exactly the multi-complaint combos that get dropped).
_COMPLETENESS_ANCHORS = [
    ({"abdominal", "abd", "epigastric", "ruq", "rlq", "luq", "llq"},
     ("stomach", "belly", "abdom", "tummy", "gut", "epigastr")),
    ({"chest"}, ("chest",)),
    ({"dyspnea", "sob", "breath", "breathing"},
     ("breath", "breathe", "winded", "out of air", "catch my")),
    ({"n", "v", "nausea", "vomiting", "emesis"},
     ("nausea", "nause", "vomit", "throw", "threw", "puk", "sick to", "queasy")),
    ({"headache", "ha", "cephalgia"}, ("head", "migraine")),
    ({"back"}, ("back",)),
    ({"fever", "febrile"}, ("fever", "feverish", "temperature", "burning up", "chills", "hot")),
    ({"dizziness", "dizzy", "vertigo"},
     ("dizz", "lighthead", "light headed", "spinning", "vertigo", "woozy")),
    ({"syncope"}, ("passed out", "pass out", "faint", "blacked out", "collapse", "unconscious")),
    ({"cough"}, ("cough",)),
    ({"weakness", "weak"}, ("weak", "strength")),
    ({"fall", "fell"}, ("fell", "fall", "slipped", "tripped")),
    ({"laceration", "lac", "cut"}, ("cut", "lacerat", "gash", "wound", "slice")),
    ({"rash"}, ("rash", "hives", "skin", "itch", "breaking out")),
    ({"seizure", "sz"}, ("seizure", "convuls", "fit", "shaking")),
    ({"brbpr", "hematuria", "epistaxis", "bld", "bleeding", "hemorrhage"},
     ("blood", "bleed")),
    ({"palpitations"}, ("palpitat", "racing", "heart rac", "pounding", "flutter", "skip")),
    ({"sore", "throat"}, ("throat",)),
    ({"diarrhea", "diarrhoea"}, ("diarr", "loose stool", "watery", "the runs")),
    ({"dysuria"}, ("pee", "urinat", "bladder")),
    ({"si", "suicidal"}, ("suicid", "kill myself", "hurt myself", "hurting myself",
                          "harm myself", "harming myself", "self harm", "self-harm",
                          "end my life", "want to die")),
    ({"anxiety"}, ("anxi", "panic", "nervous")),
    ({"hyperglycemia", "hypoglycemia"}, ("sugar", "glucose", "diabet")),
    ({"constipation"}, ("constipat", "bowel", "can't poop", "cant poop", "haven't gone")),
    ({"pain"}, ("pain", "hurt", "ache", "sore", "killing")),  # generic "X pain" fallback
]


def _alpha_tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", str(s).lower()))


def _dropped_complaints(complaint_str: str, narrative: str) -> list:
    """Complaints from the tabular row that appear to be MISSING from the lay
    narrative. Conservative: only checks complaint concepts we have anchors for;
    unknown concepts are skipped (never flagged). Independent of the eval-time
    lay->clinical map by construction (uses _COMPLETENESS_ANCHORS)."""
    nt = " " + str(narrative).lower() + " "
    dropped = []
    for comp in (c.strip() for c in str(complaint_str).split(",") if c.strip()):
        toks = _alpha_tokens(comp)
        if not toks:
            continue
        triggered = [anch for (trig, anch) in _COMPLETENESS_ANCHORS if toks & trig]
        if not triggered:
            continue  # no anchor knowledge for this concept -> don't flag
        covered = any(a in nt for anch in triggered for a in anch) or any(
            len(tok) > 3 and tok in nt for tok in toks
        )
        if not covered:
            dropped.append(comp)
    return dropped


def validate_grounding(narrative: str, fields: dict) -> tuple:
    """Return (ok, reasons). Checks three things:

    1. *Invention* - any stated age/pain must match the row, and the opening
       narrative must not contain fabricated clinical measurements (BP/HR/temp/O2,
       which belong to the nurse stage).
    2. *Clinical leakage* - the narrative must stay in lay language (no clinical
       jargon/abbreviations), so a medical-domain generator can't pre-clinicalize
       the input and unfairly inflate parser accuracy.
    3. *Completeness* - every tabular complaint should be voiced; a dropped
       complaint is flagged (conservatively, for common concepts) so the retry
       loop regenerates. Faithful lay *wording* is still NOT scored - that
       paraphrase is exactly what the E2E test measures."""
    reasons = []
    if not narrative or not narrative.strip():
        return False, ["empty narrative"]

    t = fields["triage_inputs"]
    text = narrative.strip()

    # Age contradiction
    ages = [int(m) for m in _AGE_PAT.findall(text)] + [int(m) for m in _AGE_PAT2.findall(text)]
    for a in ages:
        if a != t["age"]:
            reasons.append(f"stated age {a} != row age {t['age']}")

    # Pain contradiction (only if the row has a known pain score)
    if t["pain_score"] >= 0:
        for m in _PAIN_PAT.findall(text):
            if int(m) != t["pain_score"]:
                reasons.append(f"stated pain {m}/10 != row pain {t['pain_score']}/10")

    # Fabricated vital measurements in the opening narrative
    if _BP_PAT.search(text):
        reasons.append("narrative contains a blood-pressure-like number (vitals are nurse-stage)")
    if _VITAL_NUM_PAT.search(text):
        reasons.append("narrative contains a clinical-measurement token (bpm/mmHg/temp/O2)")

    # Clinical-jargon leakage (narrative must stay lay)
    jargon = sorted({m.group(0).lower() for m in _CLINICAL_JARGON_PAT.finditer(text)})
    if jargon:
        reasons.append(f"narrative uses clinical jargon (should be lay): {', '.join(jargon)}")

    # Completeness - flag dropped complaints
    dropped = _dropped_complaints(t["chief_complaints"], text)
    if dropped:
        reasons.append(f"narrative omits complaint(s): {', '.join(dropped)}")

    return (len(reasons) == 0), reasons


# Case Generation Agent (offline CrewAI crew - never joins the live pipeline)
@CrewBase
class CaseGenerationCrew:
    """Single-agent crew that rephrases structured ED data as a first-person
    patient narrative.

    Uses DEDICATED config files (not the live crew's config/agents.yaml /
    config/tasks.yaml) because @CrewBase maps every task in a tasks.yaml to an
    @agent method on the class - sharing the files would couple this offline
    crew to the live ProiectLicenta pipeline and break both.
    """

    agents_config = "config/case_generation_agents.yaml"
    tasks_config = "config/case_generation_tasks.yaml"

    agents: list
    tasks: list

    @agent
    def case_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["case_generator"],  # type: ignore[index]
            llm=get_llm(),
            verbose=False,
        )

    @task
    def generate_case_task(self) -> Task:
        return Task(config=self.tasks_config["generate_case_task"])  # type: ignore[index]

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
        )


def _parse_narrative(raw_output: str) -> str:
    """Pull the narrative string out of the agent's JSON output, tolerating
    code fences and stray prose."""
    s = str(raw_output).strip()
    # Strip ```json ... ``` fences if present.
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.S).strip()
    # Try to locate a JSON object.
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("narrative"):
                return str(obj["narrative"]).strip()
        except json.JSONDecodeError:
            pass
    # Fall back: treat the whole thing as the narrative.
    return s


def generate_narrative(fields: dict, max_attempts: int = 3) -> dict:
    """Kick off the generation crew for one case, validate grounding, retry on
    failure. Returns {'narrative': str, 'ok': bool, 'reasons': [...], 'attempts': n}."""
    t = fields["triage_inputs"]
    patient_data = {
        "chief_complaints": t["chief_complaints"],
        "age": t["age"],
        "gender": t["gender"],
        "arrival_transport": t["arrival_transport"],
        "pain_score": ("unknown" if t["pain_score"] < 0 else t["pain_score"]),
    }
    last_narrative, last_reasons = "", ["not generated"]
    for attempt in range(1, max_attempts + 1):
        result = CaseGenerationCrew().crew().kickoff(
            inputs={"patient_data": json.dumps(patient_data, ensure_ascii=False)}
        )
        narrative = _parse_narrative(result.raw if hasattr(result, "raw") else result)
        ok, reasons = validate_grounding(narrative, fields)
        last_narrative, last_reasons = narrative, reasons
        if ok:
            return {"narrative": narrative, "ok": True, "reasons": [], "attempts": attempt}
    return {"narrative": last_narrative, "ok": False, "reasons": last_reasons,
            "attempts": max_attempts}


# Sampling + extraction (runs the heavy loaders once)
def _doctor_v3_test_stay_ids(df_nurse: pd.DataFrame) -> set:
    """Reproduce the doctor-v3 nurse benchmark split (stratify on diagnosis)
    and return the set of held-out test stay_ids."""
    meta = json.loads((DOCTOR_V3_DIR / "metadata.json").read_text(encoding="utf-8"))
    diag_labels = meta["diagnosis_labels"]
    diag_map = {l: i for i, l in enumerate(diag_labels)}
    y_diag = df_nurse["diagnosis_group"].map(diag_map).reset_index(drop=True)
    idx = np.arange(len(df_nurse))
    _, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_diag,
    )
    return set(df_nurse.iloc[test_idx]["stay_id"].astype(int).tolist())


def sample_and_extract(
    n_admitted: int = DEFAULT_N_ADMITTED,
    n_discharged: int = DEFAULT_N_DISCHARGED,
    seed: int = DEFAULT_SEED,
) -> tuple:
    """Run the disposition + nurse_v3 loaders once, reproduce the held-out
    splits, sample a stratified admit/discharge set, and extract the full
    input bundle + ground truth + cached feature rows for each case.

    Returns (cases, feature_cache) where feature_cache[stay_id] holds the
    pre-built feature rows the feature-vector baseline needs.
    """
    # Lazy imports - heavy modules with import-time cost.
    import gc
    from proiect_licenta.training import train_doctor_disposition as dispo
    from proiect_licenta.training import train_nurse_v3 as nurse_v3
    from proiect_licenta.training import train_doctor_v3 as doctor_v3

    # The disposition build_features loads the triage v3 acuity model, which was
    # pickled with a custom `neg_quadratic_kappa` eval_metric whose __module__ is
    # '__main__'. On a fresh CLI process __main__ lacks that name, so joblib.load
    # raises AttributeError. The runtime tools inject a shim before loading; do
    # the same here so the loaders can unpickle the acuity artifact.
    from proiect_licenta.tools.triage_tool import _ensure_pickle_compat_in_main
    _ensure_pickle_compat_in_main()

    print("Case generation: sampling from held-out test splits")

    # Don't build_features on the full 418K rows: densifying the 2070-col TF-IDF
    # matrix there needs ~7 GB and OOMs. The split only needs labels for
    # stratification, so we load the cleaned frames, reproduce the splits from
    # labels alone, sample the stay_ids, and only then build features on the tiny
    # sampled sub-frames. build_features is a pure fit=False transform, so per-row
    # results match building on the full frame and slicing.

    # Disposition loader (full population: acuity + dispo + all inputs)
    print("\n[1/4] Loading disposition dataset (full population)...")
    df_d = dispo.load_and_clean_data().reset_index(drop=True)
    y_adm = df_d["admitted"].astype(int)

    # Reproduce the disposition test split (same as benchmark_doctor_disposition)
    # using indices + labels only - no feature matrix required.
    idx = np.arange(len(df_d))
    _, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_adm,
    )
    test_idx = np.sort(test_idx)
    admit_rate = float(y_adm.iloc[test_idx].mean())
    # Keep only the 83K-row test slice and free the full 418K frame BEFORE the
    # nurse loader runs. Everything downstream (intime map, df_d_sub features) is
    # a subset of the test split, so this is behaviour-preserving - and it avoids
    # holding two full MIMIC frames in memory at once, which OOMs on ~14 GB RAM.
    df_test = df_d.iloc[test_idx].copy()
    del df_d, y_adm, idx
    gc.collect()
    print(f"  disposition test split: {len(test_idx):,} rows "
          f"(admit rate {admit_rate:.3f})")

    # Nurse v3 loader (admitted, diagnosis/department ground truth)
    print("\n[2/4] Loading doctor v3 nurse dataset (admitted, diag/dept labels)...")
    df_n = nurse_v3.load_and_clean_data().reset_index(drop=True)
    diag_by_stay = dict(zip(df_n["stay_id"].astype(int), df_n["diagnosis_group"]))
    dept_by_stay = dict(zip(df_n["stay_id"].astype(int), df_n["service_group"]))
    doctor_test_ids = _doctor_v3_test_stay_ids(df_n)  # label-only split
    print(f"  admitted/non-catch-all rows: {len(df_n):,} | "
          f"doctor-v3 test stays: {len(doctor_test_ids):,}")

    # Build candidate pools
    rng = np.random.RandomState(seed)
    test_stay_ids = df_test["stay_id"].astype(int).values

    admitted_mask = df_test["admitted"].astype(int).values == 1
    admitted_candidates = [
        sid for sid in test_stay_ids[admitted_mask]
        if int(sid) in doctor_test_ids and int(sid) in diag_by_stay
    ]
    discharged_candidates = list(test_stay_ids[~admitted_mask])

    if len(admitted_candidates) < n_admitted:
        raise RuntimeError(
            f"Only {len(admitted_candidates)} admitted candidates in the "
            f"disposition∩doctor-v3 test intersection; need {n_admitted}."
        )
    if len(discharged_candidates) < n_discharged:
        raise RuntimeError(
            f"Only {len(discharged_candidates)} discharged candidates; "
            f"need {n_discharged}."
        )

    chosen_admit = [int(s) for s in
                    rng.choice(admitted_candidates, size=n_admitted, replace=False)]
    chosen_disch = [int(s) for s in
                    rng.choice(discharged_candidates, size=n_discharged, replace=False)]
    chosen = chosen_admit + chosen_disch
    chosen_set = set(chosen)
    chosen_admit_set = set(chosen_admit)
    print(f"\n[3/4] Sampled {len(chosen)} cases "
          f"({n_admitted} admitted + {n_discharged} discharged)")

    # Targeted raw pulls (med names, rhythm, vital trajectory)
    print("\n[4/4] Pulling raw medication names + cardiac rhythm + vital trajectory...")
    med_names = pull_medrecon_names(chosen)
    intime_by_stay = dict(zip(df_test["stay_id"].astype(int), df_test["intime"]))
    rhythm_readings = pull_rhythm_readings({s: intime_by_stay[s] for s in chosen})
    vital_traj = pull_vital_trajectory({s: intime_by_stay[s] for s in chosen})

    # Build features ONLY for the sampled rows (memory-safe)
    print("\n  Building features for the sampled rows only "
          "(sub-frame transform - memory-safe)...")
    df_d_sub = df_test[df_test["stay_id"].astype(int).isin(chosen_set)].copy().reset_index(drop=True)
    feats_d_sub = dispo.build_features(df_d_sub).reset_index(drop=True)
    dsub_row_by_stay = {int(s): i for i, s in enumerate(df_d_sub["stay_id"].astype(int))}

    df_n_sub = df_n[df_n["stay_id"].astype(int).isin(chosen_admit_set)].copy().reset_index(drop=True)
    if len(df_n_sub):
        feats_n_sub = nurse_v3.build_features(df_n_sub).reset_index(drop=True)
        feats_vb_sub = doctor_v3.build_features(df_n_sub).reset_index(drop=True)
        nsub_row_by_stay = {int(s): i for i, s in enumerate(df_n_sub["stay_id"].astype(int))}
    else:
        feats_n_sub = feats_vb_sub = None
        nsub_row_by_stay = {}

    cases, feature_cache = [], {}
    for sid in chosen:
        di = dsub_row_by_stay[sid]
        row = df_d_sub.iloc[di]
        is_admitted = bool(int(row["admitted"]))
        fields = extract_fields(
            row,
            diag_gt=diag_by_stay.get(sid) if is_admitted else None,
            dept_gt=dept_by_stay.get(sid) if is_admitted else None,
            med_names=med_names.get(sid, ""),
            rhythm_readings=rhythm_readings.get(sid, []),
            vital_trajectory=vital_traj.get(sid, {}),
        )
        cases.append(fields)

        cache = {"dispo_features": feats_d_sub.iloc[[di]].reset_index(drop=True)}
        if is_admitted and sid in nsub_row_by_stay:
            ni = nsub_row_by_stay[sid]
            cache["nurse_v3_features"] = feats_n_sub.iloc[[ni]].reset_index(drop=True)
            cache["v3_base_features"] = feats_vb_sub.iloc[[ni]].reset_index(drop=True)
        feature_cache[sid] = cache

    return cases, feature_cache


# Orchestration: generate narratives + persist
def generate_and_save_cases(
    n_admitted: int = DEFAULT_N_ADMITTED,
    n_discharged: int = DEFAULT_N_DISCHARGED,
    seed: int = DEFAULT_SEED,
    limit: Optional[int] = None,
    out_dir: Optional["Path"] = None,
) -> list:
    """Sample, generate validated narratives via the LLM, and persist
    cases.json + sampled_features.pkl.

    By default these go to data/derived/synthetic_cases/ (the canonical set the
    E2E benchmark loads). Pass ``out_dir`` to write to a separate directory -
    used by Experiment B to keep each LLM backend's narratives apart
    (e.g. synthetic_cases_medgemma/) so the Case-Generation comparison never
    overwrites the Flash-generated canonical cases."""
    out_dir = SYNTH_DIR if out_dir is None else Path(out_dir)
    cases_json = out_dir / "cases.json"
    features_pkl = out_dir / "sampled_features.pkl"
    cases, feature_cache = sample_and_extract(n_admitted, n_discharged, seed)

    if limit is not None:
        cases = cases[:limit]
        feature_cache = {c["stay_id"]: feature_cache[c["stay_id"]] for c in cases}

    print(f"\n{'='*70}\n  Generating {len(cases)} narratives (LLM + grounding check)\n{'='*70}")
    n_flagged = 0
    for i, case in enumerate(cases, 1):
        cc = case["triage_inputs"]["chief_complaints"]
        print(f"\n  [{i}/{len(cases)}] stay {case['stay_id']} "
              f"({'admit' if case['ground_truth']['admitted'] else 'discharge'}): {cc[:60]}")
        g = generate_narrative(case)
        case["narrative"] = g["narrative"]
        case["grounding"] = {"ok": g["ok"], "reasons": g["reasons"], "attempts": g["attempts"]}
        flag = "" if g["ok"] else f"  [FLAGGED: {'; '.join(g['reasons'])}]"
        print(f"      -> {case['narrative'][:90]}{flag}")
        if not g["ok"]:
            n_flagged += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "llm_backend": os.getenv("LLM_BACKEND", "flash"),
        "seed": seed,
        "n_admitted": n_admitted,
        "n_discharged": n_discharged,
        "n_cases": len(cases),
        "cases": cases,
    }
    cases_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump(feature_cache, features_pkl)

    print(f"  Saved {len(cases)} cases -> {cases_json}")
    print(f"  Saved feature cache -> {features_pkl}")
    print(f"  Grounding: {len(cases) - n_flagged}/{len(cases)} clean, {n_flagged} flagged")
    return cases


def load_cases() -> dict:
    """Load the saved cases payload. Raises if generation hasn't run."""
    if not CASES_JSON.exists():
        raise FileNotFoundError(
            f"{CASES_JSON} not found. Run `uv run generate_cases` first."
        )
    return json.loads(CASES_JSON.read_text(encoding="utf-8"))


def load_feature_cache() -> dict:
    if not FEATURES_PKL.exists():
        raise FileNotFoundError(
            f"{FEATURES_PKL} not found. Run `uv run generate_cases` first."
        )
    return joblib.load(FEATURES_PKL)
