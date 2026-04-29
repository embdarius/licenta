"""
Data Pipeline for Doctor Agent v3 with nurse data — MIMIC-IV ED

Same architecture and feature set as Doctor v2 in Phase A:
  - chief complaints + demographics + triage predictions (cascading)
  - vital signs (snapshot from triage.csv) + clinical flags
  - medication features (medrecon.csv)

Two changes vs v2:
  1. The "Symptoms, Signs, Ill-Defined" catch-all class is excluded
     from training and evaluation (13-class label space).
  2. The 100K stratified sample cap is removed; trains on the full
     filtered set (~102K admitted patients).

Phase B will add longitudinal vitals + rhythm from `vitalsign.csv` —
those features hook in here, alongside the existing snapshot vitals.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths (canonical layout in proiect_licenta.paths)
# ---------------------------------------------------------------------------
# v3 with-nurse reuses triage v1 base artifacts (tfidf, severity_map,
# cascading triage models) for feature consistency with v3 base, then adds
# vital + medication features on top.
from proiect_licenta.paths import (
    TRIAGE_V1_DIR as TRIAGE_MODELS_DIR,
    DOCTOR_V3_DIR as DOCTOR_MODELS_DIR,
    TRIAGE_CSV, VITALSIGN_CSV, EDSTAYS_CSV, PATIENTS_CSV,
    DIAGNOSIS_CSV, SERVICES_CSV, MEDRECON_CSV,
)

# ---------------------------------------------------------------------------
# Reuse from doctor pipeline (single source of truth for label maps)
# ---------------------------------------------------------------------------
from proiect_licenta.training.train_doctor import (
    CATCH_ALL_LABEL,
    DIAGNOSIS_GROUP_MAP,
    SERVICE_GROUP_MAP,
    DEPARTMENT_NAMES,
)
from proiect_licenta.preprocessing import normalize_complaint_text

# ---------------------------------------------------------------------------
# Reuse vital + medication helpers from train_nurse (v2)
# ---------------------------------------------------------------------------
# These helpers are stable and shared between v2 and v3 — no need to
# duplicate them. Phase B will add a sibling `_aggregate_vitalsigns`
# helper here for longitudinal aggregation.
from proiect_licenta.training.train_nurse import (
    _aggregate_medications,
    _clean_vitals,
    classify_medication,
    MED_CATEGORY_KEYWORDS,
)
from proiect_licenta.tools.med_vocab import (
    DRUG_NAME_MAP, MED_CLASS_KEYWORDS, MED_CATEGORIES,
    flags_from_row,
)


# ---------------------------------------------------------------------------
# Longitudinal vitalsign.csv aggregation (Phase B)
# ---------------------------------------------------------------------------
# vitalsign.csv records vitals at multiple ED time points per stay. We
# aggregate per stay into trajectory features (min/max/last/delta) and
# rhythm flags. Time-windowed to [intime, intime + 4h] to avoid leaking
# the late-stay disposition decision into features (the disposition-window
# leakage guard from the plan).
#
# At inference the nurse only collects a single snapshot. doctor_tool_v3
# computes the trajectory features as if min == max == last == snapshot
# and delta == 0; the model learned during training when trajectory matters
# and gracefully falls back to the snapshot at inference.

_RHYTHM_BUCKETS = [
    "sinus", "sinus_tachy", "sinus_brady", "afib_flutter",
    "paced", "av_block", "svt", "other",
]
_VITAL_COLS_LONG = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
_LONG_WINDOW_HOURS = 4.0  # keep readings within first 4h of stay


def _normalize_rhythm(raw: str) -> str:
    """Bucket a rhythm string into a small categorical label.

    The free-text `rhythm` column has dozens of variants ("Sinus Rhythm",
    "sr", "NSR", "sinus", "afib", "Atrial Fibrillation", "1° AV Block",
    ...). We collapse them into a fixed taxonomy aligned with what a nurse
    can plausibly read from a monitor at inference time.
    """
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""

    # Normal sinus
    if s in {"sr", "nsr", "sinus", "normal sinus rhythm", "sinus rhythm",
             "normal", "regular"}:
        return "sinus"
    if "sinus" in s and "tach" in s:
        return "sinus_tachy"
    if "sinus" in s and "brady" in s:
        return "sinus_brady"
    if s in {"st"}:
        return "sinus_tachy"
    if s in {"sb"}:
        return "sinus_brady"

    # Atrial fibrillation / flutter
    if any(t in s for t in ("afib", "atrial fibrillation", "atrial flutter",
                             "aflutter", " af ", "a fib", "a-fib")) or \
       s in {"af", "afib", "aflutter", "a-fib"}:
        return "afib_flutter"

    # Paced
    if "paced" in s or "pacer" in s or "pacemaker" in s:
        return "paced"

    # AV Block
    if "av block" in s or "1°" in s or "heart block" in s:
        return "av_block"

    # SVT (supraventricular tachycardia)
    if "svt" in s or "supraventricular" in s:
        return "svt"

    # Catch-all sinus token (e.g., "Sinus Arrythmia")
    if "sinus" in s:
        return "sinus"

    return "other"


def _aggregate_vitalsigns(
    stays_df: pd.DataFrame,
    vitalsign_csv_path,
) -> pd.DataFrame:
    """Aggregate vitalsign.csv readings per stay into longitudinal features.

    Parameters
    ----------
    stays_df : DataFrame with columns ["stay_id", "intime"] for the stays
        we care about. Used to (a) restrict the heavy vitalsign.csv load
        to relevant stays and (b) apply the [intime, intime + 4h] window.
    vitalsign_csv_path : Path to the MIMIC-IV ED vitalsign.csv file.

    Returns
    -------
    DataFrame keyed on stay_id with columns:
      - <vital>_min, <vital>_max, <vital>_last, <vital>_delta for each
        numeric vital (6 vitals × 4 stats = 24 cols)
      - n_fever_readings, n_tachycardia_readings, n_bradycardia_readings,
        n_tachypnea_readings, n_hypoxia_readings,
        n_hypertension_readings, n_hypotension_readings (7 cols)
      - rhythm_<bucket> one-hot for each bucket in _RHYTHM_BUCKETS (8 cols)
      - rhythm_irregular: 1 if any reading is non-sinus rhythm (1 col)
      - has_longitudinal_vitals: 1 if any reading was found for this stay
    """
    print(f"  Loading vitalsign.csv (this is the heavy one, ~115 MB)...")
    keep_stays = set(stays_df["stay_id"].astype(int).tolist())

    # Read in chunks, only keeping the columns + stays we need.
    chunks = []
    for chunk in pd.read_csv(
        vitalsign_csv_path,
        usecols=["stay_id", "charttime", "temperature", "heartrate",
                 "resprate", "o2sat", "sbp", "dbp", "rhythm"],
        chunksize=1_000_000,
    ):
        chunk = chunk[chunk["stay_id"].isin(keep_stays)]
        if len(chunk):
            chunks.append(chunk)

    if not chunks:
        print("  vitalsign.csv: no rows matched the admitted stays — empty result.")
        return pd.DataFrame(columns=["stay_id"])

    vs = pd.concat(chunks, ignore_index=True)
    print(f"  vitalsign.csv: {len(vs):,} rows across {vs['stay_id'].nunique():,} stays "
          f"(after filtering to admitted)")

    # ── Time-window: keep readings within [intime, intime + 4h] ──
    vs["charttime"] = pd.to_datetime(vs["charttime"], errors="coerce")
    vs = vs.merge(stays_df[["stay_id", "intime"]], on="stay_id", how="inner")
    # stays_df.intime is already datetime (set in load_and_clean_data)
    delta_h = (vs["charttime"] - vs["intime"]).dt.total_seconds() / 3600.0
    vs = vs[(delta_h >= 0) & (delta_h <= _LONG_WINDOW_HOURS)].copy()
    print(f"  After [intime, intime+{_LONG_WINDOW_HOURS:.0f}h] window: {len(vs):,} rows")

    if len(vs) == 0:
        return pd.DataFrame(columns=["stay_id"])

    # Convert numeric vitals
    for c in _VITAL_COLS_LONG:
        vs[c] = pd.to_numeric(vs[c], errors="coerce")

    # Clip to plausible ranges (same as snapshot _clean_vitals to keep parity)
    vs["temperature"] = vs["temperature"].clip(90, 110)
    vs["heartrate"] = vs["heartrate"].clip(20, 250)
    vs["resprate"] = vs["resprate"].clip(4, 60)
    vs["o2sat"] = vs["o2sat"].clip(50, 100)
    vs["sbp"] = vs["sbp"].clip(50, 300)
    vs["dbp"] = vs["dbp"].clip(20, 200)

    # Sort by charttime so groupby().last() / .first() make temporal sense
    vs = vs.sort_values(["stay_id", "charttime"])

    # ── Per-stay numeric aggregates (skip-NaN) ──
    grouped = vs.groupby("stay_id")
    agg_records = []
    for stay_id, sub in grouped:
        rec = {"stay_id": stay_id}
        for c in _VITAL_COLS_LONG:
            col = sub[c].dropna()
            if len(col):
                rec[f"{c}_min"] = float(col.min())
                rec[f"{c}_max"] = float(col.max())
                rec[f"{c}_last"] = float(col.iloc[-1])
                rec[f"{c}_delta"] = float(col.iloc[-1] - col.iloc[0])
            else:
                # All-null for this vital in this stay; leave NaN, filled later.
                rec[f"{c}_min"] = np.nan
                rec[f"{c}_max"] = np.nan
                rec[f"{c}_last"] = np.nan
                rec[f"{c}_delta"] = np.nan

        # Abnormal-reading counts (all use the same thresholds as _clean_vitals)
        rec["n_fever_readings"] = int((sub["temperature"] > 100.4).sum())
        rec["n_tachycardia_readings"] = int((sub["heartrate"] > 100).sum())
        rec["n_bradycardia_readings"] = int((sub["heartrate"] < 60).sum())
        rec["n_tachypnea_readings"] = int((sub["resprate"] > 20).sum())
        rec["n_hypoxia_readings"] = int((sub["o2sat"] < 94).sum())
        rec["n_hypertension_readings"] = int((sub["sbp"] > 140).sum())
        rec["n_hypotension_readings"] = int((sub["sbp"] < 90).sum())

        # Rhythm: bucket every non-null reading, take mode (most common)
        # AND set rhythm_irregular if ANY reading is non-sinus.
        bucketed = [_normalize_rhythm(r) for r in sub["rhythm"].fillna("").tolist()]
        bucketed = [b for b in bucketed if b]  # drop empties
        for bucket in _RHYTHM_BUCKETS:
            rec[f"rhythm_{bucket}"] = 0
        rec["rhythm_irregular"] = 0
        if bucketed:
            # Dominant rhythm = most common bucket
            from collections import Counter
            top = Counter(bucketed).most_common(1)[0][0]
            rec[f"rhythm_{top}"] = 1
            # Irregular if any reading is non-sinus.
            rec["rhythm_irregular"] = int(
                any(b not in ("sinus", "") for b in bucketed)
            )

        rec["has_longitudinal_vitals"] = 1
        agg_records.append(rec)

    result = pd.DataFrame(agg_records)
    print(f"  Per-stay longitudinal aggregates: {len(result):,} stays")
    return result


def _fill_longitudinal_vitals(df: pd.DataFrame):
    """Fill missing longitudinal-vital columns in-place after the left join.

    For stays with no vitalsign.csv readings (or all readings outside the
    time window), we fall back to the triage snapshot already on `df`:
      - <vital>_min/_max/_last := snapshot value
      - <vital>_delta := 0
      - n_<flag>_readings := the snapshot clinical flag (0 or 1)
      - rhythm_*  := all 0
      - rhythm_irregular := 0
      - has_longitudinal_vitals := 0

    This mirrors the inference-time behavior in doctor_tool_v3 (degraded
    trajectory from a single snapshot), so the model learns a consistent
    representation across stays with and without longitudinal coverage.
    """
    has_long = df["has_longitudinal_vitals"].fillna(0).astype(int)
    df["has_longitudinal_vitals"] = has_long

    snapshot_to_flag = {
        "n_fever_readings": "fever",
        "n_tachycardia_readings": "tachycardia",
        "n_bradycardia_readings": "bradycardia",
        "n_tachypnea_readings": "tachypnea",
        "n_hypoxia_readings": "hypoxia",
        "n_hypertension_readings": "hypertension",
        "n_hypotension_readings": "hypotension",
    }

    for vital in _VITAL_COLS_LONG:
        snap = df[vital]  # already cleaned + median-imputed by _clean_vitals
        for stat in ("min", "max", "last"):
            col = f"{vital}_{stat}"
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].where(has_long == 1, snap)
            df[col] = df[col].fillna(snap)
        delta_col = f"{vital}_delta"
        if delta_col not in df.columns:
            df[delta_col] = np.nan
        df[delta_col] = df[delta_col].where(has_long == 1, 0.0).fillna(0.0)

    for count_col, flag_col in snapshot_to_flag.items():
        if count_col not in df.columns:
            df[count_col] = 0
        df[count_col] = df[count_col].where(has_long == 1, df[flag_col]).fillna(0).astype(int)

    for bucket in _RHYTHM_BUCKETS:
        col = f"rhythm_{bucket}"
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0).astype(int)

    if "rhythm_irregular" not in df.columns:
        df["rhythm_irregular"] = 0
    df["rhythm_irregular"] = df["rhythm_irregular"].fillna(0).astype(int)


# Names of the new Phase B feature columns, exposed for the inference tool.
LONG_VITAL_FEATURE_COLS = (
    [f"{v}_min" for v in _VITAL_COLS_LONG]
    + [f"{v}_max" for v in _VITAL_COLS_LONG]
    + [f"{v}_last" for v in _VITAL_COLS_LONG]
    + [f"{v}_delta" for v in _VITAL_COLS_LONG]
    + [
        "n_fever_readings", "n_tachycardia_readings", "n_bradycardia_readings",
        "n_tachypnea_readings", "n_hypoxia_readings",
        "n_hypertension_readings", "n_hypotension_readings",
    ]
    + [f"rhythm_{b}" for b in _RHYTHM_BUCKETS]
    + ["rhythm_irregular", "has_longitudinal_vitals"]
)


# ---------------------------------------------------------------------------
# 1. Load & Clean Data — admitted patients + vitals + meds, catch-all dropped
# ---------------------------------------------------------------------------
def load_and_clean_data() -> pd.DataFrame:
    """Load admitted patient data with diagnosis, service, vitals, and meds.

    Same as train_nurse.load_and_clean_data, but filters out the catch-all
    class after applying the diagnosis grouping.
    """
    print("=" * 60)
    print("STEP 1: Loading data (v3 with-nurse — admitted + vitals + meds, "
          "catch-all excluded)")
    print("=" * 60)

    # Load triage (includes vitals)
    triage = pd.read_csv(TRIAGE_CSV)
    print(f"  triage.csv: {len(triage):,} rows")

    # Load edstays
    edstays = pd.read_csv(
        EDSTAYS_CSV,
        usecols=["subject_id", "stay_id", "hadm_id", "intime", "gender",
                 "arrival_transport", "disposition"],
    )
    print(f"  edstays.csv: {len(edstays):,} rows")

    # Load patients
    patients = pd.read_csv(
        PATIENTS_CSV,
        usecols=["subject_id", "anchor_age", "anchor_year"],
    )
    print(f"  patients.csv: {len(patients):,} rows")

    # Load primary diagnoses
    diag = pd.read_csv(DIAGNOSIS_CSV)
    diag = diag[diag["seq_num"] == 1][["stay_id", "category", "icd_code", "icd_title"]]
    print(f"  categorized_diagnosis.csv (primary): {len(diag):,} rows")

    # Load services (first per admission)
    services = pd.read_csv(SERVICES_CSV)
    services["transfertime"] = pd.to_datetime(services["transfertime"])
    services_first = (
        services.sort_values("transfertime")
        .groupby("hadm_id").first().reset_index()[["hadm_id", "curr_service"]]
    )
    print(f"  services.csv (first per admission): {len(services_first):,} rows")

    # Load medications and aggregate per stay
    print("  Loading medrecon.csv and aggregating per stay...")
    med = pd.read_csv(MEDRECON_CSV, usecols=["stay_id", "name", "etcdescription"])
    med_features = _aggregate_medications(med)
    print(f"  medrecon.csv: {len(med):,} rows -> {len(med_features):,} stay-level records")

    # ── Filter to admitted patients ──
    admitted = edstays[edstays["disposition"] == "ADMITTED"].copy()
    print(f"\n  Admitted stays: {len(admitted):,}")

    # ── Merge all tables ──
    df = triage.merge(admitted, on=["subject_id", "stay_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="left")
    df = df.merge(diag, on="stay_id", how="inner")
    df = df.merge(services_first, on="hadm_id", how="inner")
    df = df.merge(med_features, on="stay_id", how="left")  # left join: not all have meds
    print(f"  After full merge: {len(df):,}")

    # ── Compute age ──
    df["intime"] = pd.to_datetime(df["intime"])
    df["visit_year"] = df["intime"].dt.year
    df["age"] = df["anchor_age"] + (df["visit_year"] - df["anchor_year"])
    df["age"] = df["age"].clip(0, 120).fillna(50).astype(int)

    # ── Clean chief complaints ──
    initial = len(df)
    df = df.dropna(subset=["chiefcomplaint", "category", "curr_service"])
    df = df[df["chiefcomplaint"].str.strip() != ""]
    print(f"  After dropping missing data: {len(df):,} (dropped {initial - len(df):,})")

    # ── Apply category groupings ──
    df["diagnosis_group"] = df["category"].map(DIAGNOSIS_GROUP_MAP).fillna("Other")
    df["service_group"] = df["curr_service"].map(SERVICE_GROUP_MAP).fillna("OTHER")

    # ── v3-specific: drop catch-all class ──
    before = len(df)
    df = df[df["diagnosis_group"] != CATCH_ALL_LABEL].reset_index(drop=True)
    dropped = before - len(df)
    print(f"\n  [v3] Filtered catch-all '{CATCH_ALL_LABEL}': "
          f"{before:,} -> {len(df):,} ({dropped:,} dropped, "
          f"{100 * dropped / before:.1f}%)")

    # ── Clean pain ──
    df["pain_triage"] = pd.to_numeric(df["pain"], errors="coerce")
    df["pain_missing"] = df["pain_triage"].isna().astype(int)
    df["pain"] = df["pain_triage"].fillna(-1).astype(int)
    df.loc[df["pain"] > 10, "pain"] = -1
    df.loc[df["pain"] < 0, "pain"] = -1

    # ── Encode demographics ──
    df["gender_male"] = (df["gender"] == "M").astype(int)
    df["arrival_ambulance"] = (df["arrival_transport"] == "AMBULANCE").astype(int)
    df["arrival_helicopter"] = (df["arrival_transport"] == "HELICOPTER").astype(int)
    df["arrival_walk_in"] = (df["arrival_transport"] == "WALK IN").astype(int)

    # ── Acuity ──
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    df = df[df["acuity"].between(1, 5)]
    df["acuity"] = df["acuity"].astype(int)
    df["admitted"] = 1

    # ── Clean vital signs (snapshot from triage.csv) ──
    _clean_vitals(df)

    # ── Phase B: longitudinal vitals + rhythm from vitalsign.csv ──
    print("\n  Aggregating longitudinal vitals from vitalsign.csv...")
    long_vitals = _aggregate_vitalsigns(
        df[["stay_id", "intime"]],
        VITALSIGN_CSV,
    )
    df = df.merge(long_vitals, on="stay_id", how="left")
    _fill_longitudinal_vitals(df)
    coverage = (df["has_longitudinal_vitals"] == 1).mean()
    print(f"  Longitudinal vitals coverage: {100*coverage:.1f}% of admitted stays")

    # ── Fill missing medication flags (patients with no medrecon data) ──
    med_flag_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    for col in med_flag_cols:
        if col not in df.columns:
            df[col] = 0
    df["n_medications"] = df["n_medications"].fillna(0).astype(int)
    df["meds_unknown"] = df["meds_unknown"].fillna(1).astype(int)
    for flag in MED_CATEGORY_KEYWORDS:
        df[flag] = df[flag].fillna(0).astype(int)

    # ── Print distributions ──
    print(f"\n  Diagnosis Category Distribution (grouped, 13 classes):")
    for cat in df["diagnosis_group"].value_counts().index:
        count = (df["diagnosis_group"] == cat).sum()
        print(f"    {cat:45s} {count:>7,}  ({100 * count / len(df):5.1f}%)")

    print(f"\n  Vital signs availability:")
    for v in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]:
        present = (df[f"{v}_missing"] == 0).sum()
        print(f"    {v:15s}  present: {present:,}/{len(df):,} ({100*present/len(df):.1f}%)")

    has_meds = (df["meds_unknown"] == 0).sum()
    print(f"\n  Medication data: {has_meds:,}/{len(df):,} ({100*has_meds/len(df):.1f}%)")
    print(f"  Mean medications per patient (where known): "
          f"{df.loc[df['meds_unknown']==0, 'n_medications'].mean():.1f}")

    return df


# ---------------------------------------------------------------------------
# 2. Build features
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix: triage + triage preds + snapshot vitals + meds
    + longitudinal vitals + rhythm (Phase B).
    """
    print("\n" + "=" * 60)
    print("STEP 2: Feature engineering (triage + snapshot vitals + meds + longitudinal vitals + rhythm)")
    print("=" * 60)

    df = df.copy()

    # ── Load triage artifacts ──
    tfidf = joblib.load(TRIAGE_MODELS_DIR / "tfidf_vectorizer.joblib")
    severity_map = joblib.load(TRIAGE_MODELS_DIR / "severity_map.joblib")
    acuity_model = joblib.load(TRIAGE_MODELS_DIR / "acuity_model.joblib")
    disposition_model = joblib.load(TRIAGE_MODELS_DIR / "disposition_model.joblib")
    print("  Loaded triage artifacts")

    # ── Text features ──
    df["complaint_text"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    df["n_complaints"] = df["chiefcomplaint"].apply(
        lambda x: len([c.strip() for c in str(x).split(",") if c.strip()])
    )
    df["complaint_length"] = df["complaint_text"].apply(len)

    # ── TF-IDF ──
    tfidf_matrix = tfidf.transform(df["complaint_text"])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])],
        index=df.index,
    )
    print(f"  TF-IDF features: {tfidf_df.shape[1]}")

    # ── Severity priors ──
    def compute_severity_priors(text: str) -> tuple:
        words = text.split()
        severities = [severity_map[w] for w in words if w in severity_map]
        if severities:
            return (min(severities), np.mean(severities),
                    max(severities), np.std(severities) if len(severities) > 1 else 0.0)
        return 3.0, 3.0, 3.0, 0.0

    sev = df["complaint_text"].apply(compute_severity_priors)
    df["min_severity_prior"] = sev.apply(lambda x: x[0])
    df["mean_severity_prior"] = sev.apply(lambda x: x[1])
    df["max_severity_prior"] = sev.apply(lambda x: x[2])
    df["std_severity_prior"] = sev.apply(lambda x: x[3])

    # ── Age/pain bins ──
    df["age_bin"] = pd.cut(
        df["age"], bins=[0, 18, 35, 50, 65, 80, 120], labels=[0, 1, 2, 3, 4, 5],
    ).astype(float).fillna(2)
    df["pain_low"] = ((df["pain"] >= 0) & (df["pain"] <= 3)).astype(int)
    df["pain_mid"] = ((df["pain"] >= 4) & (df["pain"] <= 6)).astype(int)
    df["pain_high"] = ((df["pain"] >= 7) & (df["pain"] <= 10)).astype(int)

    # ── Interaction features ──
    df["age_ambulance"] = df["age"] * df["arrival_ambulance"]
    df["pain_x_min_severity"] = df["pain"].clip(0, 10) * (5 - df["min_severity_prior"])
    df["age_severity"] = df["age"] * (5 - df["min_severity_prior"])
    df["high_pain_ambulance"] = df["pain_high"] * df["arrival_ambulance"]
    df["elderly"] = (df["age"] >= 65).astype(int)
    df["elderly_ambulance"] = df["elderly"] * df["arrival_ambulance"]

    # ── Assemble triage feature vector (same order as triage pipeline) ──
    structured_cols = [
        "pain", "pain_missing", "pain_low", "pain_mid", "pain_high",
        "n_complaints", "complaint_length",
        "min_severity_prior", "mean_severity_prior",
        "max_severity_prior", "std_severity_prior",
        "age", "age_bin", "gender_male",
        "arrival_ambulance", "arrival_helicopter", "arrival_walk_in",
        "age_ambulance", "pain_x_min_severity", "age_severity",
        "high_pain_ambulance", "elderly", "elderly_ambulance",
    ]

    structured = df[structured_cols].reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    triage_features = pd.concat([structured, tfidf_df], axis=1)

    # ── Generate triage predictions (cascading) ──
    print("  Generating triage predictions...")
    predicted_acuity = acuity_model.predict(triage_features) + 1
    triage_features_disp = triage_features.copy()
    triage_features_disp["predicted_acuity"] = predicted_acuity
    predicted_disposition = disposition_model.predict(triage_features_disp)

    triage_features["predicted_acuity"] = predicted_acuity
    triage_features["predicted_disposition"] = predicted_disposition

    # ── Vital sign features (snapshot from triage.csv) ──
    vital_cols = [
        "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
        "temperature_missing", "heartrate_missing", "resprate_missing",
        "o2sat_missing", "sbp_missing", "dbp_missing",
        "fever", "tachycardia", "bradycardia", "tachypnea",
        "hypoxia", "hypertension", "hypotension", "map",
    ]
    vitals = df[vital_cols].reset_index(drop=True)
    print(f"  Vital sign features (snapshot): {len(vital_cols)}")

    # ── Medication features ──
    med_cols = ["n_medications", "meds_unknown"] + list(MED_CATEGORY_KEYWORDS.keys())
    meds = df[med_cols].reset_index(drop=True)
    print(f"  Medication features: {len(med_cols)}")

    # ── Phase B: longitudinal vitals + rhythm ──
    long_vitals = df[LONG_VITAL_FEATURE_COLS].reset_index(drop=True)
    print(f"  Longitudinal vital + rhythm features: {len(LONG_VITAL_FEATURE_COLS)}")

    # ── Assemble all ──
    features = pd.concat([triage_features, vitals, meds, long_vitals], axis=1)
    print(f"  Total features: {features.shape[1]}")

    return features


# ---------------------------------------------------------------------------
# 3-4. Train models
# ---------------------------------------------------------------------------
def train_model(
    X_train, y_train, X_test, y_test, label_names, model_name,
) -> XGBClassifier:
    """Train an XGBoost multiclass model with sqrt-inverse class weighting."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name} ({len(label_names)} classes, XGBoost)")
    print(f"{'='*60}")

    n_classes = len(label_names)
    class_counts = y_train.value_counts()
    total = len(y_train)
    sample_weights = y_train.map(
        lambda x: np.sqrt(total / (n_classes * class_counts[x]))
    )

    model = XGBClassifier(
        n_estimators=3000, max_depth=10, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.7,
        min_child_weight=3, gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", early_stopping_rounds=100,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    print(f"  Training (up to 3000 trees, lr=0.02, early stopping=100)...")
    model.fit(
        X_train, y_train, sample_weight=sample_weights,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    return model


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
def save_models(
    diagnosis_model, department_model,
    diagnosis_labels, department_labels,
    diagnosis_accuracy, department_accuracy,
    vital_medians,
    n_train, n_test,
):
    """Save v3 doctor (with-nurse) models and metadata."""
    print(f"\n{'='*60}")
    print("  Saving Doctor v3 with-nurse model artifacts")
    print(f"{'='*60}")

    DOCTOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(diagnosis_model, DOCTOR_MODELS_DIR / "diagnosis_model.joblib")
    joblib.dump(department_model, DOCTOR_MODELS_DIR / "department_model.joblib")

    metadata = {
        "version": "3",
        "trained_at": datetime.now().isoformat(),
        "diagnosis_labels": diagnosis_labels,
        "department_labels": department_labels,
        "department_names": DEPARTMENT_NAMES,
        "diagnosis_accuracy": round(diagnosis_accuracy, 4),
        "department_accuracy": round(department_accuracy, 4),
        "model_type": "XGBClassifier",
        "n_diagnosis_classes": len(diagnosis_labels),
        "n_department_classes": len(department_labels),
        "n_train": n_train,
        "n_test": n_test,
        "catch_all_excluded": CATCH_ALL_LABEL,
        "sub_sampled": False,
        "vital_medians": vital_medians,
        "med_category_keywords": {k: v for k, v in MED_CATEGORY_KEYWORDS.items()},
        "note": "Doctor v3 with nurse data. Catch-all excluded; full filtered "
                "dataset (no 100K sub-sample). Feature set: snapshot vitals "
                "from triage.csv + medrecon medication flags + longitudinal "
                "vitals (min/max/last/delta over [intime, intime+4h]) + "
                "abnormal-reading counts + cardiac rhythm (one-hot bucket + "
                "rhythm_irregular flag) from vitalsign.csv. The 4h window "
                "guards against late-stay disposition leakage.",
        "longitudinal_window_hours": _LONG_WINDOW_HOURS,
        "rhythm_buckets": _RHYTHM_BUCKETS,
        "long_vital_feature_cols": LONG_VITAL_FEATURE_COLS,
    }

    with open(DOCTOR_MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {DOCTOR_MODELS_DIR}")
    for fname in ["diagnosis_model.joblib", "department_model.joblib",
                   "metadata.json"]:
        print(f"  - {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "#" * 60)
    print("  Doctor Agent v3 — Model Training Pipeline (with nurse data)")
    print("  (Catch-all excluded, full dataset, snapshot + longitudinal vitals + rhythm)")
    print("#" * 60)

    # 1. Load (with catch-all filter applied inside)
    df = load_and_clean_data()

    # 2. v3: NO sub-sampling. Train on the full filtered set.
    print(f"\n{'='*60}")
    print(f"  Using full dataset (no sub-sampling) — {len(df):,} rows")
    print(f"{'='*60}")

    # 3. Build features
    features = build_features(df)

    # 4. Encode labels
    diagnosis_labels = sorted(df["diagnosis_group"].unique())
    department_labels = sorted(df["service_group"].unique())
    diag_map = {l: i for i, l in enumerate(diagnosis_labels)}
    dept_map = {l: i for i, l in enumerate(department_labels)}
    y_diag = df["diagnosis_group"].map(diag_map).reset_index(drop=True)
    y_dept = df["service_group"].map(dept_map).reset_index(drop=True)

    assert CATCH_ALL_LABEL not in diagnosis_labels, (
        f"BUG: catch-all label '{CATCH_ALL_LABEL}' is still present in v3 label space"
    )

    print(f"\n  Diagnosis classes ({len(diagnosis_labels)}): {diagnosis_labels}")
    print(f"  Department classes ({len(department_labels)}): {department_labels}")

    # 5. Split
    print(f"\n{'='*60}")
    print("  Train/test split (80/20, stratified)")
    print(f"{'='*60}")
    X_train, X_test, y_diag_train, y_diag_test, y_dept_train, y_dept_test = \
        train_test_split(features, y_diag, y_dept, test_size=0.2,
                         random_state=42, stratify=y_diag)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 6a. Train diagnosis model
    diag_model = train_model(
        X_train, y_diag_train, X_test, y_diag_test,
        diagnosis_labels, "DIAGNOSIS v3",
    )
    diag_acc = accuracy_score(y_diag_test, diag_model.predict(X_test))

    # 6b. Train department model (cascading)
    X_train_dept = X_train.copy()
    X_train_dept["predicted_diagnosis"] = diag_model.predict(X_train)
    X_test_dept = X_test.copy()
    X_test_dept["predicted_diagnosis"] = diag_model.predict(X_test)

    dept_model = train_model(
        X_train_dept, y_dept_train, X_test_dept, y_dept_test,
        department_labels, "DEPARTMENT v3",
    )
    dept_acc = accuracy_score(y_dept_test, dept_model.predict(X_test_dept))

    # 7. Save
    vital_medians = {"temperature": 98.1, "heartrate": 84, "resprate": 18,
                     "o2sat": 98, "sbp": 134, "dbp": 78}
    save_models(
        diag_model, dept_model, diagnosis_labels, department_labels,
        diag_acc, dept_acc, vital_medians,
        n_train=len(X_train), n_test=len(X_test),
    )

    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE! (v3 with nurse data, Phase A)")
    print(f"{'#'*60}")
    print(f"  Diagnosis v3 accuracy:  {diag_acc:.4f}  ({len(diagnosis_labels)} classes)")
    print(f"  Department v3 accuracy: {dept_acc:.4f}  ({len(department_labels)} classes)")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
