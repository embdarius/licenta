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
import re
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
# Optional progress bars (tqdm.auto picks Jupyter widgets in Colab, plain bar
# in a terminal). If tqdm isn't installed, fall back to a no-op wrapper so
# local `uv run train_nurse_v3` still works without an extra dependency.
# ---------------------------------------------------------------------------
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ImportError:  # pragma: no cover

    def tqdm(it, **kw):  # type: ignore[misc]
        return it


def _make_xgb_progress_callback(n_total: int, desc: str):
    """Return an XGBoost TrainingCallback that updates a tqdm bar per iteration.

    Falls back to ``None`` if tqdm or the XGBoost callback API isn't available
    (older XGBoost versions), in which case .fit() runs silently as before.
    """
    try:
        from tqdm.auto import tqdm as _tqdm_cls
        from xgboost.callback import TrainingCallback
    except ImportError:
        return None

    class _TqdmCallback(TrainingCallback):  # type: ignore[misc]
        def __init__(self):
            self.pbar = _tqdm_cls(
                total=n_total, desc=desc, unit="iter", leave=False,
            )

        def after_iteration(self, model, epoch, evals_log):
            self.pbar.update(1)
            # evals_log structure: {'validation_0': {'mlogloss': [v1, v2, ...]}}
            last_val = None
            for ds in evals_log:
                for metric in evals_log[ds]:
                    vals = evals_log[ds][metric]
                    if vals:
                        last_val = vals[-1]
            if last_val is not None and hasattr(self.pbar, "set_postfix"):
                self.pbar.set_postfix(mlogloss=f"{last_val:.4f}", refresh=False)
            return False  # do not stop training

        def after_training(self, model):
            self.pbar.close()
            return model

    return _TqdmCallback()

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
    DISCHARGE_NOTES_CSV, DIAGNOSES_ICD_CSV, ADMISSIONS_CSV,
    DERIVED_DIR,
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
from proiect_licenta.pmh_vocab import (
    PMH_CATEGORIES, flags_from_text as pmh_flags_from_text,
    extract_pmh_section,
)

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
    # vitalsign.csv has ~1.4M rows; at chunksize=1M that's ~2 chunks.
    chunks = []
    vs_reader = pd.read_csv(
        vitalsign_csv_path,
        usecols=["stay_id", "charttime", "temperature", "heartrate",
                 "resprate", "o2sat", "sbp", "dbp", "rhythm"],
        chunksize=1_000_000,
    )
    for chunk in tqdm(
        vs_reader,
        total=2,
        desc="    vitalsign.csv",
        unit="chunk",
        leave=False,
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
    for stay_id, sub in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="    long-vitals aggregate",
        unit="stay",
        leave=False,
    ):
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
# PMH aggregation — Change 1 (prior discharge-note PMH + prior ICD fallback)
# ---------------------------------------------------------------------------
# For each stay we emit:
#   - 13 binary pmh_<group> flags (one per v3 diagnosis class, catch-all
#     excluded; see PMH_CATEGORIES in pmh_vocab.py).
#   - n_prior_admissions, n_prior_ed_visits.
#   - days_since_last_admission, days_since_last_ed (capped at PMH_NO_PRIOR_DAYS
#     when there is no prior record).
#   - same_complaint_as_prior (Jaccard on tokenized chief complaint vs the
#     patient's most recent prior ED visit).
#   - no_history (1 if the patient has no prior hospital admission AND no
#     prior ED visit; matches the inference-time zero-fill fallback).
#
# Two PMH sources combined with OR:
#   (a) MIMIC-IV-Note `discharge.csv` — parse the "Past Medical History"
#       section of each prior discharge summary for the same subject_id.
#       Map keywords -> diagnosis groups via pmh_vocab.PMH_KEYWORD_MAP.
#   (b) MIMIC-IV-Hosp `diagnoses_icd.csv` — for prior admissions only,
#       look up each ICD code in an ICD→category map derived from
#       `categorized_diagnosis.csv` (the same ED label source used for
#       supervision). Falls through when the discharge note is missing or
#       the PMH section was empty.
#
# Leakage: zero by construction. Both sources are filtered to
# prior_admittime < current_intime, so we only ever see history that pre-
# dates the encounter we're predicting.

# Sentinel for "no prior visit". Picked so the model can distinguish
# first-time patients (PMH_NO_PRIOR_DAYS) from frequent flyers (small days).
PMH_NO_PRIOR_DAYS = 9999

# Feature column order. Exposed so doctor_tool_v3 can reproduce the same
# layout at inference. Sorted PMH categories first (deterministic order
# matching PMH_CATEGORIES), then the repeat-visit numerics, then the flags.
PMH_FEATURE_COLS = (
    [f"pmh_{c}" for c in PMH_CATEGORIES]
    + [
        "n_prior_admissions",
        "n_prior_ed_visits",
        "days_since_last_admission",
        "days_since_last_ed",
        "same_complaint_as_prior",
        "no_history",
    ]
)


def _build_icd_to_group(diagnosis_csv_path) -> dict:
    """Return an ICD code → diagnosis_group lookup derived from the ED label CSV.

    We reuse the same ICD→category mapping that the supervision labels are
    built from (categorized_diagnosis.csv → DIAGNOSIS_GROUP_MAP). Covers
    only ICD codes that appear in the ED. Hospital ICDs unique to inpatient
    encounters will fall through to "no flag" — that's acceptable because
    the discharge-note PMH path catches most of those anyway.
    """
    diag = pd.read_csv(
        diagnosis_csv_path,
        usecols=["icd_code", "category"],
        dtype={"icd_code": str},
    )
    diag = diag.dropna(subset=["icd_code", "category"])
    diag["group"] = diag["category"].map(DIAGNOSIS_GROUP_MAP).fillna("Other")
    # An ICD code may appear with multiple categories across rows; take the
    # mode (most common). In practice the mapping is ~1:1, but mode protects
    # against rare duplicates from typos / coding-system variants.
    mode = (
        diag.groupby("icd_code")["group"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else "Other")
        .to_dict()
    )
    return mode


def _normalize_complaint_tokens(text: str) -> set:
    """Tokenize a normalized chief-complaint string into a token set."""
    if not isinstance(text, str) or not text:
        return set()
    return {tok for tok in re.findall(r"[a-z]+", text.lower()) if len(tok) > 2}


def _aggregate_pmh(
    stays_df: pd.DataFrame,
    edstays_full: pd.DataFrame,
    diagnoses_icd_csv_path,
    admissions_csv_path,
    discharge_csv_path,
) -> pd.DataFrame:
    """Build PMH feature columns for every stay in `stays_df`.

    Parameters
    ----------
    stays_df : DataFrame with columns ["stay_id", "subject_id", "intime",
        "complaint_text_norm"]. `intime` must be datetime64. `complaint_text_norm`
        is the normalize_complaint_text output for the current stay (used for
        same_complaint_as_prior).
    edstays_full : Full edstays DataFrame (every ED visit for every patient,
        not just admitted). Used to count n_prior_ed_visits and to find the
        most recent prior chief complaint.
    diagnoses_icd_csv_path : Path to mimic-iv/hosp/diagnoses_icd.csv
    admissions_csv_path : Path to mimic-iv/hosp/admissions.csv (for admittime
        which gates the prior-admission temporal filter).
    discharge_csv_path : Path to mimic-iv-notes/.../discharge.csv (3.3 GB).
        Read in chunks; only the subject_ids in `stays_df` are kept.

    Returns
    -------
    DataFrame keyed on stay_id with the PMH_FEATURE_COLS columns.
    """
    print("  Building PMH features (this is the heaviest step)...")
    needed_subjects = set(stays_df["subject_id"].astype(int).tolist())
    print(f"    Target stays: {len(stays_df):,} across {len(needed_subjects):,} patients")

    # ── (1) Prior hospital admissions (subject_id, hadm_id, admittime) ──
    print("    Loading admissions.csv...")
    adm = pd.read_csv(
        admissions_csv_path,
        usecols=["subject_id", "hadm_id", "admittime"],
    )
    adm = adm[adm["subject_id"].isin(needed_subjects)].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    print(f"    admissions.csv (filtered): {len(adm):,} rows "
          f"across {adm['subject_id'].nunique():,} subjects")

    # subject_id -> list[(admittime, hadm_id)] sorted by admittime
    adm_by_subject: dict[int, list] = defaultdict(list)
    for sid, at, hid in zip(adm["subject_id"], adm["admittime"], adm["hadm_id"]):
        if pd.notna(at):
            adm_by_subject[int(sid)].append((at, int(hid)))
    for sid in adm_by_subject:
        adm_by_subject[sid].sort()

    # ── (2) Prior ED visits for the same patients (for n_prior_ed_visits and
    #        same_complaint_as_prior). Build subject -> sorted list of
    #        (intime, stay_id, chiefcomplaint_norm). ──
    print("    Indexing prior ED visits...")
    ed_subset = edstays_full[edstays_full["subject_id"].isin(needed_subjects)].copy()
    # `intime` from edstays_full may already be datetime; ensure it is.
    ed_subset["intime"] = pd.to_datetime(ed_subset["intime"], errors="coerce")
    # Bring chiefcomplaint from triage.csv via a lookup table. The training
    # pipeline already merged triage into `stays_df` (full chief complaints),
    # but we need complaints for *every* ED visit, including ones that
    # weren't admitted. Re-merge from triage.csv.
    triage_cc = pd.read_csv(TRIAGE_CSV, usecols=["stay_id", "chiefcomplaint"])
    ed_subset = ed_subset.merge(triage_cc, on="stay_id", how="left")
    ed_subset["complaint_norm"] = ed_subset["chiefcomplaint"].fillna("").apply(
        normalize_complaint_text
    )

    ed_by_subject: dict[int, list] = defaultdict(list)
    for sid, it, stid, cc in zip(
        ed_subset["subject_id"], ed_subset["intime"],
        ed_subset["stay_id"], ed_subset["complaint_norm"],
    ):
        if pd.notna(it):
            ed_by_subject[int(sid)].append((it, int(stid), cc))
    for sid in ed_by_subject:
        ed_by_subject[sid].sort()

    # ── (3) ICD-derived PMH flags per hadm_id (fallback / supplement) ──
    print("    Building ICD->diagnosis_group map from categorized_diagnosis.csv...")
    icd_to_group = _build_icd_to_group(DIAGNOSIS_CSV)
    print(f"    ICD codes mapped: {len(icd_to_group):,}")

    print("    Loading diagnoses_icd.csv (181 MB)...")
    icd_chunks = []
    for chunk in pd.read_csv(
        diagnoses_icd_csv_path,
        usecols=["subject_id", "hadm_id", "icd_code"],
        dtype={"icd_code": str},
        chunksize=2_000_000,
    ):
        chunk = chunk[chunk["subject_id"].isin(needed_subjects)]
        if len(chunk):
            icd_chunks.append(chunk)
    icd_df = pd.concat(icd_chunks, ignore_index=True) if icd_chunks \
        else pd.DataFrame(columns=["subject_id", "hadm_id", "icd_code"])
    print(f"    diagnoses_icd.csv (filtered): {len(icd_df):,} rows")

    # hadm_id -> set of diagnosis_groups
    icd_pmh_by_hadm: dict[int, set] = defaultdict(set)
    for hid, code in zip(icd_df["hadm_id"], icd_df["icd_code"]):
        grp = icd_to_group.get(str(code))
        if grp and grp != CATCH_ALL_LABEL:
            icd_pmh_by_hadm[int(hid)].add(grp)

    # ── (4) Discharge-note PMH per hadm_id (richer source) ──
    print(f"    Parsing discharge.csv ({discharge_csv_path}) — chunked...")
    if not Path(discharge_csv_path).exists():
        print(f"    WARNING: {discharge_csv_path} not found — skipping discharge PMH.")
        note_pmh_by_hadm: dict[int, set] = {}
    else:
        note_pmh_by_hadm = _parse_discharge_pmh(discharge_csv_path, needed_subjects)
    print(f"    Discharge-note PMH parsed for {len(note_pmh_by_hadm):,} admissions")

    # ── (5) Per-stay assembly ──
    print("    Assembling per-stay PMH features...")
    out_records = []
    _iter = zip(
        stays_df["subject_id"].astype(int),
        stays_df["stay_id"].astype(int),
        stays_df["intime"],
        stays_df["complaint_text_norm"],
    )
    for sid, stay_id, intime, complaint_norm in tqdm(
        _iter,
        total=len(stays_df),
        desc="    PMH assembly",
        unit="stay",
        leave=False,
    ):
        # Prior admissions for this patient (admittime strictly before intime).
        prior_adm = [(at, hid) for at, hid in adm_by_subject.get(sid, [])
                     if at < intime]
        # Prior ED visits (intime strictly before current intime).
        prior_ed = [(it, _, cc) for it, _, cc in ed_by_subject.get(sid, [])
                    if it < intime]

        n_prior_adm = len(prior_adm)
        n_prior_ed = len(prior_ed)

        if prior_adm:
            days_last_adm = max(
                0.0,
                (intime - prior_adm[-1][0]).total_seconds() / 86400.0,
            )
            days_last_adm = float(min(days_last_adm, PMH_NO_PRIOR_DAYS))
        else:
            days_last_adm = float(PMH_NO_PRIOR_DAYS)

        if prior_ed:
            days_last_ed = max(
                0.0,
                (intime - prior_ed[-1][0]).total_seconds() / 86400.0,
            )
            days_last_ed = float(min(days_last_ed, PMH_NO_PRIOR_DAYS))
        else:
            days_last_ed = float(PMH_NO_PRIOR_DAYS)

        # Same-complaint Jaccard vs most recent prior ED visit.
        if prior_ed and complaint_norm:
            cur_tokens = _normalize_complaint_tokens(complaint_norm)
            prev_tokens = _normalize_complaint_tokens(prior_ed[-1][2])
            if cur_tokens and prev_tokens:
                inter = len(cur_tokens & prev_tokens)
                union = len(cur_tokens | prev_tokens)
                same_complaint = inter / union if union else 0.0
            else:
                same_complaint = 0.0
        else:
            same_complaint = 0.0

        # Union of discharge-note PMH and ICD-derived PMH across all prior
        # admissions for this patient.
        flags: set = set()
        for _, hid in prior_adm:
            flags |= note_pmh_by_hadm.get(hid, set())
            flags |= icd_pmh_by_hadm.get(hid, set())

        rec = {"stay_id": stay_id}
        for cat in PMH_CATEGORIES:
            rec[f"pmh_{cat}"] = 1 if cat in flags else 0
        rec["n_prior_admissions"] = n_prior_adm
        rec["n_prior_ed_visits"] = n_prior_ed
        rec["days_since_last_admission"] = days_last_adm
        rec["days_since_last_ed"] = days_last_ed
        rec["same_complaint_as_prior"] = round(same_complaint, 4)
        rec["no_history"] = int(n_prior_adm == 0 and n_prior_ed == 0)
        out_records.append(rec)

    result = pd.DataFrame(out_records)
    # Coverage stats — how many stays got any PMH flag at all?
    flag_cols = [c for c in result.columns if c.startswith("pmh_")]
    any_flag = (result[flag_cols].sum(axis=1) > 0).mean()
    print(f"    Per-stay PMH coverage: {100*any_flag:.1f}% have ≥1 PMH flag, "
          f"{100*(1-result['no_history'].mean()):.1f}% have ≥1 prior visit")
    return result


def _parse_discharge_pmh(
    discharge_csv_path,
    needed_subjects: set,
) -> dict:
    """Chunked parse of discharge.csv -> {hadm_id: set(diagnosis_groups)}.

    Memory-cheap: discharge notes are large but we keep only the
    PMH-flag set per admission (a few bytes per row). Chunk size is small
    because each note can be 5-50 KB of text.
    """
    note_pmh_by_hadm: dict[int, set] = {}
    rows_seen = 0
    rows_kept = 0
    try:
        reader = pd.read_csv(
            discharge_csv_path,
            usecols=["subject_id", "hadm_id", "text"],
            chunksize=2000,
        )
    except FileNotFoundError:
        print("    discharge.csv not found — discharge-note PMH skipped.")
        return note_pmh_by_hadm

    # discharge.csv has ~331K rows; at chunksize=2000 that's ~166 chunks.
    # `total` is approximate — used only to size the progress bar; tqdm will
    # gracefully overrun if the count is off.
    pbar = tqdm(
        reader,
        total=170,
        desc="    discharge.csv",
        unit="chunk",
        leave=False,
    )
    for i, chunk in enumerate(pbar):
        rows_seen += len(chunk)
        # Filter to patients we care about; drop rows without hadm_id (rare).
        chunk = chunk[chunk["subject_id"].isin(needed_subjects)]
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        rows_kept += len(chunk)
        for hid, text in zip(chunk["hadm_id"], chunk["text"]):
            section = extract_pmh_section(text)
            if not section:
                continue
            flags = pmh_flags_from_text(section)
            if flags:
                # Multiple notes for the same hadm_id (rare): union them.
                if int(hid) in note_pmh_by_hadm:
                    note_pmh_by_hadm[int(hid)] |= flags
                else:
                    note_pmh_by_hadm[int(hid)] = flags
        # Live counters on the bar so the user sees PMH coverage growing.
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                rows=f"{rows_seen:,}",
                kept=f"{rows_kept:,}",
                pmh=f"{len(note_pmh_by_hadm):,}",
                refresh=False,
            )
    print(f"    Discharge parse done: {rows_seen:,} rows seen, {rows_kept:,} kept, "
          f"{len(note_pmh_by_hadm):,} admissions with PMH flags.")
    return note_pmh_by_hadm


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

    # Load edstays (full file — we need every ED visit per patient for the
    # PMH features' n_prior_ed_visits / same_complaint_as_prior lookups, not
    # just admitted stays).
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

    # ── Change 1: PMH features from prior discharge notes + ICDs ──
    print("\n  Aggregating PMH features from prior encounters...")
    # The PMH step needs the normalized complaint text on the current row
    # (for same_complaint_as_prior). Compute it here once so we don't redo
    # it inside build_features for the merge key.
    df["complaint_text_norm"] = df["chiefcomplaint"].apply(normalize_complaint_text)
    pmh_df = _aggregate_pmh(
        stays_df=df[["stay_id", "subject_id", "intime", "complaint_text_norm"]],
        edstays_full=edstays,
        diagnoses_icd_csv_path=DIAGNOSES_ICD_CSV,
        admissions_csv_path=ADMISSIONS_CSV,
        discharge_csv_path=DISCHARGE_NOTES_CSV,
    )
    df = df.merge(pmh_df, on="stay_id", how="left")
    # Stays that didn't appear in pmh_df (shouldn't happen, but guard) get
    # the no_history fallback — identical to the inference-time zero-fill.
    for col in PMH_FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df["no_history"] = df["no_history"].fillna(1).astype(int)
    for cat in PMH_CATEGORIES:
        df[f"pmh_{cat}"] = df[f"pmh_{cat}"].fillna(0).astype(int)
    df["n_prior_admissions"] = df["n_prior_admissions"].fillna(0).astype(int)
    df["n_prior_ed_visits"] = df["n_prior_ed_visits"].fillna(0).astype(int)
    df["days_since_last_admission"] = df["days_since_last_admission"].fillna(
        PMH_NO_PRIOR_DAYS
    ).astype(float)
    df["days_since_last_ed"] = df["days_since_last_ed"].fillna(
        PMH_NO_PRIOR_DAYS
    ).astype(float)
    df["same_complaint_as_prior"] = df["same_complaint_as_prior"].fillna(0.0).astype(float)

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

    # ── Change 1: PMH features (prior discharge notes + prior ICDs) ──
    pmh = df[PMH_FEATURE_COLS].reset_index(drop=True)
    print(f"  PMH features: {len(PMH_FEATURE_COLS)}")

    # ── Assemble all ──
    features = pd.concat([triage_features, vitals, meds, long_vitals, pmh], axis=1)
    print(f"  Total features: {features.shape[1]}")

    return features


# ---------------------------------------------------------------------------
# A3 — diagnosis softmax cascade helpers
# ---------------------------------------------------------------------------
# Replace the legacy single `predicted_diagnosis` int column with 13
# `diag_proba_<sanitized_label>` float columns (the full softmax over
# diagnosis classes). The department model can then weight ambiguous
# predictions ("38% Circulatory, 32% Respiratory") instead of hard-locking
# to the argmax. Shared between training (where we fit) and inference
# (where doctor_tool_v3 reproduces the same column layout from the diag
# model's predict_proba output — XGBoost requires identical column names
# in identical order between fit and predict).
import re as _re


def _sanitize_label(label: str) -> str:
    """Convert a diagnosis-group label to a column-name-safe slug.

    "Endocrine, Nutritional, Metabolic" -> "endocrine_nutritional_metabolic"
    "Nervous System and Sense Organs"   -> "nervous_system_and_sense_organs"
    """
    slug = _re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
    return slug


def build_diag_cascade_cols(diagnosis_labels) -> list:
    """Return the canonical list of diagnosis-softmax cascade column names.

    Order matches `diagnosis_labels` so the i-th column corresponds to the
    i-th class in the diagnosis model's predict_proba output.
    """
    return [f"diag_proba_{_sanitize_label(l)}" for l in diagnosis_labels]


def _attach_diag_cascade(X, diag_model, diag_cascade_cols):
    """Return X augmented with the diagnosis-softmax cascade columns.

    Uses `predict_proba` then writes 13 float columns into a copy of X.
    The copy is necessary because we don't want to mutate the caller's
    feature matrix (the diagnosis model reuses it on a re-fit).
    """
    proba = diag_model.predict_proba(X)
    if proba.shape[1] != len(diag_cascade_cols):
        raise ValueError(
            f"Diagnosis model produced {proba.shape[1]} probability columns "
            f"but {len(diag_cascade_cols)} cascade cols were expected. "
            f"This usually means the model was trained on a different label "
            f"space than diagnosis_labels."
        )
    Xd = X.copy()
    for k, col in enumerate(diag_cascade_cols):
        Xd[col] = proba[:, k]
    return Xd


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

    # Optional tqdm progress bar — one tick per boosting iteration up to
    # 3000 (early stopping typically lands at 1700-2300). The callback is
    # None when tqdm isn't installed or XGBoost's callback API is unavailable,
    # in which case .fit() runs silently as before.
    progress_cb = _make_xgb_progress_callback(
        n_total=3000, desc=f"    XGBoost ({model_name})",
    )
    callbacks = [progress_cb] if progress_cb is not None else None

    model = XGBClassifier(
        n_estimators=3000, max_depth=10, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.5, colsample_bylevel=0.7,
        min_child_weight=3, gamma=0.05, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", early_stopping_rounds=100,
        random_state=42, n_jobs=-1, verbosity=0,
        callbacks=callbacks,
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
    diag_cascade_cols=None,
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
        "pmh_categories": PMH_CATEGORIES,
        "pmh_feature_cols": PMH_FEATURE_COLS,
        "pmh_no_prior_days": PMH_NO_PRIOR_DAYS,
        # A3: department model now cascades the diagnosis softmax (13 cols).
        # The inference tool reads this list to rebuild the exact same vector.
        "diag_cascade_cols": diag_cascade_cols,
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
    # A3 — cascade the diagnosis SOFTMAX (13 probability columns) into the
    # department model instead of the single argmax integer. The dept model
    # can then weight "this complaint is 38% Circulatory, 32% Respiratory"
    # honestly instead of hard-locking to the top-1 diagnosis. Column names
    # are `diag_proba_<sanitized_label>` so the inference tool can rebuild
    # the same vector. Persisted in metadata.json as `diag_cascade_cols`.
    diag_cascade_cols = build_diag_cascade_cols(diagnosis_labels)
    X_train_dept = _attach_diag_cascade(X_train, diag_model, diag_cascade_cols)
    X_test_dept = _attach_diag_cascade(X_test, diag_model, diag_cascade_cols)

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
        diag_cascade_cols=diag_cascade_cols,
    )

    print(f"\n{'#'*60}")
    print("  TRAINING COMPLETE! (v3 with nurse data, Phase A)")
    print(f"{'#'*60}")
    print(f"  Diagnosis v3 accuracy:  {diag_acc:.4f}  ({len(diagnosis_labels)} classes)")
    print(f"  Department v3 accuracy: {dept_acc:.4f}  ({len(department_labels)} classes)")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
