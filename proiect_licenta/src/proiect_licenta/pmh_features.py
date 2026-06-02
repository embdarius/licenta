"""Shared PMH (Past Medical History) feature aggregator.

Extracted from `training/train_nurse_v3.py` (Doctor v3 nurse Change 1) so any
training pipeline can reuse the same prior-encounter feature set without
depending on the doctor pipeline. Used by:

    training/train_nurse_v3.py     (Doctor v3 nurse)
    training/train_triage_v3.py    (Triage v3 — adds PMH on top of triage v2)

Both sides emit the same `PMH_FEATURE_COLS` so any model that consumes PMH
features sees the same column layout.

Provides:
    PMH_FEATURE_COLS    Deterministic column order for the 19-feature block.
    PMH_NO_PRIOR_DAYS   Sentinel value for "no prior visit" (so the model can
                        distinguish first-time patients from frequent flyers).
    aggregate_pmh(...)  Per-stay PMH feature builder. Reads:
                          - prior hospital admissions (admissions.csv)
                          - prior ED visits (edstays)
                          - prior ICD codes (diagnoses_icd.csv)
                          - prior discharge-note PMH sections (discharge.csv)
                        Emits 13 binary pmh_<group> flags + 6 visit numerics.

Leakage is guarded by `prior_admittime < current_intime` and
`prior_intime < current_intime`. Discharge notes are filtered to prior hadm_ids
only, so the current encounter's discharge summary can never leak in.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from proiect_licenta.paths import TRIAGE_CSV
from proiect_licenta.pmh_vocab import (
    PMH_CATEGORIES,
    flags_from_text as pmh_flags_from_text,
    extract_pmh_section,
)
from proiect_licenta.preprocessing import normalize_complaint_text

# Optional tqdm — matches train_nurse_v3.py behavior (auto in notebooks, plain
# bar in terminals, silent no-op when tqdm isn't installed).
try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore

    def tqdm(it, **kw):
        return _tqdm(it, **kw)
except ImportError:  # pragma: no cover

    def tqdm(it, **kw):  # type: ignore[misc]
        return it


# Sentinel for "no prior visit". Picked so the model can distinguish
# first-time patients (PMH_NO_PRIOR_DAYS) from frequent flyers (small days).
PMH_NO_PRIOR_DAYS = 9999

# Feature column order. Sorted PMH categories first (deterministic order
# matching PMH_CATEGORIES), then the repeat-visit numerics, then no_history.
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


def _build_icd_to_group(
    diagnosis_csv_path,
    diagnosis_group_map: dict,
) -> dict:
    """Return an ICD code → diagnosis_group lookup derived from the ED label CSV.

    Reuses the same ICD→category mapping that the doctor's supervision labels
    are built from. `diagnosis_group_map` maps the ED `category` strings to
    the 13 PMH/diagnosis groups; we apply it here so the ICD-derived PMH
    flags use the same label space as the discharge-note PMH flags.
    """
    diag = pd.read_csv(
        diagnosis_csv_path,
        usecols=["icd_code", "category"],
        dtype={"icd_code": str},
    )
    diag = diag.dropna(subset=["icd_code", "category"])
    diag["group"] = diag["category"].map(diagnosis_group_map).fillna("Other")
    # Mode protects against rare duplicate (icd_code, category) pairs.
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


def _parse_discharge_pmh(
    discharge_csv_path,
    needed_subjects: set,
) -> dict:
    """Chunked parse of discharge.csv -> {hadm_id: set(diagnosis_groups)}.

    Memory-cheap: discharge notes are large but we keep only the PMH-flag
    set per admission (a few bytes per row). Chunk size is small because
    each note can be 5-50 KB of text.
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

    pbar = tqdm(
        reader,
        total=170,
        desc="    discharge.csv",
        unit="chunk",
        leave=False,
    )
    for chunk in pbar:
        rows_seen += len(chunk)
        chunk = chunk[chunk["subject_id"].isin(needed_subjects)]
        chunk = chunk.dropna(subset=["hadm_id", "text"])
        rows_kept += len(chunk)
        for hid, text in zip(chunk["hadm_id"], chunk["text"]):
            section = extract_pmh_section(text)
            if not section:
                continue
            flags = pmh_flags_from_text(section)
            if flags:
                if int(hid) in note_pmh_by_hadm:
                    note_pmh_by_hadm[int(hid)] |= flags
                else:
                    note_pmh_by_hadm[int(hid)] = flags
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


def build_pmh_index(
    subjects: set,
    edstays_full: pd.DataFrame,
    diagnoses_icd_csv_path,
    admissions_csv_path,
    discharge_csv_path,
    diagnosis_csv_path,
    diagnosis_group_map: dict,
    catch_all_label: str | None = None,
    triage_csv_path=TRIAGE_CSV,
) -> dict:
    """Build the per-subject prior-encounter structures PMH features derive from.

    This is the heavy half of the PMH pipeline (it parses the 3.3 GB
    discharge.csv); the per-stay assembly is cheap by comparison. The index is
    built ONCE over a set of subject_ids and then queried per stay by
    ``assemble_pmh_for_stay``, which applies the ``< intime`` leakage filter
    using the *current* stay's intime. Persisting this index (joblib) and
    querying it at inference is how ``PatientHistoryLookupTool`` simulates an
    EHR lookup without re-parsing the source tables.

    Returns a dict with keys:
        adm_by_subject    {subject_id: sorted [(admittime, hadm_id), ...]}
        ed_by_subject     {subject_id: sorted [(intime, stay_id, complaint_norm), ...]}
        icd_pmh_by_hadm   {hadm_id: set(diagnosis_groups)}  (ICD-derived)
        note_pmh_by_hadm  {hadm_id: set(diagnosis_groups)}  (discharge-note-derived)
    """
    needed_subjects = set(int(s) for s in subjects)
    print(f"  Building PMH index over {len(needed_subjects):,} patients...")

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

    adm_by_subject: dict[int, list] = defaultdict(list)
    for sid, at, hid in zip(adm["subject_id"], adm["admittime"], adm["hadm_id"]):
        if pd.notna(at):
            adm_by_subject[int(sid)].append((at, int(hid)))
    for sid in adm_by_subject:
        adm_by_subject[sid].sort()

    # ── (2) Prior ED visits for the same patients ──
    print("    Indexing prior ED visits...")
    ed_subset = edstays_full[edstays_full["subject_id"].isin(needed_subjects)].copy()
    ed_subset["intime"] = pd.to_datetime(ed_subset["intime"], errors="coerce")
    triage_cc = pd.read_csv(triage_csv_path, usecols=["stay_id", "chiefcomplaint"])
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
    icd_to_group = _build_icd_to_group(diagnosis_csv_path, diagnosis_group_map)
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
    icd_df = (
        pd.concat(icd_chunks, ignore_index=True)
        if icd_chunks
        else pd.DataFrame(columns=["subject_id", "hadm_id", "icd_code"])
    )
    print(f"    diagnoses_icd.csv (filtered): {len(icd_df):,} rows")

    icd_pmh_by_hadm: dict[int, set] = defaultdict(set)
    for hid, code in zip(icd_df["hadm_id"], icd_df["icd_code"]):
        grp = icd_to_group.get(str(code))
        if grp and (catch_all_label is None or grp != catch_all_label):
            icd_pmh_by_hadm[int(hid)].add(grp)

    # ── (4) Discharge-note PMH per hadm_id (richer source) ──
    print(f"    Parsing discharge.csv ({discharge_csv_path}) — chunked...")
    if not Path(discharge_csv_path).exists():
        print(f"    WARNING: {discharge_csv_path} not found — skipping discharge PMH.")
        note_pmh_by_hadm: dict[int, set] = {}
    else:
        note_pmh_by_hadm = _parse_discharge_pmh(discharge_csv_path, needed_subjects)
    print(f"    Discharge-note PMH parsed for {len(note_pmh_by_hadm):,} admissions")

    return {
        "adm_by_subject": adm_by_subject,
        "ed_by_subject": ed_by_subject,
        "icd_pmh_by_hadm": icd_pmh_by_hadm,
        "note_pmh_by_hadm": note_pmh_by_hadm,
    }


def assemble_pmh_for_stay(
    subject_id: int,
    intime,
    complaint_norm: str,
    index: dict,
) -> dict:
    """Derive the 19-column PMH feature record for ONE stay from a prebuilt
    ``index`` (see ``build_pmh_index``), applying the strict
    ``prior_*time < intime`` leakage filter.

    ``intime`` may be a pandas Timestamp / datetime or an ISO string. Returns a
    dict keyed by PMH_FEATURE_COLS (no stay_id). First-time patients (no prior
    admission or ED visit before ``intime``) get the all-zero no_history=1
    pattern — identical to the runtime ask-the-patient fallback, so known and
    unknown patients split exactly the way the model was trained to expect.
    """
    intime = pd.to_datetime(intime)
    sid = int(subject_id)

    adm_by_subject = index["adm_by_subject"]
    ed_by_subject = index["ed_by_subject"]
    icd_pmh_by_hadm = index["icd_pmh_by_hadm"]
    note_pmh_by_hadm = index["note_pmh_by_hadm"]

    prior_adm = [(at, hid) for at, hid in adm_by_subject.get(sid, [])
                 if at < intime]
    prior_ed = [(it, _, cc) for it, _, cc in ed_by_subject.get(sid, [])
                if it < intime]

    # Leakage guard: nothing at/after the current intime may survive the filter.
    # The list comprehensions already enforce this; the assertions make the
    # invariant explicit so a future change to the filter can't silently leak.
    assert all(at < intime for at, _ in prior_adm), "PMH leak: prior admission >= intime"
    assert all(it < intime for it, _, _ in prior_ed), "PMH leak: prior ED visit >= intime"

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

    flags: set = set()
    for _, hid in prior_adm:
        flags |= note_pmh_by_hadm.get(hid, set())
        flags |= icd_pmh_by_hadm.get(hid, set())

    rec = {}
    for cat in PMH_CATEGORIES:
        rec[f"pmh_{cat}"] = 1 if cat in flags else 0
    rec["n_prior_admissions"] = n_prior_adm
    rec["n_prior_ed_visits"] = n_prior_ed
    rec["days_since_last_admission"] = days_last_adm
    rec["days_since_last_ed"] = days_last_ed
    rec["same_complaint_as_prior"] = round(same_complaint, 4)
    rec["no_history"] = int(n_prior_adm == 0 and n_prior_ed == 0)
    return rec


def aggregate_pmh(
    stays_df: pd.DataFrame,
    edstays_full: pd.DataFrame,
    diagnoses_icd_csv_path,
    admissions_csv_path,
    discharge_csv_path,
    diagnosis_csv_path,
    diagnosis_group_map: dict,
    catch_all_label: str | None = None,
    triage_csv_path=TRIAGE_CSV,
) -> pd.DataFrame:
    """Build PMH feature columns for every stay in ``stays_df``.

    Parameters
    ----------
    stays_df : DataFrame with columns ["stay_id", "subject_id", "intime",
        "complaint_text_norm"]. `intime` must be datetime64.
        `complaint_text_norm` is the normalize_complaint_text output for the
        current stay (used for same_complaint_as_prior).
    edstays_full : Full edstays DataFrame (every ED visit for every patient,
        not just admitted). Used to count n_prior_ed_visits and to find the
        most recent prior chief complaint.
    diagnoses_icd_csv_path : Path to mimic-iv/hosp/diagnoses_icd.csv.
    admissions_csv_path : Path to mimic-iv/hosp/admissions.csv (for admittime
        which gates the prior-admission temporal filter).
    discharge_csv_path : Path to mimic-iv-notes/.../discharge.csv (3.3 GB).
        Read in chunks; only the subject_ids in `stays_df` are kept.
    diagnosis_csv_path : Path to categorized_diagnosis.csv (for the
        ICD→category lookup table).
    diagnosis_group_map : Maps the ED `category` strings to the 13 PMH
        diagnosis groups (same map used to build supervision labels in the
        doctor pipeline).
    catch_all_label : Optional. If provided, ICD codes that map to this label
        are dropped from the PMH flag set (since the catch-all is a coding
        artifact, not a real chronic condition). In v3 this is the
        "Symptoms, Signs, Ill-Defined" group.
    triage_csv_path : Path to triage.csv (defaults to project-level constant).
        Used to look up the chief complaint of each prior ED visit for the
        same-complaint Jaccard feature.

    Returns
    -------
    DataFrame keyed on stay_id with the PMH_FEATURE_COLS columns.
    """
    print("  Building PMH features (this is the heaviest step)...")
    needed_subjects = set(stays_df["subject_id"].astype(int).tolist())
    print(f"    Target stays: {len(stays_df):,} across "
          f"{len(needed_subjects):,} patients")

    index = build_pmh_index(
        subjects=needed_subjects,
        edstays_full=edstays_full,
        diagnoses_icd_csv_path=diagnoses_icd_csv_path,
        admissions_csv_path=admissions_csv_path,
        discharge_csv_path=discharge_csv_path,
        diagnosis_csv_path=diagnosis_csv_path,
        diagnosis_group_map=diagnosis_group_map,
        catch_all_label=catch_all_label,
        triage_csv_path=triage_csv_path,
    )

    # ── Per-stay assembly (cheap; queries the prebuilt index) ──
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
        rec = {"stay_id": stay_id,
               **assemble_pmh_for_stay(sid, intime, complaint_norm, index)}
        out_records.append(rec)

    result = pd.DataFrame(out_records)
    flag_cols = [c for c in result.columns if c.startswith("pmh_")]
    any_flag = (result[flag_cols].sum(axis=1) > 0).mean()
    print(f"    Per-stay PMH coverage: {100*any_flag:.1f}% have ≥1 PMH flag, "
          f"{100*(1-result['no_history'].mean()):.1f}% have ≥1 prior visit")
    return result


def fill_missing_pmh_columns(df: pd.DataFrame) -> None:
    """Ensure every PMH_FEATURE_COLS column exists and has sensible defaults.

    Stays that didn't appear in the aggregate_pmh result (shouldn't happen on
    a clean merge, but guard) get the no_history fallback — identical to the
    inference-time zero-fill for first-time patients.

    Mutates ``df`` in place.
    """
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
