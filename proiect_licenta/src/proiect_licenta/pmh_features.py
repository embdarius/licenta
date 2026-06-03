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

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from proiect_licenta.paths import TRIAGE_CSV, MEDRECON_CSV
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


def parse_pmh_lookup(pmh_lookup_json: str):
    """Parse a ``PatientHistoryLookupTool`` ``pmh_block`` into a full
    ``PMH_FEATURE_COLS`` dict, or return ``None`` to fall back to the
    ask-the-patient text path.

    Accepts either the raw ``pmh_block`` object or the tool's full output (with a
    nested ``pmh_block``). Returns ``None`` on empty/invalid input or a block
    missing any required column — so a malformed lookup never silently corrupts
    the feature vector; it just reverts to self-report.

    Single source of truth shared by the triage tool and the doctor disposition
    / v3 reassessment tools, all of which override self-reported PMH with the
    real prior-encounter record when a returning patient is found.
    """
    if not pmh_lookup_json or not str(pmh_lookup_json).strip():
        return None
    try:
        obj = json.loads(pmh_lookup_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and "pmh_block" in obj:
        obj = obj["pmh_block"]
    if not isinstance(obj, dict) or not all(col in obj for col in PMH_FEATURE_COLS):
        return None
    block = {}
    for col in PMH_FEATURE_COLS:
        v = obj[col]
        if col in ("days_since_last_admission", "days_since_last_ed",
                   "same_complaint_as_prior"):
            block[col] = float(v)
        else:
            block[col] = int(v)
    return block


def pmh_self_report_discrepancy(prior_history: str, lookup_block: dict) -> list:
    """Read-only reconciliation: PMH categories the patient self-reported that
    the real prior-encounter record did NOT contain.

    When an EHR lookup overrides the patient's free-text PMH (the record wins,
    by design), this surfaces what the patient mentioned that isn't in the chart
    — e.g. a condition managed outside this hospital system, or a new diagnosis
    not yet coded. Purely informational: it does NOT alter the feature vector.

    Returns a sorted list of PMH category names present in the self-report but
    absent from ``lookup_block``'s fired flags. Empty when they agree, when no
    free-text was given, or when ``lookup_block`` is falsy.
    """
    if not prior_history or not str(prior_history).strip() or not lookup_block:
        return []
    self_reported = pmh_flags_from_text(str(prior_history))
    if not self_reported:
        return []
    in_record = {c for c in PMH_CATEGORIES if lookup_block.get(f"pmh_{c}") == 1}
    return sorted(self_reported - in_record)


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
    medrecon_csv_path=MEDRECON_CSV,
    include_meds: bool = True,
) -> dict:
    """Build the per-subject prior-encounter structures PMH features derive from.

    This is the heavy half of the PMH pipeline (it parses the 3.3 GB
    discharge.csv); the per-stay assembly is cheap by comparison. The index is
    built ONCE over a set of subject_ids and then queried per stay by
    ``assemble_pmh_for_stay``, which applies the ``< intime`` leakage filter
    using the *current* stay's intime. Persisting this index (joblib) and
    querying it at inference is how ``PatientHistoryLookupTool`` simulates an
    EHR lookup without re-parsing the source tables.

    ``include_meds`` (default True) additionally aggregates ``medrecon.csv`` into
    a per-stay medication block, enabling the lookup tool to return a returning
    patient's prior med list (queried leakage-safe via ``assemble_meds_for_stay``).
    The training pipelines build their own current-stay med features, so they
    call this with ``include_meds=False`` to skip the extra read.

    Returns a dict with keys:
        adm_by_subject    {subject_id: sorted [(admittime, hadm_id), ...]}
        ed_by_subject     {subject_id: sorted [(intime, stay_id, complaint_norm), ...]}
        icd_pmh_by_hadm   {hadm_id: set(diagnosis_groups)}  (ICD-derived)
        note_pmh_by_hadm  {hadm_id: set(diagnosis_groups)}  (discharge-note-derived)
        med_by_stay       {stay_id: {MED_FEATURE_COLS}}     (per-stay medrecon block;
                          empty dict when include_meds=False)
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

    # ── (5) Per-stay medication block from medrecon.csv (ED home-med list) ──
    # Keyed on the ED stay_id (medrecon's grain). Aggregated with the SAME
    # recipe training uses (`flags_from_row` OR'd across rows). The lookup tool
    # then returns a RETURNING patient's most-recent PRIOR stay's block via
    # `assemble_meds_for_stay` (leakage-safe, `< intime`). Skipped when
    # include_meds=False (training builds its own current-stay med features).
    med_by_stay: dict[int, dict] = {}
    if include_meds:
        # Local import avoids a package-level circular import: importing
        # proiect_licenta.tools runs tools/__init__, which imports triage_tool,
        # which imports this module. By call time the cycle is resolved.
        from proiect_licenta.tools.med_vocab import med_block_from_rows
        needed_stays = set(int(s) for s in ed_subset["stay_id"].dropna().tolist())
        print(f"    Loading medrecon.csv ({medrecon_csv_path})...")
        if not Path(medrecon_csv_path).exists():
            print(f"    WARNING: {medrecon_csv_path} not found — skipping med index.")
        else:
            med = pd.read_csv(
                medrecon_csv_path, usecols=["stay_id", "name", "etcdescription"],
            )
            med = med[med["stay_id"].isin(needed_stays)]
            for sid, grp in med.groupby("stay_id"):
                med_by_stay[int(sid)] = med_block_from_rows(
                    grp["name"].tolist(), grp["etcdescription"].tolist(),
                )
            print(f"    medrecon.csv: med block built for {len(med_by_stay):,} stays")

    return {
        "adm_by_subject": adm_by_subject,
        "ed_by_subject": ed_by_subject,
        "icd_pmh_by_hadm": icd_pmh_by_hadm,
        "note_pmh_by_hadm": note_pmh_by_hadm,
        "med_by_stay": med_by_stay,
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


def assemble_meds_for_stay(subject_id: int, intime, index: dict):
    """Return the medication block for the patient's MOST RECENT PRIOR ED stay
    (strictly before ``intime``), leakage-safe by the same ``< intime`` rule as
    ``assemble_pmh_for_stay``. Returns ``None`` when the subject has no prior ED
    stay carrying a medrecon record (→ the lookup reports no med record and the
    runtime keeps its ask-the-patient path), or when the index was built without
    medications (``include_meds=False`` / an older index).

    Medications come from the patient's last documented home-med list (medrecon),
    NOT the current encounter — a live patient's current med list is exactly what
    the nurse is collecting, so the EHR can only supply the prior visit's list.
    This is the honest deployment semantics (a strong proxy for stable chronic
    meds), at the cost of not matching a visit where the patient's meds changed.
    """
    intime = pd.to_datetime(intime)
    sid = int(subject_id)
    med_by_stay = index.get("med_by_stay") or {}
    if not med_by_stay:
        return None
    ed_by_subject = index.get("ed_by_subject", {})
    # Prior ED stays strictly before intime, most-recent first.
    prior_ed = sorted(
        ((it, stid) for it, stid, _ in ed_by_subject.get(sid, []) if it < intime),
        reverse=True,
    )
    assert all(it < intime for it, _ in prior_ed), "MED leak: prior ED visit >= intime"
    for _, stid in prior_ed:
        block = med_by_stay.get(int(stid))
        if block is not None:
            return dict(block)
    return None


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
        include_meds=False,  # training builds its own current-stay med features
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
