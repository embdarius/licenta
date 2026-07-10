"""Longitudinal-vitals builder for runtime inference.

Training turns the multiple vitalsign.csv readings within [intime, intime + 4h]
into the 41-column longitudinal block the doctor v3 and disposition models
consume (per-vital min/max/last/delta, abnormal-reading counts, rhythm one-hot,
rhythm_irregular, has_longitudinal_vitals). At inference the runtime historically
had only one snapshot, collapsing the block and systematically lowering the
disposition model's admit probability (mean P(admit) 0.624 to 0.486 on the
validation admits). This module builds the same block from multiple readings when
the nurse collects a short trajectory, so inference features match training. The
thresholds, clip ranges, rhythm bucketing, and column order are imported from
train_nurse_v3 so training and inference cannot drift.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from proiect_licenta.training.train_nurse_v3 import (
    _normalize_rhythm,
    _RHYTHM_BUCKETS,
    _VITAL_COLS_LONG,
    LONG_VITAL_FEATURE_COLS,
)

# Same clip ranges as train_nurse._clean_vitals (kept local to avoid importing
# the heavy training module's private constants).
_CLIP = {
    "temperature": (90.0, 110.0),
    "heartrate": (20.0, 250.0),
    "resprate": (4.0, 60.0),
    "o2sat": (50.0, 100.0),
    "sbp": (50.0, 300.0),
    "dbp": (20.0, 200.0),
}

# Abnormal-reading thresholds (mirror _aggregate_vitalsigns / _clean_vitals).
# (count_col, vital, op, threshold) with op ">" or "<".
_ABNORMAL = [
    ("n_fever_readings", "temperature", ">", 100.4),
    ("n_tachycardia_readings", "heartrate", ">", 100),
    ("n_bradycardia_readings", "heartrate", "<", 60),
    ("n_tachypnea_readings", "resprate", ">", 20),
    ("n_hypoxia_readings", "o2sat", "<", 94),
    ("n_hypertension_readings", "sbp", ">", 140),
    ("n_hypotension_readings", "sbp", "<", 90),
]

# Snapshot clinical-flag fallback when a vital has no trajectory readings
# (mirrors _fill_longitudinal_vitals: count := snapshot 0/1 flag).
_SNAPSHOT_FLAG = {
    "n_fever_readings": lambda s: 1 if s["temperature"] > 100.4 else 0,
    "n_tachycardia_readings": lambda s: 1 if s["heartrate"] > 100 else 0,
    "n_bradycardia_readings": lambda s: 1 if s["heartrate"] < 60 else 0,
    "n_tachypnea_readings": lambda s: 1 if s["resprate"] > 20 else 0,
    "n_hypoxia_readings": lambda s: 1 if s["o2sat"] < 94 else 0,
    "n_hypertension_readings": lambda s: 1 if s["sbp"] > 140 else 0,
    "n_hypotension_readings": lambda s: 1 if s["sbp"] < 90 else 0,
}


def _clip(vital: str, value: float) -> float:
    lo, hi = _CLIP[vital]
    return float(min(max(value, lo), hi))


def _clean_readings(readings: Optional[dict]) -> dict:
    """Coerce {vital: [raw, ...]} into {vital: [clipped float, ...]}.

    Drops non-numeric/None entries and clips to plausible ranges. Chronological
    order is preserved so `last` and `delta` are time-correct.
    """
    out = {v: [] for v in _VITAL_COLS_LONG}
    if not readings:
        return out
    for vital in _VITAL_COLS_LONG:
        for r in readings.get(vital, []) or []:
            if r is None:
                continue
            try:
                out[vital].append(_clip(vital, float(r)))
            except (TypeError, ValueError):
                continue
    return out


def build_longitudinal_block(
    snapshot: dict,
    readings: Optional[dict] = None,
    rhythm: str = "",
    rhythm_readings: Optional[list] = None,
) -> dict:
    """Build the 41-column longitudinal feature block.

    snapshot supplies the single current/imputed vital values, used as the
    fallback for any vital with no trajectory readings. readings holds multiple
    chronological readings per vital. rhythm is a single free-text rhythm;
    rhythm_readings is multiple. When multiple rhythms are given the block is
    built the same way training aggregates the vitalsign.csv rhythm column: the
    one-hot is the mode and rhythm_irregular is 1 if any reading is non-sinus.
    Returns a dict with exactly the keys in LONG_VITAL_FEATURE_COLS.
    """
    cleaned = _clean_readings(readings)
    block: dict = {}

    any_reading = any(len(cleaned[v]) > 0 for v in _VITAL_COLS_LONG)

    for vital in _VITAL_COLS_LONG:
        vals = cleaned[vital]
        if vals:
            block[f"{vital}_min"] = float(min(vals))
            block[f"{vital}_max"] = float(max(vals))
            block[f"{vital}_last"] = float(vals[-1])
            block[f"{vital}_delta"] = float(vals[-1] - vals[0])
        else:
            snap = float(snapshot[vital])
            block[f"{vital}_min"] = snap
            block[f"{vital}_max"] = snap
            block[f"{vital}_last"] = snap
            block[f"{vital}_delta"] = 0.0

    for count_col, vital, op, thr in _ABNORMAL:
        vals = cleaned[vital]
        if vals:
            if op == ">":
                block[count_col] = int(sum(1 for v in vals if v > thr))
            else:
                block[count_col] = int(sum(1 for v in vals if v < thr))
        else:
            block[count_col] = int(_SNAPSHOT_FLAG[count_col](snapshot))

    # Rhythm one-hot: bucket every reading, one-hot the mode, rhythm_irregular =
    # any non-sinus. Mirrors train_nurse_v3._aggregate_vitalsigns.
    if rhythm_readings:
        buckets = [_normalize_rhythm(r) for r in rhythm_readings]
    elif rhythm:
        buckets = [_normalize_rhythm(rhythm)]
    else:
        buckets = []
    buckets = [b for b in buckets if b]  # drop empties (matches training)

    for b in _RHYTHM_BUCKETS:
        block[f"rhythm_{b}"] = 0
    block["rhythm_irregular"] = 0
    if buckets:
        top = Counter(buckets).most_common(1)[0][0]
        block[f"rhythm_{top}"] = 1
        block["rhythm_irregular"] = int(
            any(b not in ("sinus", "") for b in buckets)
        )

    # Coverage flag: 1 if we had any real reading (matches training's "stay had
    # vitalsign.csv coverage"), 0 for pure snapshot fallback.
    block["has_longitudinal_vitals"] = 1 if any_reading else 0

    # Guarantee exact column coverage / order.
    return {col: block[col] for col in LONG_VITAL_FEATURE_COLS}
