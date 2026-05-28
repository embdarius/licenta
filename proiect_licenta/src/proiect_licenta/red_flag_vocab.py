"""Red-flag keyword features for the Triage Agent (section 1.5).

A hand-curated set of ~40 binary flags that fire on canonical ESI 1-2
presentations: cardiac, neuro, respiratory, trauma, sepsis, hemorrhage,
OB/GYN, anaphylaxis, and overdose / self-harm. These are concepts that
TF-IDF either treats as bag-of-words (losing the co-occurrence) or filters
out via its min_df cutoff (losing the rare-but-critical signal).

Pattern format:
    Each red flag is a (column_name, [token_patterns]) pair. All token
    patterns must match the (normalized) chief-complaint text for the
    flag to fire. Single-token patterns trivially "all match."

Matching rules:
    - Case-insensitive.
    - Word-boundary at the start, prefix-match (``\\b<token>\\w*``) at the
      end — so "crush" matches "crush", "crushing", "crushed". "slur"
      matches "slurred", "slurring". This matches medical-text conventions
      where the stem is what carries clinical meaning.
    - Patterns containing regex metacharacters (``.* + ? ( [ { | ^ $ \\``)
      are used as-is (with no auto-wrapping), so the author can express
      more complex matches when needed.

Important: these patterns match against the OUTPUT of
`normalize_complaint_text` in `preprocessing.py`, which lowercases,
strips punctuation, and EXPANDS abbreviations. So patterns target the
expanded forms — "chest pain" (not "cp"), "shortness of breath" (not
"sob"), "altered mental status" (not "ams"), "motor vehicle collision"
(not "mvc"), "cerebrovascular accident stroke" (not "cva"), etc. The
ABBREVIATIONS table in preprocessing.py is the source of truth for what
the normalized text actually contains.

Used by:
    training/train_triage_v3.py     — feature engineering for v3 iter 3
    tools/triage_tool.py            — once inference rewiring lands
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Red-flag definitions
# ---------------------------------------------------------------------------
# Order is stable — RED_FLAG_COLS is derived from this list.
#
# Token-pattern conventions:
#   - plain string  → wrapped as r"\b<token>\w*" (stem-prefix match)
#   - regex string  → used as-is (when it contains . * + ? ( [ { | ^ $ \)
#
# For multi-token flags ALL token patterns must match anywhere in the text
# (no proximity constraint — chief complaints are short, ~20-50 chars after
# normalization, so distant co-occurrence within the same complaint is fine).

RED_FLAGS: list[tuple[str, list[str]]] = [
    # ── Cardiac (ESI 1-2) ─────────────────────────────────────────────
    ("rf_chest_crushing",     ["chest",   "crush"]),
    ("rf_chest_radiating",    ["chest",   "radiat"]),
    ("rf_chest_pressure",     ["chest",   "pressure"]),
    ("rf_chest_sob",          ["chest",   "shortness"]),
    ("rf_chest_diaphoresis",  ["chest",   "diaphores|sweat"]),
    ("rf_syncope",            ["syncop"]),
    ("rf_cardiac_arrest",     ["cardiac.*arrest|cpr"]),
    ("rf_palpitations",       ["palpitation"]),

    # ── Neuro (ESI 1-2) ───────────────────────────────────────────────
    ("rf_stroke",             ["stroke|cerebrovascular"]),  # "cva" expands to "cerebrovascular accident stroke"
    ("rf_face_droop",         ["face|facial", "droop"]),
    ("rf_slurred_speech",     ["slur|dysarth"]),
    ("rf_unilateral_weak",    ["weak|paralys|hemipare", "unilateral|left|right|side"]),
    ("rf_worst_headache",     ["worst", "headache|head"]),
    ("rf_altered_mental",     ["altered mental status"]),  # "ams" expands to this
    ("rf_seizure",            ["seizur|convuls|status epilept"]),  # "sz" expands to "seizure"
    ("rf_unresponsive",       ["unrespons|unconscious"]),
    ("rf_loss_consciousness", ["loss of consciousness|passed out|fainted"]),  # "loc" expands

    # ── Respiratory (ESI 1-2) ─────────────────────────────────────────
    ("rf_resp_distress",      ["respiratory.{0,10}distress|severe.{0,20}shortness"]),
    ("rf_cyanosis",           ["cyanos"]),
    ("rf_resp_arrest",        ["respiratory.{0,5}arrest|stopped breathing|not breathing"]),
    ("rf_stridor",            ["stridor"]),

    # ── Trauma (ESI 1-2) ──────────────────────────────────────────────
    ("rf_gunshot",            ["gunshot|gsw"]),
    ("rf_stab_wound",         ["stab"]),
    ("rf_mvc_ejected",        ["motor vehicle", "eject"]),  # "mvc" expands to "motor vehicle collision"
    ("rf_fall_from_height",   ["fall|fell",  "height|story|floor|roof|ladder|stair"]),
    ("rf_pedestrian_struck",  ["pedestrian.{0,15}struck|struck.{0,15}pedestrian|hit by"]),
    ("rf_penetrating",        ["penetrat"]),
    ("rf_assault",            ["assault"]),

    # ── Sepsis / shock (ESI 1-2) ──────────────────────────────────────
    ("rf_septic_shock",       ["sepsis|septic", "shock"]),
    ("rf_sepsis",             ["sepsis|septic"]),
    ("rf_fever_altered",      ["fever",       "altered mental status|altered"]),
    ("rf_hypotension_fever",  ["hypotens",    "fever"]),

    # ── Hemorrhage (ESI 1-2) ──────────────────────────────────────────
    ("rf_active_bleeding",    ["active.{0,5}bleed|massive.{0,5}bleed|hemorrhag"]),
    ("rf_gi_bleed_severe",    ["hematemes|melena|bright red blood per rectum|brbpr"]),
    ("rf_hemoptysis",         ["hemoptys"]),

    # ── OB/GYN emergencies (ESI 1-2) ──────────────────────────────────
    ("rf_ectopic",            ["ectopic"]),
    ("rf_imminent_delivery",  ["imminent.{0,5}deliver|crowning|active labor"]),
    ("rf_pregnant_bleed",     ["pregnan",  "bleed"]),

    # ── Anaphylaxis (ESI 1-2) ─────────────────────────────────────────
    ("rf_anaphylaxis",        ["anaphylax"]),
    ("rf_airway_swelling",    ["throat.{0,10}clos|airway.{0,10}swell|tongue.{0,10}swell"]),

    # ── Overdose / self-harm (ESI 1-2) ────────────────────────────────
    ("rf_overdose",           ["overdose|intentional.{0,10}ingest"]),
    ("rf_suicide_attempt",    ["suicide|suicidal", "attempt|ingest|cut"]),
]


def red_flag_columns() -> list[str]:
    """Stable list of feature-column names emitted by `build_red_flag_features`.

    Order matches `RED_FLAGS`; downstream code (training, inference, audits)
    should call this rather than re-deriving the list, so that adding /
    removing a red flag automatically propagates everywhere.
    """
    return [name for name, _ in RED_FLAGS] + ["rf_any_count", "rf_any"]


# Public alias matching the project's convention (PMH_FEATURE_COLS, etc.).
RED_FLAG_COLS = red_flag_columns()


_REGEX_METAS = set(".*+?()[]{}|^$\\")


def _compile_patterns():
    """Pre-compile each red-flag's token patterns to case-insensitive regexes.

    Plain-string tokens get wrapped as ``\\b<token>\\w*`` (stem-prefix match).
    Tokens containing regex metacharacters are used as-is.
    """
    compiled = []
    for name, tokens in RED_FLAGS:
        token_res = []
        for tok in tokens:
            if any(ch in tok for ch in _REGEX_METAS):
                pat = tok  # author-provided regex
            else:
                pat = r"\b" + re.escape(tok) + r"\w*"
            token_res.append(re.compile(pat, re.IGNORECASE))
        compiled.append((name, token_res))
    return compiled


# Module-level compile (~40 regexes × ~2 tokens each = ~80 patterns total).
# Compile once at import; reuse across millions of rows.
_COMPILED_RED_FLAGS = _compile_patterns()


def red_flag_flags_for_text(text: str) -> dict[str, int]:
    """Compute the binary red-flag flag set for one normalized complaint text.

    Returns a dict with one entry per `RED_FLAG_COLS` column, including
    the aggregate `rf_any_count` (sum across all rf_* flags) and `rf_any`
    (1 if any flag fired).
    """
    if not isinstance(text, str) or not text:
        out = {name: 0 for name, _ in RED_FLAGS}
        out["rf_any_count"] = 0
        out["rf_any"] = 0
        return out

    out: dict[str, int] = {}
    fired = 0
    for name, token_res in _COMPILED_RED_FLAGS:
        if all(r.search(text) for r in token_res):
            out[name] = 1
            fired += 1
        else:
            out[name] = 0
    out["rf_any_count"] = fired
    out["rf_any"] = 1 if fired > 0 else 0
    return out


def build_red_flag_features(texts: Iterable[str]) -> pd.DataFrame:
    """Vectorized red-flag feature builder for an iterable of normalized texts.

    Returns a DataFrame with one column per `RED_FLAG_COLS` entry, in the
    canonical column order, aligned to the input iterable's positional order.
    """
    rows = [red_flag_flags_for_text(t) for t in texts]
    return pd.DataFrame(rows, columns=RED_FLAG_COLS).astype(int)


# ---------------------------------------------------------------------------
# Smoke test — verifies that representative inputs produce the expected flags.
# Imported and asserted at the end of the v3 training pipeline (and in the
# Colab audit cell) to catch regex breakage from refactors.
# ---------------------------------------------------------------------------
SMOKE_TESTS = [
    # (input text after normalize_complaint_text, expected positive flags)
    (
        "chest pain crushing radiating left arm shortness of breath diaphoresis",
        {"rf_chest_crushing", "rf_chest_radiating", "rf_chest_sob", "rf_chest_diaphoresis"},
    ),
    (
        "altered mental status fever",  # "ams" → "altered mental status"
        {"rf_altered_mental", "rf_fever_altered"},
    ),
    (
        "motor vehicle collision ejected from vehicle",  # "mvc" → "motor vehicle collision"
        {"rf_mvc_ejected"},
    ),
    (
        "cerebrovascular accident stroke facial droop slurred speech",  # "cva" → "cerebrovascular accident stroke"
        {"rf_stroke", "rf_face_droop", "rf_slurred_speech"},
    ),
    (
        "worst headache of life sudden onset",
        {"rf_worst_headache"},
    ),
    (
        "gunshot wound to chest",
        {"rf_gunshot"},
    ),
    (
        "anaphylaxis hives throat closing",
        {"rf_anaphylaxis", "rf_airway_swelling"},
    ),
    (
        "fall from second story window",
        {"rf_fall_from_height"},
    ),
    (
        "suicide attempt intentional ingestion overdose",
        {"rf_suicide_attempt", "rf_overdose"},
    ),
    (
        "syncope passed out",
        {"rf_syncope", "rf_loss_consciousness"},
    ),
    (
        "minor laceration finger",  # baseline — should fire NO red flags
        set(),
    ),
]


def run_smoke_tests(verbose: bool = False) -> tuple[int, list[str]]:
    """Run the canonical smoke tests. Returns (n_passed, [failure_messages])."""
    failures: list[str] = []
    n_passed = 0
    for text, expected_positive in SMOKE_TESTS:
        actual = red_flag_flags_for_text(text)
        actual_positive = {k for k, v in actual.items()
                           if v == 1 and k.startswith("rf_")
                           and k not in ("rf_any", "rf_any_count")}
        if actual_positive == expected_positive:
            n_passed += 1
            if verbose:
                print(f"  OK   {text[:60]!r:62s} -> {sorted(actual_positive)}")
        else:
            missing = expected_positive - actual_positive
            extra = actual_positive - expected_positive
            msg = (f"FAIL {text[:60]!r}: "
                   f"missing={sorted(missing)}, extra={sorted(extra)}")
            failures.append(msg)
            if verbose:
                print(f"  {msg}")
    return n_passed, failures


if __name__ == "__main__":
    n, failures = run_smoke_tests(verbose=True)
    print(f"\n{n}/{len(SMOKE_TESTS)} smoke tests passed.")
    for f in failures:
        print(f"  {f}")
