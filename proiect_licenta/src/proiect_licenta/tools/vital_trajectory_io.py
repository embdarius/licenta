"""
Parse the LLM/nurse-provided vital-trajectory JSON string into a readings dict.

The nurse tool emits an optional ``vital_trajectory`` block — multiple
chronological readings per vital collected during the ED stay. The doctor
disposition + v3 reassessment tools accept it as a single JSON string argument
(``vital_trajectory_json``) so it travels cleanly through one LLM tool-call
parameter rather than ~36 separate numeric args.

Accepted shapes (all tolerated; unknown keys ignored):
    {"temperature": [98.1, 99.0], "heartrate": [88, 105, 112], ...}
    {"heartrate": 88}                      # scalar coerced to [88]
    ""  / "none" / "{}"                    # -> {} (snapshot fallback upstream)
"""

import json

_VITALS = ("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp")
_SKIP = {"", "none", "null", "n/a", "na", "-", "{}", "unknown", "skip"}


def parse_vital_trajectory(raw) -> dict:
    """Return {vital: [float, ...]} for whatever vitals are present, else {}.

    Never raises — malformed input degrades to an empty dict so the caller
    falls back to the single-snapshot representation."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        obj = raw
    else:
        s = str(raw).strip()
        if s.lower() in _SKIP:
            return {}
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(obj, dict):
            return {}

    out: dict = {}
    for vital in _VITALS:
        if vital not in obj:
            continue
        val = obj[vital]
        seq = val if isinstance(val, (list, tuple)) else [val]
        cleaned = []
        for r in seq:
            if r is None:
                continue
            try:
                cleaned.append(float(r))
            except (TypeError, ValueError):
                continue
        if cleaned:
            out[vital] = cleaned
    return out
