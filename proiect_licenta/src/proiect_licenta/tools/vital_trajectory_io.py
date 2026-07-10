"""Parse the vital-trajectory JSON string into a readings dict.

The nurse tool emits an optional vital_trajectory block (multiple chronological
readings per vital). The disposition and v3 tools accept it as a single JSON
string argument so it travels through one tool-call parameter rather than ~36
numeric args.

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

    Never raises - malformed input degrades to an empty dict so the caller
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


def parse_rhythm_readings(raw) -> list:
    """Extract the list of rhythm strings from the same vital-trajectory blob's
    ``rhythm`` key, else []. Tolerant: scalar -> [str], list -> [str, ...],
    drops None / empty / skip-words. Never raises.

    Carried inside ``vital_trajectory_json`` (rather than as a separate tool arg)
    so the existing single-blob plumbing flows nurse -> tools -> benchmark
    unchanged. ``parse_vital_trajectory`` ignores the ``rhythm`` key (it only
    reads the 6 numeric vitals), so the two parsers don't collide."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        obj = raw
    else:
        s = str(raw).strip()
        if s.lower() in _SKIP:
            return []
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(obj, dict):
            return []
    val = obj.get("rhythm")
    if val is None:
        return []
    seq = val if isinstance(val, (list, tuple)) else [val]
    out = []
    for r in seq:
        if r is None:
            continue
        s = str(r).strip()
        if s and s.lower() not in _SKIP:
            out.append(s)
    return out
