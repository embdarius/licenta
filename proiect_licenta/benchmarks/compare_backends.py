"""Flash vs MedGemma head-to-head from two detailed generated-cases runs.

Reads the per-backend audit folders produced by
``benchmark_pipeline_e2e.py --out-dir`` (each holding ``e2e_full_<backend>.json``
+ ``parser_audit.csv``) and emits a detailed side-by-side comparison:

  backend_comparison.json              every metric: flash | medgemma | delta
  backend_comparison_metrics.csv       flat table, one row per metric
  backend_comparison_parser_per_case.csv   per case: each backend's parse + winner

The comparison focuses on the ``parser_llm`` mode (the fair, orchestration-
independent cross-backend anchor) by default; override with ``--mode``.

Run:
    uv run python benchmarks/compare_backends.py \
        --flash-dir   .../generated_cases/flash \
        --medgemma-dir .../generated_cases/medgemma \
        --out-dir     .../generated_cases/comparison
"""

import sys
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
from pathlib import Path

import pandas as pd


# Metrics where a LOWER value is better (everything else: higher is better).
LOWER_IS_BETTER = (
    "under_triage", "over_triage", "mae", "brier", "log_loss", "ece", "mce",
    "mean_signed_error", "mean_absolute_error",
)


def _direction(metric: str) -> str:
    return "lower_better" if any(k in metric for k in LOWER_IS_BETTER) else "higher_better"


def _flatten(obj, prefix=""):
    """Flatten nested dicts to {dotted.key: numeric_value} (numbers only)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        # Lists of dicts (e.g. per-threshold / per-class) keyed by an id field.
        for i, v in enumerate(obj):
            key_id = None
            if isinstance(v, dict):
                for id_field in ("threshold", "name", "label", "group", "bin_center"):
                    if id_field in v:
                        key_id = f"{id_field}={v[id_field]}"
                        break
            out.update(_flatten(v, f"{prefix}[{key_id or i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _load_backend(d: Path):
    """Return (backend_name, full_json, parser_audit_df) for an audit dir."""
    js = sorted(d.glob("e2e_full_*.json"))
    if not js:
        raise FileNotFoundError(f"no e2e_full_*.json in {d}")
    data = json.loads(js[0].read_text(encoding="utf-8"))
    backend = data.get("llm_backend") or js[0].stem.replace("e2e_full_", "")
    pa_path = d / "parser_audit.csv"
    pa = pd.read_csv(pa_path) if pa_path.exists() else pd.DataFrame()
    return backend, data, pa


def _mode_metrics(data: dict, mode: str) -> dict:
    """Flatten the comparable metric surface for one mode of one backend."""
    flat = {}
    md = data.get("metrics_detailed", {}).get(mode, {})
    flat.update(_flatten(md, "detailed"))
    basic = data.get("metrics_basic", {}).get(mode, {})
    flat.update(_flatten(basic, "basic"))
    nlf = data.get("nl_fidelity", {}).get(mode, {})
    flat.update(_flatten(nlf, "nl_fidelity"))
    return flat


def parser_cost_table(fdata, mdata, fb, mb, out_dir,
                      ref_modes=("feature_vector_gated", "tool_direct"),
                      parser_mode="parser_llm"):
    """Write `parser_cost.csv`: the NLP-parser-cost decomposition.

    Columns per metric: the backend-invariant reference ladder
    (`feature_vector_gated`, `tool_direct` - neither uses the LLM, so they're
    identical across backends and taken from the Flash run), each backend's
    `parser_llm`, and the isolated parser cost `parser_llm - tool_direct` per
    backend (the only step that differs is the LLM-parsed chief complaint).
    """
    fmd = fdata.get("metrics_detailed", {})
    mmd = mdata.get("metrics_detailed", {})
    if parser_mode not in fmd or parser_mode not in mmd:
        print("    parser_cost: parser_llm mode missing in a backend; skipped")
        return
    refs = {rm: _flatten(fmd[rm], "") for rm in ref_modes if rm in fmd}
    td, fvg = refs.get("tool_direct", {}), refs.get("feature_vector_gated", {})
    fpl, mpl = _flatten(fmd[parser_mode], ""), _flatten(mmd[parser_mode], "")
    # Sanity: reference modes must be backend-invariant.
    for rm in ref_modes:
        if rm in fmd and rm in mmd:
            a, b = _flatten(fmd[rm], ""), _flatten(mmd[rm], "")
            md = max((abs(a[k] - b[k]) for k in a if k in b), default=0.0)
            if md > 1e-6:
                print(f"    parser_cost: WARN '{rm}' differs across backends "
                      f"(max {md:.2e}) - expected backend-invariant")
    rows = []
    for k in sorted(set(fvg) | set(td) | set(fpl) | set(mpl)):
        f_cost = (fpl[k] - td[k]) if (k in fpl and k in td) else None
        m_cost = (mpl[k] - td[k]) if (k in mpl and k in td) else None
        rows.append({
            "metric": k, "direction": _direction(k),
            "feature_vector_gated": fvg.get(k), "tool_direct": td.get(k),
            f"{fb}_parser_llm": fpl.get(k), f"{mb}_parser_llm": mpl.get(k),
            f"{fb}_parser_cost(pl_minus_td)": f_cost,
            f"{mb}_parser_cost(pl_minus_td)": m_cost,
        })
    pd.DataFrame(rows).to_csv(out_dir / "parser_cost.csv", index=False, encoding="utf-8")
    print(f"    csv -> parser_cost.csv  ({len(rows)} metrics; references backend-invariant)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flash-dir", required=True)
    ap.add_argument("--medgemma-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="parser_llm",
                    help="Which pipeline mode to compare (default parser_llm).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fb, fdata, fpa = _load_backend(Path(args.flash_dir))
    mb, mdata, mpa = _load_backend(Path(args.medgemma_dir))
    print(f"  Comparing [{fb}] vs [{mb}]  on mode '{args.mode}'")

    fflat = _mode_metrics(fdata, args.mode)
    mflat = _mode_metrics(mdata, args.mode)
    all_keys = sorted(set(fflat) | set(mflat))

    rows = []
    for k in all_keys:
        fv, mv = fflat.get(k), mflat.get(k)
        delta = (mv - fv) if (fv is not None and mv is not None) else None
        direction = _direction(k)
        winner = ""
        if delta is not None and abs(delta) > 1e-9:
            if direction == "higher_better":
                winner = mb if delta > 0 else fb
            else:
                winner = mb if delta < 0 else fb
        rows.append({
            "metric": k, "mode": args.mode, "direction": direction,
            fb: fv, mb: mv, "delta_medgemma_minus_flash": delta, "better": winner,
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "backend_comparison_metrics.csv",
                      index=False, encoding="utf-8")
    print(f"    csv -> backend_comparison_metrics.csv  ({len(metrics_df)} metrics)")

    # Parser-cost decomposition (reference ladder + isolated NLP-parser cost).
    parser_cost_table(fdata, mdata, fb, mb, out_dir)

    # Per-case parser comparison.
    parser_rows = []
    if not fpa.empty and not mpa.empty:
        fpa = fpa.set_index("stay_id")
        mpa = mpa.set_index("stay_id")
        common = [s for s in fpa.index if s in mpa.index]
        for sid in common:
            fr, mr = fpa.loc[sid], mpa.loc[sid]
            ff, mf = fr.get("complaint_f1"), mr.get("complaint_f1")
            winner = ""
            if pd.notna(ff) and pd.notna(mf) and abs(ff - mf) > 1e-9:
                winner = mb if mf > ff else fb
            parser_rows.append({
                "stay_id": sid, "narrative": fr.get("narrative"),
                "chief_complaints_tabular": fr.get("chief_complaints_tabular"),
                f"{fb}_complaints": fr.get("parser_complaints_used"),
                f"{mb}_complaints": mr.get("parser_complaints_used"),
                f"{fb}_complaint_f1": ff, f"{mb}_complaint_f1": mf,
                "complaint_f1_winner": winner,
                f"{fb}_age_ok": fr.get("age_ok"), f"{mb}_age_ok": mr.get("age_ok"),
                f"{fb}_gender_ok": fr.get("gender_ok"), f"{mb}_gender_ok": mr.get("gender_ok"),
                f"{fb}_transport_ok": fr.get("transport_ok"),
                f"{mb}_transport_ok": mr.get("transport_ok"),
            })
    pd.DataFrame(parser_rows).to_csv(
        out_dir / "backend_comparison_parser_per_case.csv", index=False, encoding="utf-8")
    print(f"    csv -> backend_comparison_parser_per_case.csv  ({len(parser_rows)} cases)")

    # JSON summary (+ a small headline block of the most-watched metrics).
    headline_keys = [k for k in all_keys if any(s in k for s in (
        "acuity.exact", "disposition.thresholds[threshold=0.4]",
        "diagnosis.top1", "department.top1", "exact_icd.strict@1",
        "exact_icd.flat@10", "exact_icd.union", "graded_best_mean",
        "complaint_f1", "complaint_jaccard", "age_ok", "gender_ok", "transport_ok"))]
    summary = {
        "flash_backend": fb, "medgemma_backend": mb, "mode": args.mode,
        "n_metrics": len(all_keys),
        "n_parser_cases": len(parser_rows),
        "headline": {k: {fb: fflat.get(k), mb: mflat.get(k),
                         "delta": (mflat.get(k) - fflat.get(k))
                         if (fflat.get(k) is not None and mflat.get(k) is not None) else None}
                     for k in headline_keys},
        "all_metrics": rows,
    }
    (out_dir / "backend_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"    json -> backend_comparison.json")
    print("  Backend comparison complete.")


if __name__ == "__main__":
    main()
