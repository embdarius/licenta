"""
Four-way comparison: v1 vs v2 vs v3 base vs v3 with-nurse.

Reads the metadata.json saved by each training pipeline and prints a
single side-by-side table of the headline metrics. No model loading,
no data loading -- fast enough to run after every training cycle.

For thesis writeup. Note that v1/v2 use a 14-class label space and v3
uses 13 classes (catch-all excluded), so the diagnosis accuracy numbers
are NOT directly comparable across versions -- the table includes a
clear column for label-space size to keep that visible.
"""

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import (
    DOCTOR_V1_DIR, DOCTOR_V2_DIR,
    DOCTOR_V3_BASE_DIR, DOCTOR_V3_DIR,
)


VERSIONS = [
    ("v1", DOCTOR_V1_DIR, "no nurse data, 100K sample, 14 classes"),
    ("v2", DOCTOR_V2_DIR, "snapshot vitals + meds, 100K sample, 14 classes"),
    ("v3 base", DOCTOR_V3_BASE_DIR, "no nurse data, full data, 13 classes (catch-all out)"),
    ("v3 nurse", DOCTOR_V3_DIR, "snapshot + longitudinal + rhythm + meds, full, 13 classes"),
]


def fmt_pct(x):
    if x is None:
        return "    n/a"
    return f"{100 * x:6.2f}%"


def fmt_int(x):
    if x is None:
        return "   n/a"
    return f"{x:>6,}"


def main():
    print("\n" + "#" * 90)
    print("  Doctor Model Comparison -- v1 / v2 / v3 base / v3 with-nurse")
    print("#" * 90)

    rows = []
    for label, art_dir, blurb in VERSIONS:
        meta_path = art_dir / "metadata.json"
        if not meta_path.exists():
            rows.append({
                "label": label,
                "blurb": blurb,
                "missing": True,
                "diag_acc": None, "dept_acc": None,
                "n_diag": None, "n_dept": None,
                "trained_at": "(not yet trained)",
                "n_train": None, "n_test": None,
                "catch_all_excluded": None,
            })
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append({
            "label": label,
            "blurb": blurb,
            "missing": False,
            "diag_acc": meta.get("diagnosis_accuracy"),
            "dept_acc": meta.get("department_accuracy"),
            "n_diag": meta.get("n_diagnosis_classes"),
            "n_dept": meta.get("n_department_classes"),
            "trained_at": meta.get("trained_at", "?"),
            "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "catch_all_excluded": meta.get("catch_all_excluded"),
        })

    # ────────────────────────────────────────────────────────────────────
    # Headline accuracy table
    # ────────────────────────────────────────────────────────────────────
    print(f"\n  {'Version':<12s}  {'Diag classes':>12s}  {'Diag acc':>10s}  "
          f"{'Dept classes':>12s}  {'Dept acc':>10s}  {'n_train':>8s}  {'n_test':>8s}")
    print(f"  {'-' * 86}")
    for r in rows:
        if r["missing"]:
            print(f"  {r['label']:<12s}  {'(no metadata.json -- not yet trained)':<70s}")
            continue
        print(
            f"  {r['label']:<12s}  "
            f"{fmt_int(r['n_diag']):>12s}  {fmt_pct(r['diag_acc']):>10s}  "
            f"{fmt_int(r['n_dept']):>12s}  {fmt_pct(r['dept_acc']):>10s}  "
            f"{fmt_int(r['n_train']):>8s}  {fmt_int(r['n_test']):>8s}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Improvement deltas (only when prior tier is available)
    # ────────────────────────────────────────────────────────────────────
    print(f"\n  Improvement deltas (top-1 diagnosis / top-1 department):")
    print(f"  {'Comparison':<35s}  {'Delta Diag':>10s}  {'Delta Dept':>10s}  Note")
    print(f"  {'-' * 86}")

    pairs = [
        ("v1 -> v2 (added nurse data)",         0, 1, "same 14-class space"),
        ("v3 base -> v3 nurse (added nurse)",   2, 3, "same 13-class space"),
        ("v2 -> v3 nurse (full pipeline)",      1, 3, "DIFFERENT label spaces; not apples-to-apples"),
    ]

    def delta_str(prev, cur):
        if prev is None or cur is None:
            return "    n/a"
        d = cur - prev
        sign = "+" if d >= 0 else ""
        return f"{sign}{100*d:5.2f}pp"

    for name, i, j, note in pairs:
        a, b = rows[i], rows[j]
        if a["missing"] or b["missing"]:
            print(f"  {name:<35s}  {'(skip)':>10s}  {'(skip)':>10s}  one or both not trained")
            continue
        d_diag = delta_str(a["diag_acc"], b["diag_acc"])
        d_dept = delta_str(a["dept_acc"], b["dept_acc"])
        print(f"  {name:<35s}  {d_diag:>10s}  {d_dept:>10s}  {note}")

    # ────────────────────────────────────────────────────────────────────
    # Per-version notes
    # ────────────────────────────────────────────────────────────────────
    print(f"\n  Version details:")
    for r in rows:
        marker = "  [MISSING]" if r["missing"] else ""
        print(f"    {r['label']:<12s} {marker}")
        print(f"       {r['blurb']}")
        if not r["missing"]:
            print(f"       trained_at: {r['trained_at']}")
            if r["catch_all_excluded"]:
                print(f"       catch_all_excluded: {r['catch_all_excluded']}")

    print("\n" + "#" * 90)
    print("  COMPARISON COMPLETE")
    print("  Reminder: v1/v2 evaluate on 14 classes (incl. catch-all bucket),")
    print("            v3 evaluates on 13 classes (catch-all excluded).")
    print("            The v3 numbers are not strictly higher than v1/v2 by")
    print("            magic -- they reflect a different (cleaner) task.")
    print("#" * 90 + "\n")


if __name__ == "__main__":
    main()
