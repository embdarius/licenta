"""
Medication Vocabulary Audit — Training vs Inference Mismatch

Problem: Doctor v2 training flags come from keyword-matching etcdescription
(pharmacy class terms like "beta blocker") in medrecon.csv, while inference
flags come from matching patient-reported drug names (e.g., "metoprolol")
against DRUG_NAME_MAP in doctor_tool_v2. The two vocabularies never touch.

This script quantifies the mismatch:
  - per-category: stays flagged by training only vs inference only vs both
  - top drug names that training catches but inference misses (candidates to
    add to DRUG_NAME_MAP)
  - top etcdescriptions that inference catches but training misses (candidates
    to add to MED_CATEGORY_KEYWORDS)

Run: uv run python audit_med_vocab.py
"""

from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# Shared vocabulary (both training and inference now import from here).
from proiect_licenta.tools.med_vocab import (
    DRUG_NAME_MAP, MED_CLASS_KEYWORDS, MED_CATEGORIES,
    flags_from_name, flags_from_text,
)

MED_CATEGORY_KEYWORDS = MED_CLASS_KEYWORDS
MED_KEYWORD_MAP = MED_CLASS_KEYWORDS


def classify_medication(etcdescription):
    """Training side: etcdescription -> flags via shared class-keyword matcher."""
    return flags_from_text(etcdescription)


def inference_flags_from_name(name):
    """Inference side: drug name -> flags via shared name-token matcher."""
    return flags_from_name(name)

MEDRECON_CSV = (
    Path(__file__).resolve().parent
    / "src" / "proiect_licenta" / "datasets" / "datasets_mimic-iv"
    / "mimic-iv-ed" / "medrecon.csv"
)


def main():
    print("=" * 70)
    print("  Medication Vocabulary Audit — Training vs Inference")
    print("=" * 70)

    print(f"\nLoading {MEDRECON_CSV.name} ...")
    med = pd.read_csv(MEDRECON_CSV, usecols=["stay_id", "name", "etcdescription"])
    print(f"  {len(med):,} rows, {med['stay_id'].nunique():,} unique stays")

    # ── Per-row flags ──
    print("\nComputing per-row flags (training side = etcdescription)...")
    med["train_flags"] = med["etcdescription"].apply(classify_medication)

    print("Computing per-row flags (inference side = name via DRUG_NAME_MAP)...")
    med["infer_flags"] = med["name"].apply(inference_flags_from_name)

    # ── Aggregate to stay level (union of flags) ──
    print("\nAggregating to stay level (union of flags per stay)...")
    train_by_stay = med.groupby("stay_id")["train_flags"].apply(
        lambda s: set().union(*s)
    )
    infer_by_stay = med.groupby("stay_id")["infer_flags"].apply(
        lambda s: set().union(*s)
    )

    n_stays = len(train_by_stay)
    print(f"  {n_stays:,} stays with medication data")

    # ── Per-category comparison ──
    print("\n" + "=" * 70)
    print("  Per-category stay-level flag comparison")
    print("=" * 70)
    print(
        f"\n  {'Category':<28} {'Train':>8} {'Infer':>8} "
        f"{'Both':>8} {'TrainOnly':>10} {'InferOnly':>10} {'Jaccard':>8}"
    )
    print("  " + "-" * 84)

    for cat in MED_CATEGORIES:
        train_has = train_by_stay.apply(lambda s: cat in s)
        infer_has = infer_by_stay.apply(lambda s: cat in s)

        n_train = int(train_has.sum())
        n_infer = int(infer_has.sum())
        n_both = int((train_has & infer_has).sum())
        n_train_only = n_train - n_both
        n_infer_only = n_infer - n_both
        n_union = n_train + n_infer - n_both
        jaccard = n_both / n_union if n_union > 0 else 0.0

        print(
            f"  {cat:<28} {n_train:>8,} {n_infer:>8,} "
            f"{n_both:>8,} {n_train_only:>10,} {n_infer_only:>10,} "
            f"{jaccard:>7.1%}"
        )

    # ── Drug names flagged by training but NOT inference (per category) ──
    print("\n" + "=" * 70)
    print("  Top drug NAMES flagged by TRAINING but MISSED by INFERENCE")
    print("  (candidates to add to DRUG_NAME_MAP)")
    print("=" * 70)

    for cat in MED_CATEGORIES:
        # Rows where training tagged this cat but inference did not
        mask = med["train_flags"].apply(lambda s: cat in s) & \
               ~med["infer_flags"].apply(lambda s: cat in s)
        missed_names = med.loc[mask, "name"].str.lower().str.strip()
        top = Counter(missed_names).most_common(15)
        if not top:
            continue
        print(f"\n  [{cat}] - {len(missed_names):,} rows miss")
        for name, count in top:
            print(f"    {count:>6,}  {name}")

    # etcdescriptions flagged by inference but NOT training (per category)
    print("\n" + "=" * 70)
    print("  Top etcdescriptions flagged by INFERENCE but MISSED by TRAINING")
    print("  (candidates to add to MED_CATEGORY_KEYWORDS)")
    print("=" * 70)

    for cat in MED_CATEGORIES:
        mask = med["infer_flags"].apply(lambda s: cat in s) & \
               ~med["train_flags"].apply(lambda s: cat in s)
        missed_descs = med.loc[mask, "etcdescription"].dropna().str.strip()
        top = Counter(missed_descs).most_common(10)
        if not top:
            continue
        print(f"\n  [{cat}] - {len(missed_descs):,} rows miss")
        for desc, count in top:
            print(f"    {count:>6,}  {desc[:80]}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    # Stays where the two vocabularies agree on the FULL flag set
    agree = sum(
        1 for sid in train_by_stay.index
        if train_by_stay[sid] == infer_by_stay.get(sid, set())
    )
    print(f"  Stays where train flags == infer flags exactly: "
          f"{agree:,} / {n_stays:,} ({100*agree/n_stays:.1f}%)")

    # Any-flag coverage
    train_any = (train_by_stay.apply(len) > 0).sum()
    infer_any = (infer_by_stay.apply(len) > 0).sum()
    print(f"  Stays with >=1 train flag: {train_any:,} ({100*train_any/n_stays:.1f}%)")
    print(f"  Stays with >=1 infer flag: {infer_any:,} ({100*infer_any/n_stays:.1f}%)")


if __name__ == "__main__":
    main()
