"""Build the Gemini ICD-title embedding cache for the graded near-miss metric.

Standalone one-time helper (no heavy MIMIC load): embeds the distinct ICD titles
the Stage-2 graded benchmark needs and caches the vectors to disk, so
benchmark_icd_resolution.py reads them offline. The title set covers every
candidate code's representative title in the resolver index plus every distinct
primary-diagnosis title in categorized_diagnosis.csv. Embedding is rate-limited
and resumable, so re-running only embeds titles not already cached.

Run: uv run python benchmarks/build_title_embeddings.py
"""
import sys
from pathlib import Path

import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import DIAGNOSIS_CSV, DOCTOR_V3_ICD_RESOLVER_DIR
from proiect_licenta import icd_resolution as icdr
from proiect_licenta import icd_similarity as icds


def main():
    cache_path = DOCTOR_V3_ICD_RESOLVER_DIR / "title_embeddings.joblib"

    titles = set()
    # 1. Resolver candidate titles (both granularities).
    resolver = icdr.load_resolver(DOCTOR_V3_ICD_RESOLVER_DIR)
    for g in icdr.GRANULARITIES:
        for entry in resolver["granularities"][g].values():
            titles.update(str(t) for t in entry["titles"])
    n_cand = len(titles)

    # 2. All distinct primary-diagnosis titles (superset of any split's true titles).
    diag = pd.read_csv(DIAGNOSIS_CSV, dtype={"icd_title": str}, usecols=["seq_num", "icd_title"])
    diag = diag[diag["seq_num"] == 1]
    titles.update(diag["icd_title"].fillna("").astype(str).unique().tolist())

    print(f"  Candidate titles: {n_cand:,} | + primary-dx titles -> {len(titles):,} distinct")
    cache = icds.build_or_load_title_embeddings(titles, cache_path)
    if cache is None:
        print("  Cache NOT built (engine unavailable). See message above.")
        return
    dim = len(next(iter(cache.values()))) if cache else 0
    print(f"  Done: {len(cache):,} vectors (dim {dim}) at {cache_path}")


if __name__ == "__main__":
    main()
