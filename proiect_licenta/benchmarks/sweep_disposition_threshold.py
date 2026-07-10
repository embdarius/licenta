"""Disposition decision-threshold sweep (no retraining).

Scores the existing calibrated disposition model over the same held-out test
split as benchmark_doctor_disposition.py and reports sensitivity, specificity,
over/under-triage, F1, and Youden's J at a range of decision thresholds. A pure
operating-point analysis: predict_proba is fixed, only the cutoff varies (lower
thresholds buy admit-recall at the cost of specificity).

Run: uv run python benchmarks/sweep_disposition_threshold.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proiect_licenta.paths import DOCTOR_V3_DIR
from proiect_licenta.training.train_doctor_disposition import (
    load_and_clean_data, build_features,
)
# build_features loads the triage v3 acuity model, pickled with a custom
# eval_metric whose __module__ is '__main__'. Inject the shim before any load
# so joblib.load can resolve it on this fresh process. (Same fix as the tools.)
from proiect_licenta.tools.triage_tool import _ensure_pickle_compat_in_main
_ensure_pickle_compat_in_main()

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
CURRENT = 0.50


def metrics_at(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / max(tp + fn, 1)          # admit recall
    spec = tn / max(tn + fp, 1)          # discharge recall
    over = fp / max(tn + fp, 1)          # discharge -> admit (false admit)
    under = fn / max(fn + tp, 1)         # admit -> discharge (MISSED admit)
    ppv = tp / max(tp + fp, 1)           # precision on admit
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
    youden = sens + spec - 1
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, sens=sens, spec=spec, over=over,
                under=under, ppv=ppv, acc=acc, f1=f1, youden=youden)


def main():
    print("  DOCTOR DISPOSITION v3 - THRESHOLD SWEEP (no retraining)")

    df = load_and_clean_data().reset_index(drop=True)
    y = df["admitted"].astype(int).reset_index(drop=True)

    # Reproduce the SAME split as benchmark_doctor_disposition.py from indices +
    # labels only (no feature matrix needed for the split). Then build features
    # on ONLY the test rows - building on the full 418K densifies a ~6.9 GB
    # TF-IDF matrix and OOMs. build_features is a pure fit=False transform, so
    # per-row results are identical to building on the full frame and slicing.
    _, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42, stratify=y,
    )
    test_idx = np.sort(test_idx)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    X_test = build_features(df_test).reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True).values

    calibrated = joblib.load(DOCTOR_V3_DIR / "disposition_model.joblib")
    proba = calibrated.predict_proba(X_test)[:, 1]

    n = len(y_test)
    n_admit = int(y_test.sum())
    n_disch = n - n_admit
    auc = roc_auc_score(y_test, proba)
    print(f"\n  Test rows: {n:,}  (admit {n_admit:,} / discharge {n_disch:,}, "
          f"admit rate {y_test.mean():.4f})")
    print(f"  ROC AUC (threshold-independent): {auc:.4f}")
    print(f"  [Lowering the threshold trades specificity for admit-recall; "
          f"AUC is fixed.]\n")

    hdr = (f"  {'thr':>5} {'acc':>7} {'sens':>7} {'spec':>7} {'under':>7} "
           f"{'over':>7} {'F1':>7} {'Youden':>7}   {'TP':>6} {'FN':>6} "
           f"{'FP':>6} {'TN':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = {}
    for thr in THRESHOLDS:
        m = metrics_at(y_test, proba, thr)
        rows[thr] = m
        star = "  <- current" if abs(thr - CURRENT) < 1e-9 else ""
        print(f"  {thr:>5.2f} {m['acc']:>7.4f} {m['sens']:>7.4f} {m['spec']:>7.4f} "
              f"{m['under']:>7.4f} {m['over']:>7.4f} {m['f1']:>7.4f} "
              f"{m['youden']:>7.4f}   {m['tp']:>6} {m['fn']:>6} {m['fp']:>6} "
              f"{m['tn']:>6}{star}")

    # Deltas vs current 0.50
    base = rows[CURRENT]
    print(f"\n  Deltas vs current threshold {CURRENT:.2f} "
          f"(positive under/over = worse):")
    print(f"  {'thr':>5} {'Δacc':>8} {'Δsens':>8} {'Δspec':>8} {'Δunder':>8} "
          f"{'Δover':>8}   {'extra_admits_caught':>20} {'extra_false_admits':>19}")
    for thr in THRESHOLDS:
        if thr == CURRENT:
            continue
        m = rows[thr]
        caught = m['tp'] - base['tp']      # additional true-admits caught
        false = m['fp'] - base['fp']       # additional false admits
        print(f"  {thr:>5.2f} {m['acc']-base['acc']:>+8.4f} "
              f"{m['sens']-base['sens']:>+8.4f} {m['spec']-base['spec']:>+8.4f} "
              f"{m['under']-base['under']:>+8.4f} {m['over']-base['over']:>+8.4f}   "
              f"{caught:>+20,} {false:>+19,}")

    # Criterion-based picks
    print(f"\n  Threshold that maximizes each criterion (on this test split):")
    best_f1 = max(THRESHOLDS, key=lambda t: rows[t]['f1'])
    best_y = max(THRESHOLDS, key=lambda t: rows[t]['youden'])
    best_acc = max(THRESHOLDS, key=lambda t: rows[t]['acc'])
    print(f"    max F1        : {best_f1:.2f}  (F1={rows[best_f1]['f1']:.4f})")
    print(f"    max Youden's J: {best_y:.2f}  (J={rows[best_y]['youden']:.4f})")
    print(f"    max accuracy  : {best_acc:.2f}  (acc={rows[best_acc]['acc']:.4f})")
    print(f"\n  'best accuracy' is biased toward the majority (discharge) class; "
          f"for a clinical disposition, under-triage (missed admits) is the "
          f"costlier error, so F1 / Youden / a target under-triage rate are the "
          f"clinically appropriate selectors.")

    print("  No model was retrained; only the cutoff varied.")


if __name__ == "__main__":
    main()
