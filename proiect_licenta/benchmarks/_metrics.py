"""Shared metric helpers for the detailed v3 re-benchmark.

Every function returns plain JSON-serializable Python so the benchmark scripts
dump one auditable JSON and build CSVs from the same numbers. Functions are
defensive: on a degenerate input (single-class subgroup, missing class) AUC-style
metrics return None rather than raising, so the small generated-cases path and the
large tabular path share the same code.

Conventions:
- Multiclass labels are 0-indexed class ids 0..K-1 aligned with the columns of
  y_prob; label_names (optional) gives a string per id.
- Binary: positive class = 1 (admit for disposition). under_triage = admit
  predicted discharge (FN rate); over_triage = discharge predicted admit (FP rate).
- Ordinal (acuity): ESI integers, lower is more acute. under_triage = predicted
  less acute (pred > true); over_triage = predicted more acute (pred < true).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, precision_recall_fscore_support, log_loss,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)


# Low-level helpers
def _f(x):
    """Cast to float, mapping NaN/inf-unsafe values to None for clean JSON."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _safe_auc_binary(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return _f(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _safe_ap_binary(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return _f(average_precision_score(y_true, y_score))
    except Exception:
        return None


def brier_multiclass(y_true, y_prob, n_classes):
    """Mean squared error between one-hot truth and predicted probabilities,
    summed over classes (the multiclass generalization of the Brier score)."""
    onehot = np.zeros((len(y_true), n_classes), dtype=float)
    onehot[np.arange(len(y_true)), np.asarray(y_true, dtype=int)] = 1.0
    return _f(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def topk_accuracy(y_true, y_prob, k):
    """Top-k accuracy computed directly from the probability matrix (robust to a
    label set that doesn't span every column, unlike sklearn's helper)."""
    if y_prob is None or k > y_prob.shape[1]:
        return None
    topk = np.argsort(-y_prob, axis=1)[:, :k]
    hit = np.any(topk == np.asarray(y_true, dtype=int)[:, None], axis=1)
    return _f(np.mean(hit))


def mrr(y_true, y_prob):
    """Mean reciprocal rank of the true class within each row's ranked probs."""
    if y_prob is None:
        return None
    order = np.argsort(-y_prob, axis=1)
    yt = np.asarray(y_true, dtype=int)
    ranks = np.empty(len(yt), dtype=float)
    for i in range(len(yt)):
        pos = np.where(order[i] == yt[i])[0]
        ranks[i] = 1.0 / (pos[0] + 1) if len(pos) else 0.0
    return _f(np.mean(ranks))


def rank_of_true(y_true_row, prob_row):
    """1-based rank of the true class in one row's probability vector."""
    order = np.argsort(-np.asarray(prob_row))
    pos = np.where(order == int(y_true_row))[0]
    return int(pos[0] + 1) if len(pos) else None


# Calibration
def calibration_report(y_true, y_prob, n_bins: int = 10):
    """Reliability diagram + ECE/MCE/Brier for a binary positive-class score.

    Returns ``{ece, mce, brier, n_bins, bins:[{bin_lo,bin_hi,bin_center,
    n,mean_predicted,observed_freq,gap}]}``. Equal-width bins on [0,1]; bins with
    no samples are omitted.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, edges[1:-1])
    bins = []
    ece = 0.0
    mce = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf = float(y_prob[mask].mean())
        obs = float(y_true[mask].mean())
        gap = abs(conf - obs)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        bins.append({
            "bin_lo": _f(edges[b]), "bin_hi": _f(edges[b + 1]),
            "bin_center": _f((edges[b] + edges[b + 1]) / 2),
            "n": cnt, "mean_predicted": _f(conf),
            "observed_freq": _f(obs), "gap": _f(gap),
        })
    brier = _f(np.mean((y_prob - y_true) ** 2))
    return {"ece": _f(ece), "mce": _f(mce), "brier": brier,
            "n_bins": n_bins, "bins": bins}


def ece_top_label(y_true, y_pred, y_prob, n_bins: int = 10):
    """Top-label ECE for a multiclass classifier (confidence = max prob)."""
    if y_prob is None:
        return None
    conf = np.max(y_prob, axis=1)
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(conf[mask].mean() - correct[mask].mean())
    return _f(ece)


# Binary report (disposition heads)
def _binary_at_threshold(y_true, y_prob, threshold):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    sens = tp / max(tp + fn, 1)          # admit recall
    spec = tn / max(tn + fp, 1)          # discharge recall
    ppv = tp / max(tp + fp, 1)           # admit precision
    npv = tn / max(tn + fn, 1)
    f1_pos = 2 * ppv * sens / max(ppv + sens, 1e-12)
    f1_neg = 2 * npv * spec / max(npv + spec, 1e-12)
    try:
        mcc = _f(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = None
    return {
        "threshold": _f(threshold),
        "accuracy": _f((tp + tn) / n),
        "sensitivity": _f(sens), "specificity": _f(spec),
        "ppv": _f(ppv), "npv": _f(npv),
        "f1_admit": _f(f1_pos), "f1_discharge": _f(f1_neg),
        "balanced_accuracy": _f((sens + spec) / 2),
        "mcc": mcc,
        "under_triage": _f(fn / max(fn + tp, 1)),   # admit -> discharge
        "over_triage": _f(fp / max(fp + tn, 1)),    # discharge -> admit
        "predicted_admit_rate": _f((tp + fp) / n),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def binary_report(y_true, y_prob, thresholds=(0.5,), n_bins: int = 10):
    """Full report for a binary admit/discharge head.

    ``thresholds`` may be any iterable; the per-threshold table carries every
    confusion-derived metric. Threshold-independent metrics (AUC, PR-AUC, Brier,
    log-loss, calibration) are reported once.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    try:
        ll = _f(log_loss(y_true, np.clip(y_prob, 1e-12, 1 - 1e-12), labels=[0, 1]))
    except Exception:
        ll = None
    cal = calibration_report(y_true, y_prob, n_bins=n_bins)
    sweep = [_binary_at_threshold(y_true, y_prob, t) for t in thresholds]
    return {
        "n": int(len(y_true)),
        "prevalence_admit": _f(y_true.mean()),
        "roc_auc": _safe_auc_binary(y_true, y_prob),
        "pr_auc": _safe_ap_binary(y_true, y_prob),
        "brier": cal["brier"],
        "log_loss": ll,
        "ece": cal["ece"], "mce": cal["mce"],
        "reliability_bins": cal["bins"],
        "thresholds": sweep,
    }


def threshold_sweep(y_true, y_prob, thresholds):
    """Just the per-threshold table (list of dicts) - for CSV export."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return [_binary_at_threshold(y_true, y_prob, t) for t in thresholds]


# Multiclass report (diagnosis / department)
def per_class_table(y_true, y_pred, labels, label_names=None):
    """Per-class precision / recall / specificity / f1 / support / accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    total = len(y_true)
    for i, lab in enumerate(labels):
        tp = int(np.sum((y_true == lab) & (y_pred == lab)))
        fp = int(np.sum((y_true != lab) & (y_pred == lab)))
        fn = int(np.sum((y_true == lab) & (y_pred != lab)))
        tn = total - tp - fp - fn
        spec = tn / max(tn + fp, 1)
        rows.append({
            "label": int(lab),
            "name": (label_names[i] if label_names is not None else str(lab)),
            "precision": _f(prec[i]), "recall": _f(rec[i]),
            "specificity": _f(spec), "f1": _f(f1[i]),
            "support": int(sup[i]), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return rows


def per_class_auc(y_true, y_prob, labels):
    """One-vs-rest AUC per class (None where a class has no positives)."""
    if y_prob is None:
        return {}
    y_true = np.asarray(y_true, dtype=int)
    out = {}
    for i, lab in enumerate(labels):
        out[int(lab)] = _safe_auc_binary((y_true == lab).astype(int), y_prob[:, i])
    return out


def confusion_frame(y_true, y_pred, labels, label_names=None):
    """Confusion matrix as nested lists (counts + row-normalized) + labels, for
    both JSON embedding and CSV export."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    names = label_names if label_names is not None else [str(l) for l in labels]
    return {
        "labels": [int(l) for l in labels],
        "label_names": names,
        "counts": cm.astype(int).tolist(),
        "row_normalized": np.round(norm, 6).tolist(),
    }


def multiclass_report(y_true, y_pred, y_prob, labels, label_names=None,
                      topk=(1, 3, 5), y_train=None):
    """Comprehensive multiclass report (diagnosis / department heads).

    ``y_train`` (optional) enables the majority-class and weighted-random
    baselines. ``labels`` are the 0-indexed class ids; ``y_prob`` columns align
    with them.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n_classes = len(labels)

    topk_acc = {f"top{k}": topk_accuracy(y_true, y_prob, k) for k in topk}
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    try:
        ll = _f(log_loss(y_true, y_prob, labels=list(range(n_classes)))) \
            if y_prob is not None else None
    except Exception:
        ll = None

    macro_auc = None
    if y_prob is not None:
        aucs = [v for v in per_class_auc(y_true, y_prob, labels).values()
                if v is not None]
        macro_auc = _f(np.mean(aucs)) if aucs else None

    report = {
        "n": int(len(y_true)),
        "n_classes": n_classes,
        "accuracy": _f(accuracy_score(y_true, y_pred)),
        **topk_acc,
        "mrr": mrr(y_true, y_prob),
        "cohen_kappa": _f(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": _f(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": _f(macro[0]), "recall_macro": _f(macro[1]),
        "f1_macro": _f(macro[2]),
        "precision_weighted": _f(weighted[0]), "recall_weighted": _f(weighted[1]),
        "f1_weighted": _f(weighted[2]),
        "log_loss": ll,
        "brier_multiclass": (brier_multiclass(y_true, y_prob, n_classes)
                             if y_prob is not None else None),
        "ece_top_label": ece_top_label(y_true, y_pred, y_prob),
        "macro_auc_ovr": macro_auc,
        "per_class_auc_ovr": per_class_auc(y_true, y_prob, labels),
        "per_class": per_class_table(y_true, y_pred, labels, label_names),
    }

    # Accuracy excluding the majority class (by support in y_true).
    counts = np.bincount(y_true, minlength=n_classes)
    maj = int(np.argmax(counts))
    mask = y_true != maj
    report["majority_class"] = maj
    report["majority_class_name"] = (label_names[maj] if label_names else str(maj))
    report["accuracy_excl_majority"] = (
        _f(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else None)

    if y_train is not None:
        y_train = np.asarray(y_train, dtype=int)
        tr_counts = np.bincount(y_train, minlength=n_classes)
        maj_tr = int(np.argmax(tr_counts))
        report["majority_baseline_acc"] = _f(np.mean(y_true == maj_tr))
        priors = tr_counts / max(tr_counts.sum(), 1)
        rng = np.random.RandomState(42)
        rand = rng.choice(n_classes, size=len(y_true), p=priors)
        report["weighted_random_baseline_acc"] = _f(np.mean(y_true == rand))
    return report


# Ordinal report (triage acuity)
def ordinal_report(y_true, y_pred, y_prob=None, levels=(1, 2, 3, 4, 5),
                   critical_levels=(1, 2)):
    """Acuity (ESI) report. ``y_pred``/``y_true`` are the ESI integers; lower =
    more acute. ``y_prob`` columns align with ``levels`` (ascending)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    err = y_pred - y_true                      # >0 under-triage, <0 over-triage
    n = len(y_true)

    # top-k from probabilities (probs columns are levels in ascending order)
    topk = {}
    if y_prob is not None:
        # map true ESI -> column index
        lvl_to_col = {lvl: i for i, lvl in enumerate(levels)}
        y_true_col = np.array([lvl_to_col[v] for v in y_true])
        for k in (2, 3):
            topk[f"top{k}"] = topk_accuracy(y_true_col, y_prob, k)

    crit_mask = np.isin(y_true, critical_levels)
    crit_under = (np.isin(y_true, critical_levels)
                  & ~np.isin(y_pred, critical_levels))
    report = {
        "n": int(n),
        "exact": _f(accuracy_score(y_true, y_pred)),
        **topk,
        "within_1": _f(np.mean(np.abs(err) <= 1)),
        "within_2": _f(np.mean(np.abs(err) <= 2)),
        "mae": _f(np.mean(np.abs(err))),
        "mean_signed_error": _f(np.mean(err)),
        "kappa_quadratic": _f(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "kappa_linear": _f(cohen_kappa_score(y_true, y_pred, weights="linear")),
        "kappa_unweighted": _f(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": _f(balanced_accuracy_score(y_true, y_pred)),
        "under_triage_rate": _f(np.mean(err > 0)),
        "over_triage_rate": _f(np.mean(err < 0)),
        "critical_under_triage_rate_overall": _f(np.mean(crit_under)),
        "critical_under_triage_rate_among_critical": (
            _f(crit_under.sum() / crit_mask.sum()) if crit_mask.any() else None),
        "macro_auc_ovr": None,
        "log_loss": None,
        "ece_top_label": None,
        "per_class": [],
        "pred_distribution": {int(l): int(np.sum(y_pred == l)) for l in levels},
        "true_distribution": {int(l): int(np.sum(y_true == l)) for l in levels},
    }
    # Per-class via the multiclass per_class_table on the raw ESI ids.
    report["per_class"] = per_class_table(
        y_true, y_pred, list(levels), [f"ESI {l}" for l in levels])
    if y_prob is not None:
        lvl_to_col = {lvl: i for i, lvl in enumerate(levels)}
        y_true_col = np.array([lvl_to_col[v] for v in y_true])
        y_pred_col = np.array([lvl_to_col[v] for v in y_pred])
        report["ece_top_label"] = ece_top_label(y_true_col, y_pred_col, y_prob)
        report["log_loss"] = (
            _f(log_loss(y_true_col, y_prob, labels=list(range(len(levels)))))
            if y_prob is not None else None)
        report["brier_multiclass"] = brier_multiclass(y_true_col, y_prob, len(levels))
        aucs = [v for v in per_class_auc(y_true_col, y_prob,
                                         list(range(len(levels)))).values()
                if v is not None]
        report["macro_auc_ovr"] = _f(np.mean(aucs)) if aucs else None
    return report


# Simple accuracy with coverage (generated-cases heads)
def accuracy_with_coverage(hits, n_eligible):
    """``hits`` = list of 0/1 over produced predictions; coverage = produced /
    eligible. Returns accuracy over eligible (missing = miss) and over produced."""
    produced = len(hits)
    correct = int(sum(hits))
    return {
        "n_eligible": int(n_eligible),
        "n_produced": produced,
        "coverage": _f(produced / n_eligible) if n_eligible else None,
        "accuracy_over_eligible": _f(correct / n_eligible) if n_eligible else None,
        "accuracy_over_produced": _f(correct / produced) if produced else None,
    }
