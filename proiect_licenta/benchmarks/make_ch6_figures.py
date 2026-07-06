"""Render the figures for thesis Chapter 6 (Testare si validare).

Every figure is built from the authoritative audit artifacts under
``benchmarks/audit/2026-06-28/`` (CSV/JSON already computed by
``benchmark_tabular_full.py`` and ``benchmark_pipeline_e2e.py``). Nothing is
recomputed from models and nothing is invented: each plot reads one of the
per-class / confusion / threshold / calibration / summary tables and draws it.

Outputs 7 PNGs into ``thesis_article/template/figs/``:
  triaj_confuzie.png      ESI 5x5 row-normalized confusion matrix
  triaj_per_clasa.png     ESI per-class precision / recall / F1
  dispozitie_prag.png     doctor disposition: acc / under- / over-triage vs threshold
  doctor_nurse_bar.png    diagnosis + department top-1/3/5, initial vs post-nurse
  doctor_calibrare.png    reliability diagram: doctor disposition vs triage disposition
  icd_graded.png          Stage-2 ICD: strict vs graded @5 (oracle, blend+vitals)
  backend_bar.png         Gemini Flash 2.5 vs MedGemma on the shared parser metrics

Run:  uv run python benchmarks/make_ch6_figures.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "benchmarks" / "audit" / "2026-06-28"
TAB = AUDIT / "tabular"
GEN = AUDIT / "generated_cases"
OUT = ROOT / "thesis_article" / "template" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

# Colour-blind-safe palette (Okabe-Ito).
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"


def save(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
def fig_triaj_confuzie():
    df = pd.read_csv(TAB / "triage_acuity_confusion_norm.csv", index_col=0)
    M = df.values
    labels = list(df.columns)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Clasa prezisă")
    ax.set_ylabel("Clasa reală")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fracțiune pe rând")
    save(fig, "triaj_confuzie.png")


def fig_triaj_per_clasa():
    df = pd.read_csv(TAB / "triage_acuity_per_class.csv")
    x = np.arange(len(df))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w, df["precision"], w, label="Precizie", color=C_BLUE)
    ax.bar(x, df["recall"], w, label="Sensibilitate (recall)", color=C_ORANGE)
    ax.bar(x + w, df["f1"], w, label="Scor F1", color=C_GREEN)
    ax.set_xticks(x, df["name"])
    ax.set_ylabel("Valoare")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "triaj_per_clasa.png")


def fig_dispozitie_prag():
    df = pd.read_csv(TAB / "doctor_disposition_thresholds.csv").sort_values("threshold")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(df["threshold"], df["accuracy"], "-o", ms=3, color=C_BLUE, label="Acuratete")
    ax.plot(df["threshold"], df["under_triage"], "-s", ms=3, color=C_RED,
            label="Rata de sub-triaj")
    ax.plot(df["threshold"], df["over_triage"], "-^", ms=3, color=C_GREEN,
            label="Rata de supra-triaj")
    ax.axvline(0.40, color="grey", ls="--", lw=1)
    ax.text(0.415, 0.05, "prag live 0.40", rotation=90, va="bottom", fontsize=8,
            color="grey")
    ax.set_xlabel("Prag de decizie")
    ax.set_ylabel("Valoare")
    ax.set_ylim(0, 1)
    ax.legend(loc="center right", fontsize=8)
    save(fig, "dispozitie_prag.png")


def fig_doctor_nurse_bar():
    tf = json.loads((TAB / "tabular_full.json").read_text(encoding="utf-8"))
    dh = tf["doctor_heads"]

    def row(head):
        b, n = head["v3_base"], head["v3_nurse"]
        return ([b["accuracy"], b["top3"], b["top5"]],
                [n["accuracy"], n["top3"], n["top5"]])

    diag_b, diag_n = row(dh["diagnosis"])
    dept_b, dept_n = row(dh["department"])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, (b, n), title in [
        (axes[0], (diag_b, diag_n), "Diagnostic (13 clase)"),
        (axes[1], (dept_b, dept_n), "Departament (11 clase)"),
    ]:
        x = np.arange(3)
        w = 0.38
        ax.bar(x - w / 2, b, w, label="Inițial (v3-base)", color=C_BLUE)
        ax.bar(x + w / 2, n, w, label="Post-asistent (v3)", color=C_ORANGE)
        ax.set_xticks(x, ["Top-1", "Top-3", "Top-5"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        for xi, (bv, nv) in enumerate(zip(b, n)):
            ax.text(xi - w / 2, bv + 0.01, f"{bv*100:.1f}", ha="center", fontsize=7)
            ax.text(xi + w / 2, nv + 0.01, f"{nv*100:.1f}", ha="center", fontsize=7)
    axes[0].set_ylabel("Acuratețe")
    axes[0].legend(loc="lower right", fontsize=8)
    save(fig, "doctor_nurse_bar.png")


def fig_doctor_calibrare():
    doc = pd.read_csv(TAB / "doctor_disposition_calibration.csv")
    tri = pd.read_csv(TAB / "triage_disposition_calibration.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibrare perfectă")
    ax.plot(doc["mean_predicted"], doc["observed_freq"], "-o", ms=4, color=C_BLUE,
            label="Dispoziție doctor (ECE 0.004)")
    ax.plot(tri["mean_predicted"], tri["observed_freq"], "-s", ms=4, color=C_ORANGE,
            label="Dispoziție triaj (ECE 0.075)")
    ax.set_xlabel("Probabilitate prezisă (medie pe coș)")
    ax.set_ylabel("Frecvența observată a internării")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "doctor_calibrare.png")


def fig_icd_graded():
    df = pd.read_csv(TAB / "icd_resolution_summary.csv")
    o = df[(df["section"] == "oracle") & (df["granularity"] == "rollup")
           & (df["variant"] == "blend+vitals")]

    def val(metric):
        return float(o[o["metric"] == metric]["value"].iloc[0])

    names = ["Strict@1", "Strict@5", "TF-IDF\ngrad@5", "Gemini\ngrad@5", "Arbore ICD\ngrad@5"]
    vals = [val("oracle@1"), val("oracle@5"),
            val("graded:tfidf:graded@5"), val("graded:gemini:graded@5"),
            val("graded:tree:graded@5")]
    colors = [C_RED, C_RED, C_BLUE, C_GREEN, C_PURPLE]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Recall @k (oracle, rollup, blend+vitals)")
    ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}",
                ha="center", fontsize=8)
    save(fig, "icd_graded.png")


def fig_backend_bar():
    df = pd.read_csv(GEN / "comparison" / "backend_comparison_metrics.csv")

    def val(metric, col):
        return float(df[df["metric"] == metric][col].iloc[0])

    labels = ["Acuitate\nexactă", "Dispoziție\nacc@0.4", "Diag.\ntop-1",
              "Dept.\ntop-1", "ICD\nstrict@1", "Fidelitate\nsimptome"]
    metrics = ["detailed.acuity.exact", "basic.dispo", "basic.diag_top1",
               "basic.dept_top1", "basic.icd_top1", "nl_fidelity.complaint_jaccard_mean"]
    flash = [val(m, "flash") for m in metrics]
    med = [val(m, "medgemma") for m in metrics]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(x - w / 2, flash, w, label="Gemini Flash 2.5", color=C_BLUE)
    ax.bar(x + w / 2, med, w, label="MedGemma", color=C_ORANGE)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Valoare")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    for xi, (f, m) in enumerate(zip(flash, med)):
        ax.text(xi - w / 2, f + 0.01, f"{f:.2f}", ha="center", fontsize=7)
        ax.text(xi + w / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=7)
    save(fig, "backend_bar.png")


def main():
    print(f"Rendering Chapter 6 figures into {OUT.relative_to(ROOT)} ...")
    fig_triaj_confuzie()
    fig_triaj_per_clasa()
    fig_dispozitie_prag()
    fig_doctor_nurse_bar()
    fig_doctor_calibrare()
    fig_icd_graded()
    fig_backend_bar()
    print("Done.")


if __name__ == "__main__":
    main()
