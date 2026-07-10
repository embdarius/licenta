"""Generate the architecture / flow diagrams for the thesis (Cap. 2, 4, 5).

Pure-matplotlib box-and-arrow diagrams rendered to
``thesis_article/template/figs/`` as PNG, matching the filenames the LaTeX
placeholders reference. Analogous to ``benchmarks/make_ch6_figures.py`` (which
renders the Cap. 6 result plots). No system Graphviz needed.

Run:
    uv run --with matplotlib python benchmarks/make_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parent.parent / "thesis_article" / "template" / "figs"

# (fill, edge) palette - soft fills, darker edges.
C = {
    "user":    ("#EAEAEA", "#555555"),
    "parser":  ("#DAE8FC", "#3B6EA5"),
    "triaj":   ("#F8CECC", "#B85450"),
    "doctor":  ("#E1D5E7", "#7D5BA6"),
    "nurse":   ("#D5E8D4", "#5E8C4A"),
    "data":    ("#FFF2CC", "#B8860B"),
    "pred":    ("#FDEBD0", "#CA7C1B"),
    "neutral": ("#F5F5F5", "#666666"),
    "server":  ("#DDEBF7", "#2E6DA4"),
}
FONT = "DejaVu Sans"
INK = "#222222"


def rbox(ax, cx, cy, w, h, text, kind="neutral", fs=10, bold=False,
         tc=INK, rounding=0.6, lw=1.4):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            family=FONT, color=tc, fontweight="bold" if bold else "normal",
            zorder=3, linespacing=1.25)
    return (cx, cy, w, h)


def sqbox(ax, cx, cy, w, h, text, kind="neutral", fs=10, bold=False, tc=INK, lw=1.4):
    """Sharp-cornered box (for UML / tables)."""
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h, boxstyle="square,pad=0",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, family=FONT,
            color=tc, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.25)
    return (cx, cy, w, h)


def arrow(ax, p0, p1, color="#444444", lw=1.6, ls="-", style="-|>",
          conn="arc3,rad=0", mut=13):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=mut, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=conn, zorder=1,
        shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=8.5, color="#555555", ha="center", va="center",
          italic=False, bold=False):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, family=FONT, color=color,
            style="italic" if italic else "normal",
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    print("wrote", p)


# 1. Cap. 2 - obiective_flux.png : patient-journey overview (user perspective)
def fig_obiective_flux():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    stages = [
        ("Pacient\n(descriere în\nlimbaj natural)", "user"),
        ("Parser NLP\n(LLM)", "parser"),
        ("Triaj\n(acuitate +\ndispoziție)", "triaj"),
        ("Doctor\n(evaluare\ninițială)", "doctor"),
        ("Asistent\n(vitale, medicație,\nantecedente)", "nurse"),
        ("Doctor\n(dispoziție +\nreevaluare)", "doctor"),
    ]
    n = len(stages)
    w, h = 14.0, 15
    xs = [8.5 + i * 16.6 for i in range(n)]
    y = 26
    for (txt, kind), x in zip(stages, xs):
        rbox(ax, x, y, w, h, txt, kind=kind, fs=9.5, bold=True)
    for i in range(n - 1):
        arrow(ax, (xs[i] + w / 2, y), (xs[i + 1] - w / 2, y))
    label(ax, xs[0], y - h / 2 - 4.2,
          "„Bărbat de 68 de ani cu\ndurere în piept de o oră…”", italic=True, fs=8)
    label(ax, xs[-1], y - h / 2 - 4.2,
          "Raport: acuitate, internare,\ndiagnostic, departament,\ncod ICD", fs=8)
    # bracket over the two doctor passes
    bx0, bx1, by = xs[3] - w / 2, xs[5] + w / 2, y + h / 2 + 2.0
    ax.plot([bx0, bx0, bx1, bx1], [by, by + 2.2, by + 2.2, by],
            color=C["triaj"][1], lw=1.3, zorder=1)
    label(ax, (bx0 + bx1) / 2, by + 3.6,
          "evaluarea doctorului: înainte și după datele asistentului",
          color=C["triaj"][1], fs=8.5, bold=True)
    save(fig, "obiective_flux.png")


# 2. Cap. 4 - arhitectura_logica.png : agents, 6 tasks, prediction cascade
def fig_arhitectura_logica():
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")

    tasks = [
        ("1. Parsare\nlimbaj natural", "parser"),
        ("2. Triaj", "triaj"),
        ("3. Evaluare\ninițială", "doctor"),
        ("4. Colectare\ndate", "nurse"),
        ("5. Dispoziție\nrafinată", "doctor"),
        ("6. Reevaluare", "doctor"),
    ]
    n = len(tasks)
    w, h = 13.5, 12
    xs = [8.5 + i * 16.6 for i in range(n)]
    ty = 50
    for (txt, kind), x in zip(tasks, xs):
        rbox(ax, x, ty, w, h, txt, kind=kind, fs=9.5, bold=True)
    for i in range(n - 1):
        arrow(ax, (xs[i] + w / 2, ty), (xs[i + 1] - w / 2, ty))
    label(ax, 50, 61, "Cele șase sarcini executate de cei patru agenți",
          fs=10.5, bold=True, color=INK)

    # prediction cascade
    preds = ["Acuitate\n(ESI 1–5)", "Dispoziție\n(internare)",
             "Diagnostic\n(13 categorii)", "Departament\n(11 servicii)",
             "Cod ICD\n(exact)"]
    pw, ph = 15.0, 10
    pxs = [11 + i * 19.5 for i in range(len(preds))]
    py = 20
    for txt, x in zip(preds, pxs):
        rbox(ax, x, py, pw, ph, txt, kind="pred", fs=9.5, bold=True)
    for i in range(len(preds) - 1):
        arrow(ax, (pxs[i] + pw / 2, py), (pxs[i + 1] - pw / 2, py),
              color=C["pred"][1], lw=1.8)
    label(ax, 50, 31, "Cascada de predicții (fiecare ieșire alimentează etapa următoare)",
          fs=10.5, bold=True, color=C["pred"][1])
    # thin links task -> prediction it produces
    link = dict(color="#999999", lw=1.0, ls=(0, (3, 3)), style="-|>", mut=10)
    arrow(ax, (xs[1], ty - h / 2), (pxs[0], py + ph / 2), **link)          # triaj -> acuitate
    arrow(ax, (xs[1], ty - h / 2), (pxs[1], py + ph / 2), **link)          # triaj -> dispozitie (screening)
    arrow(ax, (xs[2], ty - h / 2), (pxs[2], py + ph / 2),                  # doctor initial -> diagnostic
          conn="arc3,rad=0.12", **link)
    arrow(ax, (xs[2], ty - h / 2), (pxs[3], py + ph / 2),                  # doctor initial -> departament
          conn="arc3,rad=0.12", **link)
    arrow(ax, (xs[4], ty - h / 2), (pxs[1], py + ph / 2), **link)          # dispozitie rafinata
    arrow(ax, (xs[5], ty - h / 2), (pxs[2], py + ph / 2), **link)          # reevaluare -> diagnostic
    arrow(ax, (xs[5], ty - h / 2), (pxs[3], py + ph / 2), **link)          # reevaluare -> departament
    arrow(ax, (xs[5], ty - h / 2), (pxs[4], py + ph / 2), **link)          # reevaluare -> cod ICD
    label(ax, 50, 13, "diagnosticul și departamentul sunt prezise de două ori: "
          "inițial (v3_base) și după asistent", fs=8, color="#777777",
          ha="center", italic=True)

    # legend (agent colours)
    leg = [("Parser NLP (LLM)", "parser"), ("Agent triaj", "triaj"),
           ("Agent doctor (×3)", "doctor"), ("Agent asistent", "nurse")]
    lx = 8
    for txt, kind in leg:
        ax.add_patch(FancyBboxPatch((lx, 2.2), 3.2, 3.2,
                     boxstyle="round,pad=0.02,rounding_size=0.4",
                     linewidth=1.2, edgecolor=C[kind][1], facecolor=C[kind][0]))
        label(ax, lx + 4.0, 3.8, txt, ha="left", fs=8.5, color=INK)
        lx += 23.5
    save(fig, "arhitectura_logica.png")


# 3. Cap. 5 - arhitectura_pachet.png : on-disk package structure
def fig_arhitectura_pachet():
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")

    # container package
    ax.add_patch(FancyBboxPatch((4, 20), 60, 48, boxstyle="square,pad=0",
                 linewidth=1.6, edgecolor="#3B6EA5", facecolor="#F3F7FC", zorder=1))
    ax.add_patch(FancyBboxPatch((4, 64), 26, 5, boxstyle="square,pad=0",
                 linewidth=1.6, edgecolor="#3B6EA5", facecolor="#DAE8FC", zorder=1))
    label(ax, 17, 66.5, "src/proiect_licenta", fs=10, bold=True, color=INK)

    mods = ["main", "crew", "paths", "preprocessing", "llm_config", "interaction"]
    mw, mh = 16, 5.5
    for i, m in enumerate(mods):
        r, c = divmod(i, 2)
        sqbox(ax, 15 + c * 18, 58 - r * 8.0, mw, mh, m, kind="neutral", fs=9)
    label(ax, 34, 62.5, "module de nivel superior", fs=8.5, color="#3B6EA5")

    # training placed rightmost so its "writes / reads" arrows reach the external
    # stores without passing behind another subpackage box.
    subs = [("tools", "instrumente CrewAI"),
            ("config", "agents.yaml /\ntasks.yaml"),
            ("training", "pipeline-uri\nde antrenare")]
    subx = [16, 34, 52]
    for (name, desc), cx in zip(subs, subx):
        sqbox(ax, cx, 30, 17, 9.5, name + "\n" + desc, kind="server", fs=8.5)
    label(ax, 34, 37.5, "subpachete", fs=8.5, color="#3B6EA5")

    # external artifact / data stores
    art_t = sqbox(ax, 82, 58, 26, 7, "artifacts/triage\n{v1, v2, v3}", kind="data", fs=9)
    art_d = sqbox(ax, 82, 47, 26, 7, "artifacts/doctor\n{v1, v2, v3_base, v3}", kind="data", fs=9)
    data = sqbox(ax, 82, 27, 26, 8, "data/mimic-iv-ed\n(CSV brute)", kind="data", fs=9)

    # relations: training (rightmost subpackage) writes the models and reads data
    arrow(ax, (60.5, 32.5), (70, 57), color="#B8860B", lw=1.5, conn="arc3,rad=-0.22")
    arrow(ax, (60.5, 31), (70, 47), color="#B8860B", lw=1.5, conn="arc3,rad=-0.12")
    label(ax, 65.5, 40, "scrie\nmodele", fs=8, color="#B8860B")
    arrow(ax, (60.5, 28), (70, 28), color="#B8860B", lw=1.5)
    label(ax, 65, 25.5, "citește", fs=8, color="#B8860B")
    # tools load the trained models at inference (arcs above config + training)
    arrow(ax, (24.5, 34), (69, 45.5), color="#7D5BA6", lw=1.4, ls=(0, (4, 3)),
          conn="arc3,rad=-0.12")
    label(ax, 55, 45.5, "tools încarcă\nmodelele", fs=8, color="#7D5BA6")
    save(fig, "arhitectura_pachet.png")


# 4. Cap. 5 - flux_executie.png : UML-ish sequence of the 6 tasks
def fig_flux_executie():
    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 84); ax.axis("off")

    actors = [("Utilizator", "user", 10), ("Parser", "parser", 30),
              ("Triaj", "triaj", 50), ("Doctor", "doctor", 70),
              ("Asistent", "nurse", 90)]
    top, bot = 78, 6
    xof = {}
    for name, kind, x in actors:
        rbox(ax, x, top, 15, 5.5, name, kind=kind, fs=10, bold=True)
        ax.plot([x, x], [top - 3, bot], color="#AAAAAA", lw=1.1, ls=(0, (4, 3)), zorder=0)
        xof[name] = x

    def msg(y, a, b, text, dashed=False, color="#444444"):
        x0, x1 = xof[a], xof[b]
        arrow(ax, (x0, y), (x1, y), color=color, lw=1.5,
              ls=(0, (4, 3)) if dashed else "-", style="-|>")
        mid = (x0 + x1) / 2
        label(ax, mid, y + 1.8, text, fs=8, color=INK)

    def selfmsg(y, a, text, color="#7D5BA6"):
        x = xof[a]
        ax.add_patch(FancyArrowPatch((x, y + 1.2), (x, y - 1.2),
                     arrowstyle="-|>", mutation_scale=11, color=color, lw=1.4,
                     connectionstyle="arc3,rad=-2.6", zorder=1))
        label(ax, x + 2, y, text, fs=8, color=INK, ha="left")

    msg(72, "Utilizator", "Parser", "descriere în limbaj natural")
    msg(66, "Parser", "Utilizator", "confirmare intake (formular editabil)", dashed=True)
    msg(60, "Parser", "Triaj", "STRUCTURED_DATA")
    msg(54, "Triaj", "Doctor", "acuitate + dispoziție (screening)")
    selfmsg(48, "Doctor", "evaluare inițială (v3_base):\ndiagnostic + departament")
    msg(41, "Doctor", "Asistent", "cerere de colectare a datelor")
    msg(35, "Asistent", "Utilizator", "întrebări: vitale, medicație, antecedente")
    msg(29, "Asistent", "Doctor", "date colectate", dashed=True)
    selfmsg(23, "Doctor", "dispoziție rafinată\n(model calibrat)")
    selfmsg(15, "Doctor", "reevaluare: diagnostic +\ndepartament + cod ICD")

    # conditioning note
    ax.add_patch(FancyBboxPatch((60, 8.5), 38, 4.4,
                 boxstyle="round,pad=0.15,rounding_size=0.4", linewidth=1.2,
                 edgecolor="#CA7C1B", facecolor="#FDEBD0", zorder=2))
    label(ax, 79, 10.7, "reevaluarea completă are loc doar dacă\ndispoziția rafinată = internare",
          fs=8, color="#8a5a12")
    save(fig, "flux_executie.png")


# 5. Cap. 5 - diagrama_clase_tools.png : UML class diagram of the tools
def fig_diagrama_clase_tools():
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 130); ax.set_ylim(0, 76); ax.axis("off")

    def umlclass(cx, cy, w, name, schema, kind="neutral", fs=8):
        h1, h2, h3 = 4.2, 3.6, 3.6
        h = h1 + h2 + h3
        top = cy + h / 2
        fill, edge = C[kind]
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="square,pad=0", linewidth=1.3, edgecolor=edge,
                     facecolor=fill, zorder=2))
        ax.plot([cx - w / 2, cx + w / 2], [top - h1, top - h1], color=edge, lw=1.0, zorder=3)
        ax.plot([cx - w / 2, cx + w / 2], [top - h1 - h2, top - h1 - h2], color=edge, lw=1.0, zorder=3)
        ax.text(cx, top - h1 / 2, name, ha="center", va="center", fontsize=fs,
                family=FONT, fontweight="bold", color=INK, zorder=4)
        ax.text(cx, top - h1 - h2 / 2, schema, ha="center", va="center",
                fontsize=fs - 0.8, family=FONT, color="#444444", zorder=4)
        ax.text(cx, top - h1 - h2 - h3 / 2, "_run()", ha="center", va="center",
                fontsize=fs - 0.8, family=FONT, color="#444444", zorder=4)
        return (cx, cy, w, h)

    # parent
    base = umlclass(65, 68, 26, "BaseTool  (CrewAI)", "name, description", kind="server", fs=9)

    subs = [
        ("TriagePredictionTool", "args: TriageInput", "triaj"),
        ("DoctorPredictionToolV3Base", "args: DoctorV3Input", "doctor"),
        ("DoctorDispositionTool", "args: DispositionInput", "doctor"),
        ("DoctorPredictionToolV3", "args: DoctorV3Input", "doctor"),
        ("NurseDataCollectionTool", "args: NurseInput", "nurse"),
        ("AskPatientTool", "args: AskInput", "parser"),
        ("ConfirmIntakeTool", "args: IntakeInput", "parser"),
        ("PatientHistoryLookupTool", "args: MrnInput", "data"),
    ]
    w = 29
    xs = [17 + i * 32 for i in range(4)]
    rowy = [40, 16]
    bus_y = 55
    ax.plot([base[0], base[0]], [68 - base[3] / 2, bus_y], color="#666666", lw=1.2, zorder=1)
    ax.plot([xs[0], xs[-1]], [bus_y, bus_y], color="#666666", lw=1.2, zorder=1)
    # hollow inheritance triangle at parent
    ax.add_patch(Polygon([[base[0] - 1.6, bus_y], [base[0] + 1.6, bus_y],
                          [base[0], bus_y + 2.4]], closed=True, facecolor="white",
                          edgecolor="#666666", lw=1.2, zorder=3))

    for i, (name, schema, kind) in enumerate(subs):
        r, c = divmod(i, 4)
        cx, cy = xs[c], rowy[r]
        box = umlclass(cx, cy, w, name, schema, kind=kind, fs=8)
        ax.plot([cx, cx], [bus_y, cy + box[3] / 2], color="#666666", lw=1.2, zorder=1)

    label(ax, 108, 66, "toate instrumentele\nextind BaseTool și își\ndeclară argumentele\nprintr-un model Pydantic",
          fs=8.5, color="#555555", ha="left")
    save(fig, "diagrama_clase_tools.png")


# 6. Cap. 5 - model_date.png : MIMIC tables -> feature blocks -> labels
def fig_model_date():
    fig, ax = plt.subplots(figsize=(12.5, 8.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 82); ax.axis("off")

    tables = ["triage", "edstays", "diagnosis", "services", "patients",
              "vitalsign", "medrecon", "diagnoses_icd", "discharge"]
    blocks = ["Structurate\n(vârstă, sosire)", "Semne vitale", "Vitale\nlongitudinale",
              "Medicație", "Antecedente\n(PMH)", "Text TF-IDF\n(motiv prezentare)"]
    labels = ["Acuitate", "Dispoziție", "Diagnostic", "Departament", "Cod ICD"]

    tx, bx, lx = 14, 50, 86
    def col(items, x, kind, w, top, gap):
        ys = []
        for i, it in enumerate(items):
            y = top - i * gap
            sqbox(ax, x, y, w, gap * 0.66, it, kind=kind, fs=8.3)
            ys.append(y)
        return ys
    ty = col(tables, tx, "data", 20, 76, 8.2)
    by = col(blocks, bx, "neutral", 21, 71, 11.5)
    ly = col(labels, lx, "pred", 17, 70, 13.5)

    label(ax, tx, 81, "Tabele MIMIC-IV-ED", fs=10, bold=True, color=INK)
    label(ax, bx, 81, "Blocuri de caracteristici", fs=10, bold=True, color=INK)
    label(ax, lx, 81, "Etichete", fs=10, bold=True, color=INK)

    g = dict(color="#B0B0B0", lw=1.0, style="-|>", mut=9)
    def link_tb(ti, bi):
        arrow(ax, (tx + 10, ty[ti]), (bx - 10.5, by[bi]), **g)
    link_tb(0, 0); link_tb(0, 1); link_tb(0, 5); link_tb(4, 0); link_tb(1, 0)
    link_tb(5, 2); link_tb(6, 3); link_tb(7, 4); link_tb(8, 4)

    # central models node: every feature block feeds the XGBoost models, which
    # emit every label (avoids implying a 1:1 block -> label mapping).
    ax.add_patch(FancyBboxPatch((64.5, 12), 9, 60,
                 boxstyle="round,pad=0.1,rounding_size=1.0", linewidth=1.5,
                 edgecolor="#CA7C1B", facecolor="#FDEBD0", zorder=2))
    ax.text(69, 42, "modele XGBoost", rotation=90, ha="center", va="center",
            fontsize=10.5, family=FONT, fontweight="bold", color="#8a5a12", zorder=3)
    for i in range(len(blocks)):
        arrow(ax, (bx + 10.5, by[i]), (64.5, by[i]),
              color="#B8860B", lw=1.0, style="-|>", mut=9)
    for i in range(len(labels)):
        arrow(ax, (73.5, ly[i]), (lx - 8.5, ly[i]),
              color="#CA7C1B", lw=1.0, style="-|>", mut=9)

    # anti-leakage note
    ax.add_patch(FancyBboxPatch((4, 1.5), 92, 4.2,
                 boxstyle="round,pad=0.15,rounding_size=0.4", linewidth=1.2,
                 edgecolor="#B85450", facecolor="#F8CECC", zorder=2))
    label(ax, 50, 3.6,
          "Filtrare temporală anti-scurgere: doar evenimente cu prior_*time < intime alimentează antecedentele",
          fs=8.5, color="#7a3532", bold=True)
    save(fig, "model_date.png")


# 7. Cap. 5 - interfata_web.png : browser <-> FastAPI <-> crew thread
def fig_interfata_web():
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 130); ax.set_ylim(0, 66); ax.axis("off")

    # browser
    rbox(ax, 20, 40, 34, 40, "", kind="parser", lw=1.6)
    label(ax, 20, 57, "Browser", fs=11, bold=True, color=INK)
    label(ax, 20, 53.5, "(React, Vite, Tailwind)", fs=8.5, color="#3B6EA5")
    for i, t in enumerate(["Transcrierea conversației", "Panoul celor 4 agenți",
                            "Widget-uri de întrebări", "Carduri de rezultate"]):
        sqbox(ax, 20, 47 - i * 7.2, 28, 5.4, t, kind="neutral", fs=8.3)

    # server
    rbox(ax, 65, 40, 26, 40, "", kind="server", lw=1.6)
    label(ax, 65, 57, "Server FastAPI", fs=11, bold=True, color=INK)
    label(ax, 65, 53.5, "webapp/backend", fs=8.5, color="#2E6DA4")
    sqbox(ax, 65, 46, 21, 6, "rute /api/live/\nstart · stream · answer", kind="neutral", fs=8.3)
    sqbox(ax, 65, 36, 21, 6, "listeneri pe\ncrewai_event_bus", kind="neutral", fs=8.3)

    # crew thread
    rbox(ax, 108, 40, 30, 40, "", kind="doctor", lw=1.6)
    label(ax, 108, 57, "Fir de execuție", fs=11, bold=True, color=INK)
    label(ax, 108, 53.5, "(per sesiune)", fs=8.5, color="#7D5BA6")
    sqbox(ax, 108, 47, 25, 6, "ProiectLicenta()\n.crew().kickoff()", kind="neutral", fs=8.3)
    sqbox(ax, 108, 38.5, 25, 6, "SessionChannel\n(ask blochează firul)", kind="neutral", fs=8.3)
    sqbox(ax, 108, 30, 25, 5.5, "instrumente interactive", kind="neutral", fs=8.3)

    # browser <-> server
    arrow(ax, (37, 43), (52, 43), color="#2E6DA4", lw=1.8, conn="arc3,rad=-0.25")
    label(ax, 44.5, 48, "flux SSE\n(evenimente, întrebări)", fs=8, color="#2E6DA4")
    arrow(ax, (52, 36), (37, 36), color="#3B6EA5", lw=1.8, conn="arc3,rad=-0.25")
    label(ax, 44.5, 31.5, "POST start /\nanswer", fs=8, color="#3B6EA5")

    # server <-> crew
    arrow(ax, (78, 43), (93, 43), color="#7D5BA6", lw=1.8, conn="arc3,rad=-0.22")
    label(ax, 85.5, 47.5, "pornește firul", fs=8, color="#7D5BA6")
    arrow(ax, (93, 36), (78, 36), color="#7D5BA6", lw=1.8, conn="arc3,rad=-0.22")
    label(ax, 85.5, 31.5, "evenimente\n+ întrebări", fs=8, color="#7D5BA6")

    label(ax, 65, 12,
          "Serverul rulează același echipaj ca varianta din linia de comandă;\n"
          "predicțiile sunt identice, diferă doar transportul interacțiunii.",
          fs=9, color="#555555", bold=False)
    save(fig, "interfata_web.png")


def main():
    fig_obiective_flux()
    fig_arhitectura_logica()
    fig_arhitectura_pachet()
    fig_flux_executie()
    fig_diagrama_clase_tools()
    fig_model_date()
    fig_interfata_web()
    print("\nAll diagrams written to", OUT)


if __name__ == "__main__":
    main()
