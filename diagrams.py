"""Structural diagrams for the thesis deck (not data plots).

  fig_workflow()     the whole pipeline: done stages + what remains
  fig_risk_models()  candidate risk-model formulations, for discussion

Run from an activated env:  python diagrams.py
"""
from math import gamma as gamma_fn
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path("output/figures")

INK, MUTED = "#0b0b0b", "#52514e"
DYN, STAT = "#2a78d6", "#1baf7a"        # dynamic / static data streams
HUB, TODO = "#184f95", "#9a9a95"        # the oracle / not-yet-built

plt.rcParams.update({"font.size": 10, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {p}")


def _box(ax, x, y, w, h, title, what, keys, color, done=True, title_size=10.5):
    """A stage box: title, one line saying what the step does, then the numbers
    worth remembering. Dashed and greyed if the step is not built yet."""
    for fill in (True, False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=1.6, edgecolor=color,
            facecolor=(color if done else "white") if fill else "none",
            alpha=(0.13 if done else 1.0) if fill else 1.0,
            linestyle="-" if done else (0, (4, 3)), zorder=2 + fill))
    cx = x + w / 2
    ax.text(cx, y + h - 0.22, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=color, zorder=4)
    ax.text(cx, y + h - 0.55, what, ha="center", va="top", fontsize=8.8,
            color=MUTED, style="italic", zorder=4)
    for i, k in enumerate(keys):
        ax.text(cx, y + h - 0.88 - i * 0.30, k, ha="center", va="top",
                fontsize=8.7, fontweight="bold",
                color=INK if done else MUTED, zorder=4)


def _arrow(ax, p0, p1, color=MUTED, style="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color, linestyle=style,
                                 shrinkA=2, shrinkB=2, zorder=1))


def fig_workflow():
    fig, ax = plt.subplots(figsize=(13.5, 9.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(1.25, 11.4)
    ax.axis("off")

    ax.text(3.20, 11.2, "DYNAMIC DATA   (what cyclists measured)", ha="center",
            fontsize=11.5, fontweight="bold", color=DYN)
    ax.text(10.05, 11.2, "STATIC DATA   (what the street is)", ha="center",
            fontsize=11.5, fontweight="bold", color=STAT)

    # ---- dynamic stream -----------------------------------------------------
    _box(ax, 0.40, 9.65, 5.6, 1.15, "TASK 0 · Acquire",
         "pull every senseBox:bike reading in Münster from the API",
         ["328,523 readings · 74 boxes · Jul 2024 → Jul 2026"], DYN)
    _box(ax, 0.40, 8.15, 5.6, 1.15, "TASK 2a · Ride quality",
         "drop broken rides, never rides with many overtakes",
         ["863 → 652 rides kept  ·  2,510 km"], DYN)
    _box(ax, 0.40, 6.65, 5.6, 1.15, "TASK 2b · Extract overtakes",
         "an overtake = a burst of classifier confidence ≥ 0.5",
         ["2,206 events  ·  0.88 per km"], DYN)
    _box(ax, 0.40, 5.15, 5.6, 1.15, "TASK 3 · Map-match",
         "place each ride on the street graph as a directed path",
         ["97% of points placed · 98% of events linked"], DYN)

    # ---- static stream ------------------------------------------------------
    _box(ax, 7.25, 9.65, 5.6, 1.15, "TASK 1 · Network",
         "OSMnx cyclable network, split by direction and classified",
         ["96,238 directed edges  ·  11 riding regimes"], STAT)
    _box(ax, 7.25, 8.15, 5.6, 1.15, "TASK 1b · Covariates",
         "describe each street from sources independent of the riders",
         ["accidents · centrality · speed limit · AADT"], STAT)

    # ---- the hub ------------------------------------------------------------
    _box(ax, 2.65, 3.35, 8.1, 1.45, "TASK 4 · THE ORACLE   (screening frame)",
         "one row per directed edge — the whole city, ridden or not",
         ["84,183 never ridden · 10,759 ridden, no overtake · 1,296 with ≥ 1",
          "exposure per edge: traversals · rider-km · rider-hours"],
         HUB, title_size=11.5)

    # ---- remaining ----------------------------------------------------------
    _box(ax, 0.40, 1.35, 4.15, 1.35, "TASK 5 · Risk model",
         "turn counts into a rate per edge",
         ["test models progressively"], TODO, done=False)
    _box(ax, 4.72, 1.35, 4.15, 1.35, "TASK 6 · Sampling experiments",
         "thin the data on purpose, then refit",
         ["fewer edges · fewer revisits"], TODO, done=False)
    _box(ax, 9.05, 1.35, 4.15, 1.35, "TASK 7 · Sampling playbook",
         "what a future campaign must collect",
         ["edges per regime · revisits per edge",
          "generalize → Osnabrück · Stuttgart · Zürich"], TODO, done=False)

    _save(fig, "fig10_workflow.png")


def _gamma_inset(ax, x0, y0, w, h, color):
    """A small Gamma density inside a model box: what 'rates vary, Gamma-shaped'
    actually looks like. Shape 1.8, mean set to the observed 0.89 / rider-km."""
    k, mean = 1.8, 0.89
    theta = mean / k
    ins = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    x = np.linspace(1e-3, 3.2, 400)
    pdf = x ** (k - 1) * np.exp(-x / theta) / (gamma_fn(k) * theta ** k)
    ins.fill_between(x, pdf, color=color, alpha=0.18)
    ins.plot(x, pdf, color=color, lw=1.8)
    ins.axvline(mean, color=MUTED, ls="--", lw=1)
    ins.annotate(r"regime mean $\alpha_r/\beta_r$", (mean, pdf.max() * 0.86),
                 (mean + 0.16, pdf.max() * 0.98), fontsize=7.2, color=MUTED,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
    ins.text(0.30, pdf.max() * 0.30, "most edges\nsit low", fontsize=7.2, color=INK,
             ha="center", va="center")
    ins.annotate("a few are\nmuch riskier", (2.15, pdf.max() * 0.055),
                 (1.75, pdf.max() * 0.42), fontsize=7.2, color=INK, ha="center",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ins.set_xlabel(r"overtake rate  $\lambda_e$", fontsize=7.5, labelpad=1.5)
    ins.set_xlim(0, 3.2)
    ins.set_ylim(0, pdf.max() * 1.22)
    ins.set_yticks([])
    ins.set_xticks([0, 1, 2, 3])
    ins.tick_params(labelsize=7, length=0, pad=1.5)
    for s in ("top", "right", "left"):
        ins.spines[s].set_visible(False)
    ins.spines["bottom"].set_color(MUTED)
    ins.patch.set_alpha(0)


def fig_risk_models():
    """Candidate formulations for task 5, cheapest first. No title — the slide
    carries it. Each model states what it uses, assumes, gains and costs, with
    its own notation on the right and its sources set off underneath."""
    models = [
        ("M0 · Raw rate on each edge", "no pooling · trust each edge alone", DYN,
         r"$\hat{\lambda}_e = N_e \,/\, E_e$",
         [("Uses", "the edge's own counts only"),
          ("Assumes", "one edge's data suffices"),
          ("Pro", "no model, fully transparent"),
          ("Con", "89% of ridden edges have zero overtakes; one ride gives 1.0 or 0.0")],
         "",   # plain division — no method to attribute

         [(r"$N_e$", "overtakes on edge $e$"),
          (r"$E_e$", "exposure on $e$"),
          (r"$\hat{\lambda}_e$", "estimated rate")]),

        ("M1 · One pooled rate per riding regime",
         "complete pooling · trust only the regime", DYN,
         r"$\hat{\lambda}_r = \sum_{e \in r} N_e \;/\; \sum_{e \in r} E_e$",
         [("Uses", "all counts inside one regime"),
          ("Assumes", "one rate per infrastructure type"),
          ("Pro", "thousands of traversals per estimate; no distributional assumption"),
          ("Con", "no within-regime detail")],
         "Rao, J. N. K., & Molina, I. (2015). Small area estimation (2nd ed.). Wiley.",
         [(r"$r$", "riding regime (11 classes)"),
          (r"$e \in r$", "edges of that regime"),
          (r"$\hat{\lambda}_r$", "one rate per regime")]),

        ("M2 · Empirical-Bayes shrinkage (Gamma–Poisson)",
         "partial pooling · edge shrunk toward its regime", HUB,
         r"$N_e \sim \mathrm{Poisson}(\lambda_e E_e), \quad "
         r"\lambda_e \sim \mathrm{Gamma}(\alpha_r, \beta_r)"
         r"\;\Rightarrow\; \hat{\lambda}_e = \dfrac{N_e + \alpha_r}{E_e + \beta_r}$",
         [("Uses", "edge counts + Gamma fitted per regime"),
          ("Assumes", "rates vary between edges, Gamma-shaped"),
          ("Pro", "closed form; thin edges lean on their regime, rich edges on themselves"),
          ("Con", "per-regime Gamma is shaky when most edges have ≤ 1 traversal (low-mean bias)")],
         "Hauer et al. (2002).    Clayton, D., & Kaldor, J. (1987). "
         "Biometrics, 43(3), 671–681.\n"
         "Low-mean dispersion caveat: Lord & Mannering (2010), Transp. Res. A 44(5), §2.5.",
         [(r"$\lambda_e$", "true rate on $e$"),
          (r"$\alpha_r, \beta_r$", "Gamma parameters,"),
          ("", "fitted per regime $r$"),
          (r"$\alpha_r/\beta_r$", "the regime's prior mean")]),

        ("M3 · Count regression on street attributes  (a safety performance function)",
         "pooling + covariates · the only route to never-ridden edges", STAT,
         r"$N_e \sim \mathrm{NegBin}(\mu_e, \theta), \quad "
         r"\log \mu_e = \log E_e + \beta_0 + \beta_{\mathrm{regime}} "
         r"+ \beta_1 x_{1e} + \dots$",
         [("Uses", "speed limit, betweenness, accidents, traffic volume"),
          ("Assumes", "log-linear in the attributes; negative binomial absorbs the overdispersion"),
          ("Pro", "the only model that can put a number on the 87% nobody rode"),
          ("Con", "covariates second-order & regime-collinear here → adds little over regime (≈ M2)")],
         "Lord, D., & Mannering, F. (2010). Transportation Research Part A, 44(5), 291–305 —\n"
         "overdispersion → negative binomial; weak covariates → unobserved heterogeneity dominates.",
         [(r"$\mu_e$", "expected count on $e$"),
          (r"$\theta$", "overdispersion parameter"),
          (r"$x_e$", "street attributes"),
          (r"$\beta$", "their coefficients"),
          (r"$\log E_e$", "exposure as an offset")]),

    ]

    LINE, NOTE_LINE, NOTE_X, REF_GAP = 0.30, 0.285, 10.1, 0.34
    INSET_H = 1.75          # extra room in M2 for the Gamma sketch
    heights, layouts = [], []
    for title, sub, _c, _formula, fields, ref, notation in models:
        # header = title (+ subtitle) then the formula, which may be multi-line
        # or tall (a fraction); fields start below all of that
        head = 0.34 + (0.30 if sub else 0)
        f_lines = _formula.count("\n") + 1
        f_h = 0.34 * f_lines + (0.24 if "frac" in _formula else 0) + 0.40
        n_left = sum(t.count("\n") + 1 for _l, t in fields)
        n_ref = (ref.count("\n") + 1) if ref else 0
        left_h = head + f_h + LINE * n_left + (REF_GAP if ref else 0) + 0.265 * n_ref
        right_h = head + f_h + NOTE_LINE * (len(notation) + 1)
        h = max(left_h, right_h) + 0.42
        if title.startswith("M2"):
            h += INSET_H
        heights.append(h)
        layouts.append((head, f_h))

    total = sum(heights) + 0.24 * len(heights) + 0.85
    fig, ax = plt.subplots(figsize=(14.8, total * 0.80))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, total)
    ax.axis("off")

    y = total - 0.15
    for (title, sub, color, formula, fields, ref, notation), h, (head, f_h) in zip(
            models, heights, layouts):
        for face, alpha, z in ((color, 0.07, 1), ("none", 1.0, 2)):
            ax.add_patch(FancyBboxPatch(
                (0.3, y - h), 14.2, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                linewidth=1.6, edgecolor=color, facecolor=face, alpha=alpha, zorder=z))
        ax.text(0.58, y - 0.28, title, fontsize=11.5, fontweight="bold", color=color,
                va="top", zorder=3)
        if sub:
            ax.text(0.58, y - 0.60, sub, fontsize=8.8, color=color, va="top",
                    style="italic", zorder=3)
        ax.text(0.58, y - 0.28 - head, formula, fontsize=12, color=INK,
                va="top", zorder=3, linespacing=1.3)
        yy = y - 0.28 - head - f_h            # where the two columns begin

        yl = yy
        for label, text in fields:
            ax.text(0.62, yl, label, fontsize=9, fontweight="bold", color=color,
                    va="top", zorder=3)
            ax.text(1.55, yl, text, fontsize=9, color=INK, va="top", zorder=3,
                    linespacing=1.45)
            yl -= LINE * (text.count("\n") + 1)
        # sources: no label, set off below the fields (M0 has none — it is arithmetic)
        if ref:
            ax.text(1.55, yl - REF_GAP, ref, fontsize=8.4, color=MUTED, va="top",
                    style="italic", zorder=3, linespacing=1.55)

        ax.plot([NOTE_X - 0.3, NOTE_X - 0.3], [y - h + 0.28, yy + 0.30],
                color=color, lw=1, alpha=0.35, zorder=3)
        yr = yy
        ax.text(NOTE_X, yr, "notation", fontsize=8.6, fontweight="bold",
                color=color, va="top", zorder=3)
        yr -= NOTE_LINE
        for symbol, meaning in notation:
            if symbol:
                ax.text(NOTE_X, yr, symbol, fontsize=9.5, color=INK, va="top", zorder=3)
            ax.text(NOTE_X + 1.1, yr, meaning, fontsize=8.8, color=MUTED,
                    va="top", zorder=3)
            yr -= NOTE_LINE

        if title.startswith("M2"):
            _gamma_inset(ax, 1.55, y - h + 0.55, 5.2, INSET_H - 0.75, color)
            ax.text(7.1, y - h + INSET_H - 0.30,
                    "the prior on $\\lambda_e$: a right-skewed spread of rates,\n"
                    "not one rate shared by every edge in the regime",
                    fontsize=8.6, color=MUTED, va="top", style="italic", zorder=3)
        y -= h + 0.24

    ax.text(7.4, 0.42,
            "Every rate carries rider-clustered uncertainty — repeated rides from one box are "
            "correlated, so naive standard errors are too small  (Lord & Mannering 2010, §2.4).",
            ha="center", fontsize=9, color=MUTED, va="center", style="italic", zorder=3)

    _save(fig, "fig11_risk_models.png")


def _mini_gamma(ax, x0, y0, w, h, color):
    """Tiny Gamma density used as the 'spread of true rates' picture."""
    k, mean = 1.8, 0.9
    theta = mean / k
    ins = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
    xx = np.linspace(1e-3, 3.0, 300)
    pdf = xx ** (k - 1) * np.exp(-xx / theta) / (gamma_fn(k) * theta ** k)
    ins.fill_between(xx, pdf, color=color, alpha=0.18)
    ins.plot(xx, pdf, color=color, lw=1.6)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ("top", "right", "left"):
        ins.spines[s].set_visible(False)
    ins.spines["bottom"].set_color(MUTED)
    ins.patch.set_alpha(0)


def fig_model_spec():
    """The full recommended specification, annotated: the generative story on the
    left, the estimator with every symbol glossed on the right, and the three
    transparency checks that wrap the thin model along the bottom."""
    DATA, PRIOR = DYN, "#7a4fb0"          # your data / borrowed from the regime
    RED = "#c0392b"
    fig, ax = plt.subplots(figsize=(14.6, 11.4))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 11.9)
    ax.axis("off")

    ax.text(0.3, 11.7, "A regime-pooled Empirical-Bayes rate  —  the model in full",
            fontsize=15, fontweight="bold", color=INK, va="top")

    def panel(x, y, w, h, color):
        for face, a, z in ((color, 0.05, 1), ("none", 1.0, 2)):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                linewidth=1.6, edgecolor=color, facecolor=face, alpha=a, zorder=z))

    def node(x, y, w, h, color, lines, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.06", linewidth=1.3,
            edgecolor=color, facecolor="white", zorder=4))
        ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center",
                fontsize=fs, color=INK, zorder=5, linespacing=1.3)

    def darrow(x, y0, y1, label):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
            mutation_scale=13, linewidth=1.4, color=MUTED, zorder=3))
        ax.text(x + 0.15, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=8.6, color=MUTED, style="italic", zorder=3)

    # ---------- LEFT: the generative story ----------------------------------
    panel(0.3, 5.35, 6.5, 5.65, MUTED)
    ax.text(0.55, 10.75, "①  How the data comes to be", fontsize=12,
            fontweight="bold", color=INK, va="top")
    ax.text(0.55, 10.4, "(the story the model assumes)", fontsize=8.8,
            color=MUTED, va="top", style="italic")

    node(0.7, 9.05, 4.2, 1.05, PRIOR,
         "LEVEL 2\neach street TYPE $r$ has a\nspread of true rates", fs=9.6)
    _mini_gamma(ax, 5.05, 9.1, 1.45, 0.95, PRIOR)
    ax.text(5.78, 8.95, r"$\lambda \sim \mathrm{Gamma}(\alpha_r,\beta_r)$",
            ha="center", fontsize=9.5, color=PRIOR, va="top")
    darrow(2.8, 9.05, 8.52, "pick this street's true rate")

    node(0.7, 7.55, 5.7, 0.95, INK,
         r"LEVEL 1a — this street's true (unseen) rate  $\lambda_e$", fs=10)
    darrow(2.8, 7.55, 7.05, "multiply by riding, then count at random")
    node(0.7, 5.95, 5.7, 1.05, DATA,
         r"LEVEL 1b — what you record" "\n"
         r"$N_e \sim \mathrm{Poisson}(\lambda_e \cdot E_e)$", fs=10)

    # ---------- RIGHT: the estimator ----------------------------------------
    panel(7.1, 5.35, 7.2, 5.65, HUB)
    ax.text(7.35, 10.75, "②  How we estimate it", fontsize=12,
            fontweight="bold", color=INK, va="top")
    ax.text(7.35, 10.4, "invert the story with Bayes — closed form", fontsize=8.8,
            color=MUTED, va="top", style="italic")

    ax.text(10.7, 9.55, r"$\hat{\lambda}_e = \dfrac{N_e + \alpha_r}{E_e + \beta_r}$",
            ha="center", fontsize=26, color=INK, va="center")

    # colour-coded glossary of the four symbols
    ax.text(7.55, 8.35, "YOUR DATA", fontsize=9, fontweight="bold", color=DATA)
    ax.text(7.55, 8.02, r"$N_e$  overtakes you recorded", fontsize=10, color=INK)
    ax.text(7.55, 7.66, r"$E_e$  km you actually rode (exposure)", fontsize=10, color=INK)
    ax.text(11.0, 8.35, "BORROWED FROM THE TYPE", fontsize=9, fontweight="bold", color=PRIOR)
    ax.text(11.0, 8.02, r"$\alpha_r$  overtakes the regime lends", fontsize=10, color=INK)
    ax.text(11.0, 7.66, r"$\beta_r$  km the regime lends", fontsize=10, color=INK)

    ax.plot([7.35, 14.05], [7.35, 7.35], color=HUB, lw=1, alpha=0.4)
    ax.text(7.55, 7.02, r"$w = \dfrac{\beta_r}{E_e+\beta_r}$", fontsize=13, color=INK, va="top")
    ax.text(9.35, 7.0, "how hard the edge is pulled toward its type\n"
            "$w\\approx0$: trust the edge   ·   $w\\approx1$: trust the type",
            fontsize=9, color=MUTED, va="top", linespacing=1.4)

    for i, (txt, col) in enumerate([
            ("well-ridden edge  →  trusts its own rate", DATA),
            (r"never-ridden edge  →  becomes the type average $\alpha_r/\beta_r$", PRIOR)]):
        yy = 6.15 - i * 0.42
        ax.add_patch(FancyBboxPatch((7.5, yy - 0.14), 6.35, 0.34,
            boxstyle="round,pad=0.03,rounding_size=0.06", linewidth=0,
            facecolor=col, alpha=0.12, zorder=3))
        ax.text(7.65, yy + 0.03, txt, fontsize=9.4, color=INK, va="center", zorder=4)

    # ---------- BOTTOM: the three transparency checks ------------------------
    ax.text(0.3, 4.95, "Three transparency checks wrapped around the thin model",
            fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0.3, 4.62, "we don't model the uncertainty away — we diagnose it, and vary it in the "
            "sampling experiments (Task 6)", fontsize=8.8, color=MUTED, va="top", style="italic")

    checks = [
        (STAT, "SPATIAL", "Are nearby edges still correlated after pooling?",
         ["Moran's I on the residuals,", "over the street network.",
          "Low → regime pooling already", "absorbed it, no CAR needed.",
          "High → add spatial pooling later."]),
        (DYN, "TEMPORAL", "Does the rate drift over time?",
         ["Stability test only where data allows", "(data-rich regimes, 2 windows).",
          "Elsewhere: flag it — most edges", "are seen in a single month.",
          "Exposure already handles uneven amounts."]),
        (RED, "PREFERENTIAL", "Riders choose where to ride.",
         ["No ground truth → can't de-bias.", "Estimate within-regime · cluster",
          "bootstrap by rider · report coverage ·", "state ignorability-within-regime ·",
          "stress-test the bias in Task 6."]),
    ]
    w3, gap = 4.42, 0.26
    for i, (col, tag, q, body) in enumerate(checks):
        x = 0.3 + i * (w3 + gap)
        ax.add_patch(FancyBboxPatch((x, 0.35), w3, 3.85,
            boxstyle="round,pad=0.05,rounding_size=0.08", linewidth=1.5,
            edgecolor=col, facecolor=col, alpha=0.07, zorder=1))
        ax.add_patch(FancyBboxPatch((x, 0.35), w3, 3.85,
            boxstyle="round,pad=0.05,rounding_size=0.08", linewidth=1.5,
            edgecolor=col, facecolor="none", zorder=2))
        ax.text(x + 0.25, 3.95, tag, fontsize=11.5, fontweight="bold", color=col, va="top")
        ax.text(x + 0.25, 3.58, q, fontsize=9, color=INK, va="top", style="italic")
        for j, ln in enumerate(body):
            ax.text(x + 0.25, 3.12 - j * 0.42, ln, fontsize=9, color=INK, va="top")

    _save(fig, "fig13_model_spec.png")


def fig_formalism():
    """The full model written out as typeset equations — the generative story,
    what conjugacy gives, the estimate, the empirical step, and the extension
    that shows where spatial and temporal terms enter."""
    STAT_G, RED = STAT, "#c0392b"
    rows = []   # (kind, text, note)  kind in {head, eq, sub}
    rows += [("head", "①  Generative model — the story the model tells", None)]
    rows += [("eq", r"$N_e \mid \lambda_e \;\sim\; \mathrm{Poisson}(\lambda_e\, E_e)$",
              "Level 1 · the overtake count you record on edge e")]
    rows += [("eq", r"$\lambda_e \;\sim\; \mathrm{Gamma}(\alpha_r,\, \beta_r)$",
              "Level 2 · the regime-level prior (spread of true rates in type r)")]
    rows += [("head", "②  Conjugacy gives two things for free", None)]
    rows += [("eq", r"$N_e \;\sim\; \mathrm{NegBin}\!\left(\alpha_r,\ \frac{\beta_r}{\beta_r+E_e}\right)$",
              "marginal count · THIS is your overdispersion")]
    rows += [("eq", r"$\lambda_e \mid N_e \;\sim\; \mathrm{Gamma}(\alpha_r+N_e,\ \beta_r+E_e)$",
              "posterior · still a Gamma → closed form")]
    rows += [("head", "③  The estimate  =  posterior mean  =  shrinkage", None)]
    rows += [("eq", r"$\hat{\lambda}_e \;=\; \frac{N_e+\alpha_r}{E_e+\beta_r}"
              r"\;=\;(1-w_e)\,\frac{N_e}{E_e}\;+\;w_e\,\frac{\alpha_r}{\beta_r}$",
              "blend the edge's own rate with the regime mean")]
    rows += [("eq", r"$w_e \;=\; \frac{\beta_r}{E_e+\beta_r}$",
              "shrink weight · w≈0 trust the edge, w≈1 trust the regime")]
    rows += [("head", "④  Empirical step — fit the prior from each regime", None)]
    rows += [("eq", r"$\bar{\lambda}_r=\frac{\sum_{e\in r} N_e}{\sum_{e\in r} E_e},"
              r"\quad \frac{\alpha_r}{\beta_r}=\bar{\lambda}_r,"
              r"\quad \mathrm{Var}(\lambda_e)=\frac{\alpha_r}{\beta_r^{2}}$",
              "method of moments · 'Empirical' = prior learned from the data")]
    rows += [("head", "⑤  Extension — where SPATIAL and TEMPORAL would enter", None)]
    rows += [("eq", r"$\log \mu_e = \log E_e + \theta_r + x_e^{\top}\beta"
              r" + \mathbf{u}_e + \gamma_t ,\quad N_e\sim\mathrm{NegBin}(\mu_e,\phi)$",
              None)]
    rows += [("sub", r"$\mathbf{u}\sim N\!\left(0,\ \tau^{2}(D-\rho W)^{-1}\right)$",
              "SPATIAL · CAR prior, W = street-network adjacency  (Barua et al. 2016)")]
    rows += [("sub", r"$\gamma_t\sim N(\gamma_{t-1},\ \sigma_\gamma^{2})$",
              "TEMPORAL · per-time-window drift  (Mannering 2018)")]

    # content-aware heights: fractions and sums render tall
    def eq_h(text):
        if "sum" in text:
            return 1.15
        if "frac" in text:
            return 0.80
        return 0.42

    def row_h(kind, text, note):
        if kind == "head":
            return 0.60
        h = eq_h(text) + 0.26                 # equation + gap to next row
        if note:
            h += 0.46                          # note line + its gap
        return h

    total = 0.95 + sum(row_h(k, t, n) for k, t, n in rows) + 0.85
    fig, ax = plt.subplots(figsize=(12.4, total * 0.80))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, total)
    ax.axis("off")

    ax.text(0.3, total - 0.15,
            "Empirical-Bayes Poisson-Gamma with a regime-level prior",
            fontsize=14.5, fontweight="bold", color=INK, va="top")

    y = total - 0.95
    for kind, text, note in rows:
        if kind == "head":
            ax.text(0.3, y, text, fontsize=11.5, fontweight="bold",
                    color=HUB, va="top")
            y -= 0.60
            continue
        indent = 1.5 if kind == "sub" else 0.9
        ax.text(indent, y, text, fontsize=16 if kind == "eq" else 14,
                color=INK, va="top")
        h = eq_h(text)
        if note:
            ncol = MUTED
            if "SPATIAL" in note:
                ncol = STAT_G
            elif "TEMPORAL" in note:
                ncol = DYN
            ax.text(indent + 0.15, y - h - 0.06, note, fontsize=9.2,
                    color=ncol, va="top", style="italic")
            y -= h + 0.46 + 0.26
        else:
            y -= h + 0.26

    ax.text(0.3, 0.42,
            "The thin model is this extension with  $\\mathbf{u}_e=0,\\ \\gamma_t=0$  and no covariates — "
            "the diagnostics decide whether to switch those terms on.",
            fontsize=9.6, color=RED, va="center", style="italic")

    _save(fig, "fig14_formalism.png")


def fig_risk_models_simple():
    """A stripped, talk-friendly version of the model ladder: one compact card
    per model — title, pooling level, formula, what it does, the catch. No
    notation column, no reference block (those live on the reading slide)."""
    # (color, title, pooling, formula, what it does, the catch)
    cards = [
        (DYN, "M0 · Raw rate on each edge", "no pooling",
         r"$\hat{\lambda}_e = N_e / E_e$",
         "each edge on its own — no model at all",
         "89% of ridden edges read 0; a single ride swings it to 1.0"),
        (DYN, "M1 · One rate per riding regime", "complete pooling",
         r"$\hat{\lambda}_r = \sum N_e \,/\, \sum E_e$",
         "one rate for the whole street type",
         "stable (thousands of passes), but blind to differences within a type"),
        (HUB, "M2 · Empirical-Bayes shrinkage", "partial pooling",
         r"$\hat{\lambda}_e = \dfrac{N_e + \alpha_r}{E_e + \beta_r}$",
         "each edge pulled toward its regime by how much data it has — the workhorse",
         "assumes a Gamma spread; shaky when most edges have ≤ 1 pass"),
        (STAT, "M3 · Covariate regression (SPF)", "pooling + covariates",
         r"$\log \mu_e = \log E_e + \beta_0 + \beta_{\mathrm{regime}} + \beta_1 x_{1e} + \dots$",
         "adds street attributes → the only route to never-ridden edges",
         "our covariates are second-order & regime-collinear, so it barely beats M2"),
    ]

    GAP = 0.34
    head, foot = 0.55, 0.70
    # a fraction formula needs a taller card; the flat ones stay compact
    hts = [2.04 if "frac" in f else 1.74 for _c, _t, _p, f, _d, _k in cards]
    total = head + sum(hts) + (len(cards) - 1) * GAP + foot
    fig, ax = plt.subplots(figsize=(12.6, total * 0.92))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, total)
    ax.axis("off")

    ax.text(0.35, total - 0.18, "Turning counts into a per-edge rate",
            fontsize=13.5, fontweight="bold", color=INK, va="top")
    ax.text(8.7, total - 0.20, "increasing pooling  ↓", fontsize=10.5,
            color=MUTED, va="top", style="italic")

    x0, w = 0.35, 11.9
    y = total - head
    for (color, title, pooling, formula, does, catch), HC in zip(cards, hts):
        top = y
        frac = "frac" in formula
        for face, alpha, z in ((color, 0.08, 1), ("none", 1.0, 2)):
            ax.add_patch(FancyBboxPatch(
                (x0, top - HC), w, HC, boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.6, edgecolor=color, facecolor=face, alpha=alpha, zorder=z))
        # thick colour accent down the left edge
        ax.add_patch(plt.Rectangle((x0, top - HC + 0.08), 0.09, HC - 0.16,
                                    color=color, zorder=3))
        ax.text(x0 + 0.34, top - 0.26, title, fontsize=12.5, fontweight="bold",
                color=color, va="top", zorder=3)
        ax.text(x0 + w - 0.30, top - 0.30, pooling, fontsize=10.5, color=color,
                va="top", ha="right", style="italic", zorder=3)
        ax.text(x0 + 0.40, top - 0.80, formula, fontsize=13.5, color=INK,
                va="top", zorder=3)
        does_y = top - (1.58 if frac else 1.32)
        ax.text(x0 + 0.40, does_y, does, fontsize=11, color=INK, va="top", zorder=3)
        ax.text(x0 + 0.40, does_y - 0.30, "the catch:  " + catch, fontsize=9.6,
                color=MUTED, va="top", style="italic", zorder=3)
        y -= HC + GAP

    ax.text(6.3, 0.36,
            "All four carry rider-clustered uncertainty.   Task 6 asks: how much pooling must we "
            "lean on as the data thins?",
            ha="center", fontsize=9.5, color=MUTED, va="center", style="italic", zorder=3)

    _save(fig, "fig12_risk_models_simple.png")


def fig_sampling_design():
    """The Task-6 design formalised to the same standard as the Poisson-Gamma
    'steps' slide: estimand -> point estimate -> clustered uncertainty -> the two
    thinning operators -> recovery metric -> sample-size budget. EB Poisson-Gamma
    is demoted to an appendix line."""
    GREEN, PURPLE, CYAN, ORANGE, RED = "#1f9d55", "#7a4fb0", "#2a78d6", "#e07b1a", "#c0392b"
    # (color, step, title, formula, caption[<=2 lines])
    steps = [
        (GREEN, "①", "Estimand — the reference 'truth'",
         r"$\lambda_r=\frac{\sum_{e\in r}N_e}{\sum_{e\in r}E_e}$",
         "the regime (×window) rate. Per-edge is unidentified (median shrink w = 0.90),\n"
         "so the regime is the level the data actually supports."),
        (PURPLE, "②", "Point estimate — thin, overdispersion-aware",
         r"$N_e\sim\mathrm{NegBin}(\mu_e,\theta),\quad \log\mu_e=\log E_e+\theta_r\,(+\gamma_t)$",
         "one intercept per regime (and window); NB absorbs the 1–13× overdispersion.\n"
         "No shrinkage machinery — you report the regime, not the edge."),
        (CYAN, "③", "Uncertainty — rider-clustered, not naive",
         r"$\mathrm{CI}(\lambda_r)$  via cluster bootstrap over riders $b$",
         "resample whole boxes → honest interval, ~6× wider than naive\n"
         "(effective n ≈ 21, not 47). Same engine as the experiment below."),
        (ORANGE, "④", "Thin on purpose — the two starvation axes",
         r"$\Pi^{\mathrm{edge}}_p$: keep fraction $p$ of edges $\;\cdot\;$ "
         r"$\Pi^{\mathrm{time}}_q$: keep fraction $q$ of revisits",
         "spatial breadth vs temporal depth; refit  $\\lambda_r^{(p)},\\ \\lambda_r^{(q)}$  "
         "on each thinned draw (R resamples)."),
        (RED, "⑤", "Recovery — how far the thinned estimate drifts",
         r"$\delta_r(p)=\frac{|\lambda_r^{(p)}-\lambda_r|}{\lambda_r}\quad$ compare $\;\delta(p)$ vs $\delta(q)$",
         "whichever axis makes the error rise faster per unit data removed is the\n"
         "binding starvation — this is the thesis answer (+ rank order, CI coverage)."),
        (HUB, "⑥", "Budget → playbook (Task 7)",
         r"$\mathrm{SE}(\lambda_r)\approx\sqrt{\varphi_r/E_r^{\mathrm{eff}}}"
         r"\;\Rightarrow\;\#\mathrm{edges}_r\propto \varphi_r/\tau^2$",
         "per-regime dispersion  φ_r (1–13×)  sets the sample size; read the minimum\n"
         "(p, q) that keeps  δ ≤ τ.  High-φ regimes need many more edges."),
    ]

    GAP, head, foot = 0.30, 0.92, 1.05
    hts = [2.02 if "frac" in f or "sqrt" in f else 1.66 for *_ , f, _ in steps]
    total = head + sum(hts) + (len(steps) - 1) * GAP + foot
    fig, ax = plt.subplots(figsize=(13.2, total * 0.92))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, total)
    ax.axis("off")

    ax.text(0.35, total - 0.16, "Task 6 · from a regime rate to a validated sampling budget",
            fontsize=14.5, fontweight="bold", color=INK, va="top")
    ax.text(0.35, total - 0.50,
            "N = overtakes · E = rider-km · e = edge · r = regime · b = rider/box · "
            "t = time window · φ = dispersion · τ = tolerance",
            fontsize=8.6, color=MUTED, va="top", style="italic")

    x0, w = 0.35, 12.5
    y = total - head
    for i, ((color, num, title, formula, caption), HC) in enumerate(zip(steps, hts), 1):
        top = y
        for face, alpha, z in ((color, 0.07, 1), ("none", 1.0, 2)):
            ax.add_patch(FancyBboxPatch(
                (x0, top - HC), w, HC, boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.5, edgecolor=color, facecolor=face, alpha=alpha, zorder=z))
        ax.add_patch(plt.Rectangle((x0, top - HC + 0.08), 0.09, HC - 0.16,
                                   color=color, zorder=3))
        ax.text(x0 + 0.34, top - 0.24, f"{num}  Step {i} — {title}",
                fontsize=12, fontweight="bold", color=color, va="top", zorder=3)
        ax.text(x0 + 0.40, top - 0.78, formula, fontsize=14, color=INK, va="top", zorder=3)
        cap_y = top - (1.52 if HC > 1.9 else 1.20)
        ax.text(x0 + 0.40, cap_y, caption, fontsize=9.6, color=MUTED, va="top",
                zorder=3, linespacing=1.4)
        y -= HC + GAP

    ax.text(6.6, 0.42,
            "Appendix (demoted):  EB Poisson–Gamma  $\\hat{\\lambda}_e=(N_e+\\alpha_r)/(E_e+\\beta_r)$  "
            "→ a per-edge risk map for the ~1% well-ridden hotspots only, not the sampling engine.",
            ha="center", fontsize=9.4, color=MUTED, va="center", style="italic", zorder=3)

    _save(fig, "fig15_sampling_design.png")


def fig_full_model():
    """The full estimator that accounts for exposure, spatial autocorrelation,
    temporal drift, rider clustering and preferential sampling — formalised to the
    same detail as the Poisson-Gamma slide. The key move: swap the Gamma prior
    (which cannot carry correlation) for a Gaussian latent field, so every source
    of variation is one additive term in log-lambda (a Poisson-lognormal LGM)."""
    GREEN, BLUE, PURPLE, ORANGE, RED = "#1f9d55", "#2a78d6", "#7a4fb0", "#e07b1a", "#c0392b"
    # (color, num, title, formula, caption)
    steps = [
        (HUB, "①", "Count model — what you record",
         r"$N_e \sim \mathrm{Poisson}(\lambda_e\, E_e)$",
         "overtakes scale with exposure $E_e$ (rider-km) — the offset. Poisson + an edge\n"
         "noise term below = Poisson-lognormal, so every source is one additive Gaussian term."),
        (HUB, "②", "Log-rate — one named term per source of variation",
         r"$\log \lambda_e = \mu + \theta_r + u_e + v_e + \gamma_t + c_b$",
         "baseline μ · regime $\\theta_r$ · spatial $u_e$ · edge-noise $v_e$ · time $\\gamma_t$ · "
         "rider $c_b$.\nThis is why the Gamma prior is dropped: a Gaussian field can hold all of them."),
        (GREEN, "③", "Spatial prior — the autocorrelation (Moran signal)",
         r"$\mathbf{u}\sim N\!\left(0,\ \tau^2 (D-\rho W)^{-1}\right)\ \cdot\ v_e\sim N(0,\sigma_v^2)$",
         "$W$ = street-network adjacency: neighbouring streets share risk (CAR/BYM).\n"
         "$v_e$ soaks up the 1–13× overdispersion (the unstructured part)."),
        (BLUE, "④", "Temporal prior — the drift",
         r"$\gamma_t \sim N(\gamma_{t-1},\ \sigma_\gamma^2)$",
         "a random walk across months/windows — lets the rate drift smoothly\n"
         "(the drift the clustered test could not resolve on thin data)."),
        (PURPLE, "⑤", "Rider prior — the clustering",
         r"$c_b \sim N(0,\ \sigma_c^2)$",
         "repeated rides from one box are correlated — this term does inside the model\n"
         "what the ×6 cluster bootstrap did outside it."),
        (RED, "⑥", "Preferential sampling — where people choose to ride",
         r"$\log \Lambda^{\mathrm{ride}}_e = \alpha_0 + x_e^{\top}\beta + \varphi\, u_e$",
         "the ride process shares the risk field $u_e$; $\\varphi\\neq0$ = preferential.\n"
         "Fitting the two jointly de-biases $\\lambda$ (Diggle, Menezes & Su 2010)."),
        (ORANGE, "⑦", "Estimate & use",
         r"$\hat{\lambda}_e = \mathrm{E}[\lambda_e \mid N, E]$  (INLA / MCMC)",
         "the posterior already carries spatial + temporal + rider uncertainty; the variance\n"
         "components $(\\tau,\\sigma_\\gamma,\\sigma_c,\\varphi)$ say how much each axis matters. Task 6 thins $E$ and refits."),
    ]

    GAP, head, foot = 0.30, 0.92, 1.05
    hts = [2.02 if ("frac" in f or "sqrt" in f or "Lambda" in f) else 1.66
           for *_, f, _ in steps]
    total = head + sum(hts) + (len(steps) - 1) * GAP + foot
    fig, ax = plt.subplots(figsize=(13.6, total * 0.92))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, total)
    ax.axis("off")

    ax.text(0.35, total - 0.16, "Task 5 · the full estimator — a Poisson-lognormal latent-Gaussian model",
            fontsize=14.5, fontweight="bold", color=INK, va="top")
    ax.text(0.35, total - 0.50,
            "N = overtakes · E = exposure (rider-km) · λ = rate · e = edge · r = regime · b = rider · "
            "t = window · W = street adjacency · φ = preferential strength · τ,σ = SDs",
            fontsize=8.4, color=MUTED, va="top", style="italic")

    x0, w = 0.35, 12.9
    y = total - head
    for (color, num, title, formula, caption), HC in zip(steps, hts):
        top = y
        for face, alpha, z in ((color, 0.07, 1), ("none", 1.0, 2)):
            ax.add_patch(FancyBboxPatch(
                (x0, top - HC), w, HC, boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=1.5, edgecolor=color, facecolor=face, alpha=alpha, zorder=z))
        ax.add_patch(plt.Rectangle((x0, top - HC + 0.08), 0.09, HC - 0.16,
                                   color=color, zorder=3))
        ax.text(x0 + 0.34, top - 0.24, f"{num}  {title}", fontsize=12,
                fontweight="bold", color=color, va="top", zorder=3)
        ax.text(x0 + 0.40, top - 0.78, formula, fontsize=13.5, color=INK, va="top", zorder=3)
        ax.text(x0 + 0.40, top - (1.52 if HC > 1.9 else 1.20), caption, fontsize=9.4,
                color=MUTED, va="top", zorder=3, linespacing=1.4)
        y -= HC + GAP

    ax.text(6.8, 0.42,
            "The thin model is this with  $u=v=\\gamma=c=0$  and  $\\varphi=0$  — the diagnostics decide "
            "which terms to switch on.",
            ha="center", fontsize=9.4, color=RED, va="center", style="italic", zorder=3)

    _save(fig, "fig16_full_model.png")


def fig_lean_model():
    """The LEAN baseline in the exact 'Steps' style of the Poisson-Gamma slide:
    a Negative-Binomial rate per street type with an exposure offset and
    rider-clustered intervals — the crash-frequency standard, nothing more."""
    G, P, B, O = "#3aa757", "#a12a9c", "#1f9fd0", "#e07b1a"   # step colours (match the PG slide)
    fig, ax = plt.subplots(figsize=(13.3, 7.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.4, 8.7, "Task 5:", fontsize=19, fontweight="bold", color=INK, va="top")
    ax.text(2.05, 8.7, "lean risk estimator — from counts to a per-type rate",
            fontsize=19, color=INK, va="top")

    # ---- left column: what it is / what is deferred ------------------------
    ax.text(0.5, 7.4, "Estimates:", fontsize=12.5, fontweight="bold", color=INK, va="top")
    for i, t in enumerate(["overtake rate per street type", "per rider-km (exposure)",
                           "the level the thin data supports"]):
        ax.text(0.7, 6.95 - i * 0.42, "•  " + t, fontsize=11.5, color=INK, va="top")

    ax.text(0.5, 5.35, "Deferred — a documented ladder,", fontsize=11.5,
            fontweight="bold", color=MUTED, va="top")
    ax.text(0.5, 5.05, "switched on only if diagnostics demand:", fontsize=11.5,
            fontweight="bold", color=MUTED, va="top")
    for i, t in enumerate(["spatial autocorrelation", "temporal drift",
                           "preferential sampling"]):
        ax.text(0.7, 4.6 - i * 0.42, "•  " + t, fontsize=11.5, color=MUTED, va="top")

    ax.text(0.5, 2.9,
            "The crash-frequency standard:\nNegative Binomial + exposure offset\n"
            "(Lord & Mannering 2010; exposure\nfrom GPS — Strauss 2015; near-miss\n"
            "rate — Aldred 2015).",
            fontsize=9.6, color=MUTED, va="top", style="italic", linespacing=1.5)

    # ---- notation (top right) ----------------------------------------------
    notation = ["N = overtakes", "E = exposure (rider-km)", "λ = overtake rate",
                "e = edge (street)", "r = street type", "b = rider (senseBox)",
                "μ = expected count", "θ = overdispersion dial"]
    for i, t in enumerate(notation):
        ax.text(8.6, 8.0 - i * 0.34, t, fontsize=10.5, color=INK, va="top")

    # ---- the four steps -----------------------------------------------------
    def step(y, color, label, formula, caption):
        ax.text(8.6, y, label, fontsize=13, fontweight="bold", color=color, va="top")
        ax.text(9.75, y, formula, fontsize=13, color=INK, va="top")
        ax.text(8.6, y - 0.40, caption, fontsize=10.3, color=MUTED, va="top",
                style="italic", linespacing=1.35)

    step(5.05, G, "Step 1", r"$\lambda_e = N_e \,/\, E_e$",
         "divide each edge's overtakes by how far it was ridden\n"
         "→ a rate per km (but one edge's rate is pure noise)")
    step(3.75, P, "Step 2", r"$\lambda_r = (\sum N_e) \,/\, (\sum E_e)$",
         "pool every edge of the SAME street type into one stable\n"
         "rate — the estimand (per-edge is unidentified here)")
    step(2.45, B, "Step 3",
         r"$N_e \sim \mathrm{NegBin}(\mu_e,\theta),\ \ \log\mu_e = \log E_e + \beta_r$",
         "fit it as the standard crash-count model; θ absorbs the\n"
         "hotspots (variance ≈ 6× Poisson).  Rate $\\hat\\lambda_r = e^{\\beta_r}$")
    step(1.05, O, "Step 4", r"95% interval on $\lambda_r$  by resampling whole riders $b$",
         "repeated rides from one sensor aren't independent →\n"
         "clustering by rider widens the interval ~6×")

    _save(fig, "fig17_lean_model.png")


if __name__ == "__main__":
    fig_workflow()
    fig_risk_models()
    fig_risk_models_simple()
    fig_model_spec()
    fig_formalism()
    fig_sampling_design()
    fig_full_model()
    fig_lean_model()
