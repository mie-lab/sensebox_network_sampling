"""Slide figures for the pipeline. Each function is standalone; main runs all.

  fig_box_activity()        task 0  when each box was active (episodic collection)
  fig_network_regimes()     task 1  network coloured by riding regime (risk gradient)
  fig_overtake_extraction() task 2  one ride's classifier trace -> gated events
  fig_coverage_map()        task 3  matched rides + overtakes over the network

Run from an activated env:  python plots.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString

BASE = Path(".")
EDGES = BASE / "input/muenster_edges_classified.gpkg"
SUMMARY = BASE / "output/task2_trajectories/trajectory_summary_task2.csv"
POINTS2 = BASE / "output/task2_trajectories/trajectory_points_task2.gpkg"
EVENTS2 = BASE / "output/task2_trajectories/overtake_events_task2.gpkg"
MATCHED_POINTS = BASE / "output/task3_matching/matched_points_task3.gpkg"
MATCHED_EVENTS = BASE / "output/task3_matching/matched_events_task3.gpkg"
ORACLE_CSV = BASE / "output/task4_oracle/edge_oracle_task4.csv"
ORACLE_TRAV = BASE / "output/task4_oracle/edge_traversals_task4.csv"
ORACLE_EVENTS = BASE / "output/task4_oracle/edge_events_task4.csv"
FIG_DIR = BASE / "output/figures"

INK, MUTED, NET = "#0b0b0b", "#52514e", "#d9d9d6"
RIDE, EVENT = "#2a78d6", "#e34948"
MAN_P_TAU = 0.5

# riding regimes ordered + coloured as an exposure gradient: away from traffic
# (green/blue) -> painted/semi (yellow) -> shared with motor traffic (orange/red)
REGIME_COLORS = {
    "independent_path":   "#1baf7a",
    "offroad_track":      "#5cc99a",
    "pedestrian_zone":    "#9ad9bf",
    "roadside_track":     "#2a78d6",
    "bicycle_street":     "#6da7ec",
    "painted_lane":       "#eda100",
    "service_way":        "#f0b96b",
    "residential_street": "#eb6834",
    "minor_road_shared":  "#e8632f",
    "main_road_shared":   "#e34948",
    "bus_lane":           "#a11526",
}

plt.rcParams.update({"font.size": 12, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {p}")


# ------------------------------------------------------------------ task 0
def fig_box_activity():
    """One row per box: a faint lifespan bar (first→last ride) with a tick per ride.
    Sorted by first activity, so boxes appearing over time form a staircase."""
    s = pd.read_csv(SUMMARY, parse_dates=["start", "end"])
    s["start_num"] = mdates.date2num(s["start"])
    first = s.groupby("boxName")["start_num"].min()
    last = s.groupby("boxName")["start_num"].max()
    order = first.sort_values(ascending=False).index
    ypos = {name: i for i, name in enumerate(order)}

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("white")
    for name in order:
        y = ypos[name]
        ax.plot([first[name], last[name]], [y, y], color="0.86", lw=3,
                solid_capstyle="round", zorder=1)
    ax.scatter(s["start_num"], [ypos[n] for n in s["boxName"]],
               s=12, color=RIDE, alpha=0.8, marker="|", linewidths=1.1, zorder=2)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n[:22] for n in order], fontsize=7)
    ax.set_ylim(-1, len(order))
    ax.set_xlim(s["start_num"].min() - 15, s["start_num"].max() + 15)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", labelsize=10, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"When each senseBox was active  ({s['boxName'].nunique()} boxes, "
                 f"{len(s)} rides)\nCollection is episodic — most boxes ride briefly, "
                 "a few carry the dataset",
                 loc="left", fontsize=14, pad=12)
    _save(fig, "fig0_box_activity.png")


# --------------------------------------------------- coverage analysis (task 4)
def _coverage_table():
    """Per undirected edge: regime, length, traversal count, distinct months seen.
    Every network edge is present (uncovered ones get n_trav=0)."""
    def uv(df):
        d = df.dropna(subset=["u", "v"]).copy()
        d["uv"] = [tuple(sorted((int(a), int(b)))) for a, b in zip(d["u"], d["v"])]
        return d

    mp = gpd.read_file(MATCHED_POINTS, layer="matched_points_task3")
    mp["createdAt"] = pd.to_datetime(mp["createdAt"], utc=True)
    mp = uv(mp)
    mp["month"] = mp["createdAt"].dt.to_period("M").astype(str)
    trav = (mp.drop_duplicates(["uv", "traj_id"]).groupby("uv")
            .agg(n_trav=("traj_id", "size"), n_months=("month", "nunique")).reset_index())

    ed = uv(gpd.read_file(EDGES, layer="muenster_edges_classified"))
    net = ed.groupby("uv").agg(edge_class=("edge_class", "first"),
                               length=("length", "first")).reset_index()
    full = net.merge(trav, on="uv", how="left")
    full["n_trav"] = full["n_trav"].fillna(0).astype(int)
    full["n_months"] = full["n_months"].fillna(0).astype(int)
    full["covered"] = full["n_trav"] > 0
    return full


def _edge_counts():
    """Covered edges with traversals (T), overtakes (N), and regime."""
    def uv(df):
        d = df.dropna(subset=["u", "v"]).copy()
        d["uv"] = [tuple(sorted((int(a), int(b)))) for a, b in zip(d["u"], d["v"])]
        return d

    mp = uv(gpd.read_file(MATCHED_POINTS, layer="matched_points_task3"))
    me = uv(gpd.read_file(MATCHED_EVENTS, layer="matched_events_task3"))
    ed = uv(gpd.read_file(EDGES, layer="muenster_edges_classified"))
    t = mp.drop_duplicates(["uv", "traj_id"]).groupby("uv").size().rename("T")
    n = me.groupby("uv").size().rename("N")
    reg = ed.groupby("uv")["edge_class"].first().rename("regime")
    df = pd.concat([t, n, reg], axis=1)
    df = df[df["T"].notna()].copy()
    df["N"] = df["N"].fillna(0).astype(int)
    df["T"] = df["T"].astype(int)
    return df


def fig_overtake_rate():
    """Overtakes vs exposure per edge: the count scales with traversals (a rate),
    but spreads beyond the Poisson band — real hotspots, overdispersed risk."""
    df = _edge_counts()
    lam = df["N"].sum() / df["T"].sum()
    rng = np.random.default_rng(0)
    jx = df["T"] + rng.uniform(-0.28, 0.28, len(df))
    jy = df["N"] + rng.uniform(-0.16, 0.16, len(df))
    # 95% Poisson upper bound per edge; points above it exceed chance
    hi = lam * df["T"] + 1.96 * np.sqrt(lam * df["T"])
    hot = df["N"] > np.maximum(hi, 0.9)

    tt = np.linspace(0, df["T"].max(), 200)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.fill_between(tt, np.maximum(lam * tt - 1.96 * np.sqrt(lam * tt), 0),
                    lam * tt + 1.96 * np.sqrt(lam * tt), color=RIDE, alpha=0.12,
                    label="Poisson 95% band (pure chance)")
    ax.plot(tt, lam * tt, color=RIDE, lw=2, label=f"expected rate  ({lam:.3f} / traversal)")
    ax.scatter(jx[~hot], jy[~hot], s=7, color="0.6", alpha=0.35, edgecolor="none", zorder=2)
    ax.scatter(jx[hot], jy[hot], s=14, color=EVENT, alpha=0.7, edgecolor="none",
               zorder=3, label=f"above the chance band ({hot.sum()} edges, {hot.mean():.0%})")

    ax.set_xlabel("exposure — traversals of the edge  →")
    ax.set_ylabel("overtakes recorded on the edge  →")
    ax.set_xlim(0, df["T"].max() + 1)
    ax.set_ylim(-0.5, df["N"].max() + 0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_title("Overtakes scale with how often an edge is ridden — but not evenly\n"
                 "the count follows a rate per traversal, yet risk is overdispersed: "
                 "some edges far exceed chance",
                 loc="left", fontsize=13.5, pad=10)
    _save(fig, "fig7_overtake_rate.png")


def fig_coverage_saturation():
    """As rides accumulate (collection order), how many edges reach 1 / 5 / 10
    traversals — broad coverage saturates, deep coverage barely builds."""
    from collections import defaultdict
    mp = gpd.read_file(MATCHED_POINTS, layer="matched_points_task3")
    mp["createdAt"] = pd.to_datetime(mp["createdAt"], utc=True)
    mp = mp.dropna(subset=["u", "v"]).copy()
    mp["uv"] = [tuple(sorted((int(a), int(b)))) for a, b in zip(mp["u"], mp["v"])]
    order = mp.groupby("traj_id")["createdAt"].min().sort_values().index
    ride_edges = mp.groupby("traj_id")["uv"].apply(lambda s: set(s.unique()))

    cnt = defaultdict(int)
    c = {1: 0, 5: 0, 10: 0}
    series = {1: [], 5: [], 10: []}
    for tid in order:
        for e in ride_edges[tid]:
            cnt[e] += 1
            for k in (1, 5, 10):
                if cnt[e] == k:
                    c[k] += 1
        for k in (1, 5, 10):
            series[k].append(c[k])
    x = np.arange(1, len(order) + 1)

    shades = {1: "#9ec5f4", 5: "#3987e5", 10: "#184f95"}
    labels = {1: "≥ 1 traversal (touched at all)",
              5: "≥ 5 traversals", 10: "≥ 10 traversals (usable for a rate)"}
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for k in (1, 5, 10):
        ax.plot(x, series[k], color=shades[k], lw=2.6, label=labels[k])
        ax.annotate(f"{series[k][-1]:,}", (x[-1], series[k][-1]), (8, 0),
                    textcoords="offset points", va="center", fontsize=11,
                    color=shades[k], fontweight="bold")
    half = len(x) // 2
    ax.annotate(f"first half of rides:\n{series[1][half]:,} edges touched",
                (x[half], series[1][half]), (12, -55), textcoords="offset points",
                fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("rides collected  (in chronological order)  →")
    ax.set_ylabel("network edges reaching the threshold")
    ax.set_xlim(0, len(x) + 40)
    ax.set_ylim(0, series[1][-1] * 1.08)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_title("More rides bring diminishing new coverage\n"
                 "broad reach flattens as rides retread the same corridors; the repeat "
                 "visits a rate needs barely accumulate",
                 loc="left", fontsize=13.5, pad=10)
    _save(fig, "fig8_coverage_saturation.png")


def fig_coverage_concentration():
    """Lorenz-style curve: how few edges hold most of the recorded data."""
    full = _coverage_table()
    tr = np.sort(full["n_trav"].to_numpy())[::-1]
    n = len(tr)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(tr) / tr.sum()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1, label="if coverage were uniform")
    ax.plot(x, y, color=RIDE, lw=2.4, label="observed")
    ax.fill_between(x, y, color=RIDE, alpha=0.08)
    for fx in (0.05, 0.10):
        fy = y[int(fx * n) - 1]
        ax.plot([fx, fx], [0, fy], color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"top {fx:.0%} of edges\nhold {fy:.0%} of all data",
                    (fx, fy), (fx + 0.06, fy - 0.13), fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    cov = full["covered"].mean()
    ax.set_xlabel("edges, ranked from most- to least-recorded  →")
    ax.set_ylabel("cumulative share of recorded traversals")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.set_title(f"Recording effort is highly concentrated\nonly {cov:.0%} of the "
                 f"network is ever recorded; the rest is a blank",
                 loc="left", fontsize=14, pad=10)
    _save(fig, "fig4_coverage_concentration.png")


def fig_coverage_by_regime():
    """Per regime, two aligned bars: breadth (% of edges ever recorded) and
    depth (mean recordings per recorded edge). Sorted by breadth."""
    full = _coverage_table()
    g = full.groupby("edge_class").agg(
        n_edges=("uv", "size"), cov=("covered", "mean")).reset_index()
    depth = full[full["covered"]].groupby("edge_class")["n_trav"].mean()
    g["depth"] = g["edge_class"].map(depth)
    g = g[g["n_edges"] >= 20].sort_values("cov")   # ascending -> best on top
    y = np.arange(len(g))
    cols = [REGIME_COLORS.get(c, RIDE) for c in g["edge_class"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True,
                                   gridspec_kw={"wspace": 0.06})
    # left: breadth
    axL.barh(y, g["cov"] * 100, color=cols, height=0.72, zorder=2)
    for yi, (cov, ne) in enumerate(zip(g["cov"], g["n_edges"])):
        axL.text(cov * 100 + 1.5, yi, f"{cov:.0%}", va="center", fontsize=10.5,
                 color=INK, fontweight="bold")
    axL.set_xlim(0, 92)
    axL.invert_xaxis()                       # bars grow leftward, meet the labels
    axL.set_title("BREADTH\nshare of the regime's edges ever recorded",
                  loc="right", fontsize=11.5, color=MUTED, pad=8)

    # right: depth
    axR.barh(y, g["depth"], color=cols, height=0.72, zorder=2)
    for yi, d in enumerate(g["depth"]):
        axR.text(d + 0.15, yi, f"{d:.1f}×", va="center", fontsize=10.5,
                 color=INK, fontweight="bold")
    axR.set_xlim(0, max(g["depth"]) * 1.18)
    axR.set_title("DEPTH\nmean recordings per recorded edge",
                  loc="left", fontsize=11.5, color=MUTED, pad=8)

    for ax in (axL, axR):
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks([])
    axL.set_yticks(y)
    axL.set_yticklabels([f"{c.replace('_', ' ')}\n{ne:,} edges"
                         for c, ne in zip(g["edge_class"], g["n_edges"])], fontsize=9.5)
    axL.tick_params(axis="y", pad=28)
    for lab, c in zip(axL.get_yticklabels(), cols):
        lab.set_color(INK)

    fig.subplots_adjust(top=0.84)
    fig.suptitle("Which regimes are data-rich, and which are starved\n"
                 "bicycle streets are recorded broadly and often; the big everyday "
                 "regimes — residential, service, paths — are barely touched",
                 x=0.5, y=1.05, fontsize=14, ha="center")
    _save(fig, "fig5_coverage_by_regime.png")


def fig_revisit_by_regime():
    """Temporal depth: of each regime's recorded edges, how many distinct months
    were they seen in (1 = one-off, 4+ = repeatedly across seasons)."""
    full = _coverage_table()
    n_edges = full.groupby("edge_class").size()
    keep = n_edges[n_edges >= 20].index
    cov = full[full["covered"] & full["edge_class"].isin(keep)].copy()
    cov["band"] = pd.cut(cov["n_months"], [0, 1, 3, 999],
                         labels=["1 month", "2–3 months", "4+ months"])

    counts = (cov.groupby(["edge_class", "band"], observed=False).size()
              .unstack(fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    order = full.groupby("edge_class")["covered"].mean().sort_values().index
    share = share.reindex([c for c in order if c in share.index])

    bands = ["1 month", "2–3 months", "4+ months"]
    shades = {"1 month": "#cde2fb", "2–3 months": "#5598e7", "4+ months": "#184f95"}
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(share))
    left = np.zeros(len(share))
    for band in bands:
        ax.barh(y, share[band].to_numpy(), left=left, color=shades[band],
                label=band, height=0.68)
        left += share[band].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in share.index], fontsize=10)
    ax.set_ylim(-0.6, len(share) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the regime's recorded edges", labelpad=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              frameon=False, fontsize=11,
              title="distinct months an edge was recorded")
    ax.set_title("Most recorded edges are one-off visits  (51% seen in a single month)\n"
                 "temporal depth is thin — the revisits a rate estimate needs are rare",
                 loc="left", fontsize=13.5, pad=10)
    _save(fig, "fig6_revisit_by_regime.png")


# ------------------------------------------------------------------ task 2
def fig_overtake_extraction(traj_id="67226da749d0900007ca343c_40",
                            win_start_s=60, win_end_s=210):
    """One ride's classifier confidence over time; gated bursts become events."""
    pts = gpd.read_file(POINTS2, layer="trajectory_points_task2")
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    t = pts[pts["traj_id"] == traj_id].sort_values("createdAt").copy()
    t0 = t["createdAt"].min()
    t["s"] = (t["createdAt"] - t0).dt.total_seconds()
    w = t[(t["s"] >= win_start_s) & (t["s"] <= win_end_s)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    # shade gated bursts (man_p >= tau), merged if <=5 s apart
    gated = w[w["man_p"] >= MAN_P_TAU]
    if len(gated):
        brk = gated["s"].diff().gt(5).cumsum()
        for _, g in gated.groupby(brk):
            for ax in (ax1, ax2):
                ax.axvspan(g["s"].min() - 0.5, g["s"].max() + 0.5,
                           color=EVENT, alpha=0.13, zorder=0)

    ax1.plot(w["s"], w["man_p"], color=RIDE, lw=1.6)
    ax1.axhline(MAN_P_TAU, color=MUTED, ls="--", lw=1)
    ax1.text(w["s"].min(), MAN_P_TAU + 0.03, f"gate  τ = {MAN_P_TAU}",
             color=MUTED, fontsize=10)
    ax1.set_ylabel("classifier\nconfidence")
    ax1.set_ylim(-0.03, 1.03)

    d = w[w["value"] > 0]
    ax2.plot(w["s"], w["value"].where(w["value"] > 0), color="0.55", lw=1.2)
    ax2.set_ylabel("distance\n(cm)")
    ax2.set_xlabel("seconds into ride")

    n_ev = 0 if not len(gated) else (gated["s"].diff().gt(5).sum() + 1)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    ax1.set_title("From sensor stream to overtake events\n"
                  f"red bands = seconds the classifier flags a passing car → "
                  f"{int(n_ev)} events in this 2.5-min window",
                  loc="left", fontsize=14, pad=10)
    _save(fig, "fig2_overtake_extraction.png")


# ------------------------------------------------------------------ task 3
def _ride_lines(points):
    recs = []
    for tid, t in points.groupby("traj_id"):
        t = t.sort_values("createdAt")
        if len(t) >= 2:
            recs.append({"traj_id": tid, "geometry": LineString(t.geometry.values)})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=points.crs)


def fig_coverage_map():
    edges = gpd.read_file(EDGES, layer="muenster_edges_classified")
    pts = gpd.read_file(MATCHED_POINTS, layer="matched_points_task3")
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    ev = gpd.read_file(MATCHED_EVENTS, layer="matched_events_task3")
    lines = _ride_lines(pts)

    xlo, xhi = pts.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = pts.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    fig, ax = plt.subplots(figsize=(11, 11))
    edges.plot(ax=ax, color=NET, linewidth=0.4, zorder=0)
    lines.plot(ax=ax, color=RIDE, linewidth=0.55, alpha=0.35, zorder=1)
    ev.plot(ax=ax, color=EVENT, markersize=7, alpha=0.55,
            edgecolor="white", linewidth=0.2, zorder=2)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    handles = [
        Line2D([], [], color=NET, lw=1.8, label="cyclable network"),
        Line2D([], [], color=RIDE, lw=1.8, alpha=0.7,
               label=f"ridden tracks ({lines['traj_id'].nunique()} rides, "
                     f"{lines.length.sum() / 1000:.0f} km)"),
        Line2D([], [], color=EVENT, marker="o", ls="", markersize=8,
               markeredgecolor="white", markeredgewidth=0.4,
               label=f"overtake events ({len(ev):,})"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12.5,
                    handletextpad=0.7, borderpad=0.9, labelspacing=0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.85)
    ax.set_title("Map-matched rides and car-overtake events — Münster",
                 fontsize=17, color=INK, loc="left", pad=12)
    _save(fig, "fig3_coverage_map.png")


def fig_events_by_date():
    """Same map, but each overtake is coloured by WHEN it was recorded — to see
    whether spatial clusters are single visits (one colour) or built up across
    months (mixed colours), i.e. spatial vs. temporal collocation."""
    edges = gpd.read_file(EDGES, layer="muenster_edges_classified")
    pts = gpd.read_file(MATCHED_POINTS, layer="matched_points_task3")
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    ev = gpd.read_file(MATCHED_EVENTS, layer="matched_events_task3")
    ev["start"] = pd.to_datetime(ev["start"], utc=True, format="mixed").dt.tz_convert(None)
    lines = _ride_lines(pts)

    xlo, xhi = pts.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = pts.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    from matplotlib.colors import LinearSegmentedColormap
    reds = LinearSegmentedColormap.from_list(   # older = light, closer to today = deep red
        "event_reds", ["#f9c0bd", "#ef7b76", EVENT, "#a81f1d", "#6b0f0e"])

    dnum = mdates.date2num(ev["start"])
    order = np.random.RandomState(42).permutation(len(ev))   # avoid date-ordered occlusion

    fig, ax = plt.subplots(figsize=(11.6, 11))
    edges.plot(ax=ax, color=NET, linewidth=0.4, zorder=0)
    lines.plot(ax=ax, color=RIDE, linewidth=0.55, alpha=0.35, zorder=1)
    sc = ax.scatter(ev.geometry.x.values[order], ev.geometry.y.values[order],
                    c=dnum[order], cmap=reds, s=7, alpha=0.55,
                    edgecolors="white", linewidths=0.2, zorder=2)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.015)
    cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=3))
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    cbar.set_label("event date", fontsize=12)
    cbar.outline.set_visible(False)

    ax.set_title("Overtake events by date — Münster\n"
                 "colour = when the pass was recorded  ·  one colour along a street = a single visit",
                 fontsize=15, color=INK, loc="left", pad=12)
    _save(fig, "fig_events_by_date.png")


def fig_events_by_box():
    """Same map, but each overtake is coloured by WHICH box (rider) recorded it —
    a whole route in one colour means one rider produced it, separating rider
    collocation from calendar-date collocation."""
    edges = gpd.read_file(EDGES, layer="muenster_edges_classified")
    pts = gpd.read_file(MATCHED_POINTS, layer="matched_points_task3")
    ev = gpd.read_file(MATCHED_EVENTS, layer="matched_events_task3")
    lines = _ride_lines(pts)

    xlo, xhi = pts.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = pts.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    boxes = sorted(ev["boxId"].astype(str).unique())
    palette = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
               + list(plt.cm.tab20c.colors))                    # 60 distinct colours
    cmap = {b: palette[i % len(palette)] for i, b in enumerate(boxes)}
    colors = ev["boxId"].astype(str).map(cmap).tolist()
    order = np.random.RandomState(42).permutation(len(ev))

    fig, ax = plt.subplots(figsize=(11.6, 11))
    edges.plot(ax=ax, color=NET, linewidth=0.4, zorder=0)
    lines.plot(ax=ax, color=RIDE, linewidth=0.55, alpha=0.35, zorder=1)
    ax.scatter(ev.geometry.x.values[order], ev.geometry.y.values[order],
               c=[colors[i] for i in order], s=7, alpha=0.7,
               edgecolors="white", linewidths=0.2, zorder=2)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.set_title(f"Overtake events by box — Münster\n"
                 f"colour = which box (rider) recorded it, {len(boxes)} boxes  ·  "
                 f"one colour along a route = one rider",
                 fontsize=15, color=INK, loc="left", pad=12)
    _save(fig, "fig_events_by_box.png")


def _binned_rate(events, exposure, labels):
    """Rate per unit exposure with a 95% Poisson interval, per bin."""
    n = np.asarray(events, float)
    e = np.asarray(exposure, float)
    rate = np.where(e > 0, n / e, np.nan)
    lo = np.where(e > 0, np.maximum(n - 1.96 * np.sqrt(n), 0) / e, np.nan)
    hi = np.where(e > 0, (n + 1.96 * np.sqrt(n)) / e, np.nan)
    return pd.DataFrame({"label": labels, "n": n, "rate": rate,
                         "lo": rate - lo, "hi": hi - rate})


def fig_rate_predictors():
    """One shared scale: how many times the average overtake rate does each group
    experience? Everything on one axis, reference line at 1x, so a predictor that
    matters visibly spreads and one that does not collapses onto the line."""
    o = pd.read_csv(ORACLE_CSV)
    obs = o[o["is_observed"]].copy()
    tr = pd.read_csv(ORACLE_TRAV)
    ev = pd.read_csv(ORACLE_EVENTS)
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    tr["km"] = tr["length_m"] / 1000
    overall = obs["n_events"].sum() / obs["rider_km"].sum()

    rows = []   # (group, label, ratio, lo, hi, n_events)

    def add(group, labels, events, km):
        d = _binned_rate(events, km, labels)
        for _, r in d.iterrows():
            if r["n"] >= 5 and np.isfinite(r["rate"]):      # skip empty bins
                rows.append((group, r["label"], r["rate"] / overall,
                             (r["rate"] - r["lo"]) / overall,
                             (r["rate"] + r["hi"]) / overall, int(r["n"])))

    def edge_bins(group, col, bins, labels):
        d = obs.dropna(subset=[col]).copy()
        d["_b"] = pd.cut(d[col], bins, labels=labels)
        g = d.groupby("_b", observed=True).agg(e=("n_events", "sum"), k=("rider_km", "sum"))
        add(group, list(g.index), g["e"], g["k"])

    # riding regime — the predictor that actually separates
    g = (obs.groupby("edge_class").agg(e=("n_events", "sum"), k=("rider_km", "sum"))
         .sort_values("e", ascending=False))
    g = g[g["e"] >= 20]
    add("riding regime", [c.replace("_", " ") for c in g.index], g["e"], g["k"])

    edge_bins("bicycle accidents", "n_acc_bike", [-1, 0, 1, 3, 1000],
              ["none", "1", "2–3", "4+"])
    edge_bins("speed limit", "maxspeed_kmh", [0, 20, 30, 50, 200],
              ["≤20 km/h", "21–30", "31–50", ">50"])
    edge_bins("betweenness", "betweenness", [-1, 1e-5, 1e-4, 1e-3, 1],
              ["lowest", "low", "mid", "high"])
    edge_bins("measured AADT", "aadt_kfz", [0, 5000, 12000, 1e6],
              ["<5k veh/day", "5–12k", ">12k"])

    # time: coarse, interpretable blocks
    hb = [-1, 6, 9, 15, 19, 24]
    hl = ["night", "morning peak", "midday", "evening peak", "late"]
    add("time of day",
        hl,
        pd.cut(ev["time"].dt.hour, hb, labels=hl).value_counts().reindex(hl, fill_value=0),
        tr.groupby(pd.cut(tr["enter_time"].dt.hour, hb, labels=hl),
                   observed=False)["km"].sum().reindex(hl))
    wl = ["weekday", "weekend"]
    add("day type", wl,
        (ev["time"].dt.dayofweek >= 5).map({False: "weekday", True: "weekend"})
        .value_counts().reindex(wl, fill_value=0),
        tr.groupby((tr["enter_time"].dt.dayofweek >= 5)
                   .map({False: "weekday", True: "weekend"}))["km"].sum().reindex(wl))

    d = pd.DataFrame(rows, columns=["group", "label", "ratio", "lo", "hi", "n"])
    groups = list(dict.fromkeys(d["group"]))
    colors = {g: c for g, c in zip(groups, [RIDE, EVENT, "#eda100", "#1baf7a",
                                            "#4a3aa7", "#eb6834", "#52514e"])}

    fig, ax = plt.subplots(figsize=(10.5, 0.34 * len(d) + 2.6))
    y, ticks, labels, seps = 0, [], [], []
    for grp in groups:
        sub = d[d["group"] == grp]
        for _, r in sub.iterrows():
            ax.plot([r["lo"], r["hi"]], [y, y], color=colors[grp], lw=2, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["ratio"], y, "o", color=colors[grp], markersize=8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ticks.append(y)
            labels.append(f"{r['label']}   ({r['n']})")
            y += 1
        seps.append(y - 0.5)
        ax.text(0.033, y - len(sub) / 2 - 0.5, grp.upper(), fontsize=10,
                color=colors[grp], va="center", ha="left", fontweight="bold")
        y += 1

    for s in seps[:-1]:
        ax.axhline(s + 0.5, color="0.92", lw=1, zorder=0)
    ax.axvline(1, color=INK, ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["¼×", "½×", "average", "2×", "4×"], fontsize=11)
    ax.set_xlim(0.03, 6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-1, y - 1)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("overtake rate relative to the network average  "
                  "(log scale; bars = 95% interval, n = overtakes)", fontsize=10.5)
    ax.set_title("Street type dominates; the other predictors are marginal and regime-tangled\n"
                 "regime spans ~10x — but these panels are UNIVARIATE: adjusted for regime they "
                 "shrink to ±20% (accidents & speed wash out, betweenness inverts)",
                 loc="left", fontsize=13.5, pad=12)
    _save(fig, "fig9_rate_predictors.png")


if __name__ == "__main__":
    fig_box_activity()
    fig_overtake_extraction()
    fig_overtake_rate()
    fig_coverage_map()
    fig_coverage_concentration()
    fig_coverage_saturation()
    fig_coverage_by_regime()
    fig_revisit_by_regime()
    fig_rate_predictors()
