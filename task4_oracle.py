"""Build the per-edge oracle, the table the risk model screens.

The unit is a directed edge (u -> v). Both orientations of every street are present,
because a cyclist can ride either way and contraflow on one-ways is common. The undirected
key (u_lo, v_hi) is carried so directions can be pooled where data is thin. Every edge in
the network gets a row, including never-ridden ones (n_traversals = 0): screening covers
the whole network, so a blank has to be a row rather than a gap.

A traversal is not a fixed amount of exposure. Median time on an edge is a few seconds, but
some last minutes when a rider waits for traffic. So each traversal carries both its
distance (the edge's OSM length) and its time (summed per-point dwell). Rates are offered
per traversal, per rider-km and per rider-hour; choosing one is a task 5 decision.

The descriptive figures are built here too, straight from these tables, so a figure cannot
disagree with the oracle it describes.

Writes to output/task4_oracle/:
  task4_edge_traversals.csv   one row per directed edge and ride
  task4_edge_events.csv       one row per overtake
  task4_edge_oracle.csv       one row per directed edge: exposure, counts, covariates
plus coverage and support tables, and figures to output/figures/.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task1_network import GRAPH_PATH   # for the junction check below

OUT_DIR = Path("output/task4_oracle")
FIG_DIR = Path("output/figures")
MATCHED_POINTS = Path("output/task3_matching/task3_matched_points.gpkg")
MATCHED_EVENTS = Path("output/task3_matching/task3_matched_events.gpkg")
COVARIATES_CSV = Path("input/muenster_edge_covariates.csv")

TRAVERSALS_CSV = OUT_DIR / "task4_edge_traversals.csv"
EVENTS_CSV = OUT_DIR / "task4_edge_events.csv"
ORACLE_CSV = OUT_DIR / "task4_edge_oracle.csv"

GAP_CAP_S = 15

INK, MUTED = "black", "dimgrey"
BLUE, RED = "blue", "red"           # primary / accent
# riding regimes coloured as an exposure gradient: away from traffic (green/blue)
# -> painted/semi (yellow) -> shared with motor traffic (orange/red)
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
plt.rcParams.update({"font.size": 11, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def directed_key(df):
    """Travel-order (u, v) as ints, plus the undirected key (u_lo, v_hi)."""
    d = df.dropna(subset=["u", "v"]).copy()
    d["u"] = d["u"].astype(int)
    d["v"] = d["v"].astype(int)
    d["u_lo"] = np.minimum(d["u"], d["v"])
    d["v_hi"] = np.maximum(d["u"], d["v"])
    return d


def load_covariates(path=COVARIATES_CSV):
    """Static per-street covariates (run task1_network.py first)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run task1_network.py first")
    cov = pd.read_csv(path)
    cov["u_lo"] = np.minimum(cov["u"], cov["v"])
    cov["v_hi"] = np.maximum(cov["u"], cov["v"])
    return cov


def build_inventory(cov):
    """Every directed edge in the network: both orientations of every street,
    carrying that street's covariates."""
    digitised = set(zip(cov["u"], cov["v"]))
    streets = cov.drop_duplicates(subset=["u_lo", "v_hi"]).drop(columns=["u", "v"])
    fwd = streets.assign(u=streets["u_lo"], v=streets["v_hi"])
    rev = streets.assign(u=streets["v_hi"], v=streets["u_lo"])
    inv = pd.concat([fwd, rev], ignore_index=True)
    inv["is_digitised_dir"] = [(u, v) in digitised for u, v in zip(inv["u"], inv["v"])]
    return inv


def build_traversals(points_path=MATCHED_POINTS, cov=None):
    """One row per (directed edge, ride): when the ride was on the edge, how long,
    and the edge's length as distance ridden."""
    pts = gpd.read_file(points_path, layer="task3_matched_points",
                        columns=["traj_id", "boxId", "createdAt", "u", "v"],
                        ignore_geometry=True)
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    pts = pts.sort_values(["traj_id", "createdAt"])
    nxt = pts.groupby("traj_id")["createdAt"].shift(-1)
    pts["dwell_s"] = (nxt - pts["createdAt"]).dt.total_seconds().clip(upper=GAP_CAP_S)
    pts["dwell_s"] = pts["dwell_s"].fillna(pts["dwell_s"].median())
    pts = directed_key(pts)

    t = (pts.groupby(["u", "v", "traj_id"])
            .agg(boxId=("boxId", "first"), enter_time=("createdAt", "min"),
                 exit_time=("createdAt", "max"), n_points=("createdAt", "size"),
                 on_edge_s=("dwell_s", "sum"),
                 u_lo=("u_lo", "first"), v_hi=("v_hi", "first"))
            .reset_index())
    t["is_weekend"] = t["enter_time"].dt.dayofweek >= 5

    lengths = cov.drop_duplicates(subset=["u_lo", "v_hi"])[["u_lo", "v_hi", "length_m"]]
    return t.merge(lengths, on=["u_lo", "v_hi"], how="left")


def build_events(events_path=MATCHED_EVENTS):
    """One row per overtake event that got an edge."""
    ev = directed_key(gpd.read_file(events_path, layer="task3_matched_events",
                                    ignore_geometry=True))
    ev["start"] = pd.to_datetime(ev["start"], utc=True)
    ev["is_weekend"] = ev["start"].dt.dayofweek >= 5
    keep = ["u", "v", "u_lo", "v_hi", "event_uid", "traj_id", "boxId", "start",
            "is_weekend", "max_man_p", "min_clearance_cm", "edge_class", "link_via"]
    return ev[[c for c in keep if c in ev.columns]].rename(columns={"start": "time"})


def build_oracle(inventory, traversals, events):
    """One row per directed edge: exposure, counts, rates, covariates, and the
    temporal structure — weekday/weekend split of exposure & events,
    plus per-month traversal/event intensity."""
    trav = (traversals.groupby(["u", "v"])
            .agg(n_traversals=("traj_id", "size"), n_boxes=("boxId", "nunique"),
                 first_visit=("enter_time", "min"), last_visit=("enter_time", "max"),
                 rider_s=("on_edge_s", "sum"),
                 n_months=("enter_time",
                           lambda s: s.dt.tz_convert(None).dt.to_period("M").nunique()),
                 n_trav_weekend=("is_weekend", "sum"))
            .reset_index())
    ev = (events.groupby(["u", "v"])
          .agg(n_events=("time", "size"), n_events_weekend=("is_weekend", "sum"))
          .reset_index())

    o = (inventory.merge(trav, on=["u", "v"], how="left")
                  .merge(ev, on=["u", "v"], how="left"))
    for c, fill in (("n_traversals", 0), ("n_boxes", 0), ("n_events", 0),
                    ("rider_s", 0.0), ("n_months", 0), ("n_trav_weekend", 0),
                    ("n_events_weekend", 0)):
        o[c] = o[c].fillna(fill)
    for c in ("n_traversals", "n_events", "n_trav_weekend", "n_events_weekend"):
        o[c] = o[c].astype(int)

    o["rider_km"] = o["n_traversals"] * o["length_m"] / 1000
    o["rider_h"] = o["rider_s"] / 3600
    o["is_observed"] = o["n_traversals"] > 0

    # ---- temporal structure -------------------------------------------------
    o["n_trav_weekday"] = o["n_traversals"] - o["n_trav_weekend"]
    o["n_events_weekday"] = o["n_events"] - o["n_events_weekend"]
    o["rider_km_weekday"] = o["n_trav_weekday"] * o["length_m"] / 1000
    o["rider_km_weekend"] = o["n_trav_weekend"] * o["length_m"] / 1000
    # traversals / overtakes per active month (how often it is revisited)
    with np.errstate(divide="ignore", invalid="ignore"):
        o["trav_per_month"] = np.where(o["n_months"] > 0,
                                       o["n_traversals"] / o["n_months"], np.nan)
        o["events_per_month"] = np.where(o["n_months"] > 0,
                                         o["n_events"] / o["n_months"], np.nan)
        # mean seconds spent crossing the edge (exit_time - entry_time)
        o["sec_per_traversal"] = np.where(o["n_traversals"] > 0,
                                          o["rider_s"] / o["n_traversals"], np.nan)

        o["rate_per_traversal"] = np.where(o["n_traversals"] > 0,
                                           o["n_events"] / o["n_traversals"], np.nan)
        o["rate_per_rider_km"] = np.where(o["rider_km"] > 0,
                                          o["n_events"] / o["rider_km"], np.nan)
        o["rate_per_rider_h"] = np.where(o["rider_h"] > 0,
                                         o["n_events"] / o["rider_h"], np.nan)

    drop = ["rider_s", "is_sidepath", "is_cycling_street", "has_track_tag",
            "has_lane_tag", "n_accidents", "n_acc_severe", "n_accidents_recent",
            "n_acc_bike_recent", "aadt_hgv", "is_digitised_dir"]
    o = o.drop(columns=[c for c in drop if c in o.columns])

    front = ["u", "v", "u_lo", "v_hi", "edge_class", "is_observed",
             "n_traversals", "n_events", "rider_km", "rider_h", "n_boxes",
             "n_months", "trav_per_month", "events_per_month", "sec_per_traversal",
             "n_trav_weekday", "n_trav_weekend", "n_events_weekday",
             "n_events_weekend", "rider_km_weekday", "rider_km_weekend",
             "rate_per_traversal", "rate_per_rider_km", "rate_per_rider_h",
             "first_visit", "last_visit"]
    rest = [c for c in o.columns if c not in front]
    return o[front + rest]


# ========================= coverage figures =========================
# All of these describe COVERAGE, which belongs to a street rather than to one travel
# direction, so they pool (u, v) and (v, u) via the undirected key (u_lo, v_hi).

def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {p}")


def _write(df, name):
    p = OUT_DIR / name
    df.to_csv(p, index=False)
    print(f"[csv] saved -> {p}")


def _poisson_ci(n, e):
    """95% interval on a rate n/e from the Poisson count n."""
    if e <= 0:
        return np.nan, np.nan, np.nan
    return n / e, max(n - 1.96 * np.sqrt(n), 0) / e, (n + 1.96 * np.sqrt(n)) / e


def coverage_table(oracle, trav):
    """Per undirected street: regime, traversals, overtakes, distinct months seen.
    Every network street is present — never-ridden ones get n_trav = 0, because a
    blank is a finding here, not a missing row."""
    net = (oracle.groupby(["u_lo", "v_hi"], as_index=False)
           .agg(edge_class=("edge_class", "first"),
                n_trav=("n_traversals", "sum"),
                n_events=("n_events", "sum")))
    t = trav[["u", "v", "enter_time"]].copy()
    t["u_lo"] = np.minimum(t["u"], t["v"])
    t["v_hi"] = np.maximum(t["u"], t["v"])
    t["month"] = pd.to_datetime(t["enter_time"], utc=True,
                                format="mixed").dt.strftime("%Y-%m")
    months = (t.groupby(["u_lo", "v_hi"], as_index=False)["month"].nunique()
              .rename(columns={"month": "n_months"}))
    full = net.merge(months, on=["u_lo", "v_hi"], how="left")
    full["n_months"] = full["n_months"].fillna(0).astype(int)
    full["covered"] = full["n_trav"] > 0
    return full


def fig_overtake_rate(cov):
    """Overtakes vs exposure per street: the count scales with traversals (a rate),
    but spreads beyond the Poisson band — real hotspots, overdispersed risk."""
    df = cov[cov["covered"]]
    T, N = df["n_trav"].to_numpy(), df["n_events"].to_numpy()
    lam = N.sum() / T.sum()
    rng = np.random.default_rng(0)
    jx = T + rng.uniform(-0.28, 0.28, len(df))
    jy = N + rng.uniform(-0.16, 0.16, len(df))
    hi = lam * T + 1.96 * np.sqrt(lam * T)      # 95% Poisson upper bound per street
    hot = N > np.maximum(hi, 0.9)

    tt = np.linspace(0, T.max(), 200)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.fill_between(tt, np.maximum(lam * tt - 1.96 * np.sqrt(lam * tt), 0),
                    lam * tt + 1.96 * np.sqrt(lam * tt), color=BLUE, alpha=0.12,
                    label="Poisson 95% band (pure chance)")
    ax.plot(tt, lam * tt, color=BLUE, lw=2, label=f"expected rate  ({lam:.3f} / traversal)")
    ax.scatter(jx[~hot], jy[~hot], s=7, color="0.6", alpha=0.35, edgecolor="none", zorder=2)
    ax.scatter(jx[hot], jy[hot], s=14, color=RED, alpha=0.7, edgecolor="none",
               zorder=3, label=f"above the chance band ({hot.sum()} edges, {hot.mean():.0%})")

    ax.set_xlabel("exposure — traversals of the edge  →")
    ax.set_ylabel("overtakes recorded on the edge  →")
    ax.set_xlim(0, T.max() + 1)
    ax.set_ylim(-0.5, N.max() + 0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_title("Overtakes scale with how often an edge is ridden — but not evenly\n"
                 "the count follows a rate per traversal, yet risk is overdispersed: "
                 "some edges far exceed chance",
                 loc="left", fontsize=13.5, pad=10)
    _save(fig, "task4_overtake_rate.png")


def fig_coverage_saturation(trav):
    """As rides accumulate (collection order), how many streets reach 1 / 5 / 10
    traversals — broad coverage saturates, deep coverage barely builds."""
    from collections import defaultdict
    t = trav[["u", "v", "traj_id", "enter_time"]].copy()
    t["uv"] = list(zip(np.minimum(t["u"], t["v"]), np.maximum(t["u"], t["v"])))
    t["enter_time"] = pd.to_datetime(t["enter_time"], utc=True, format="mixed")
    order = t.groupby("traj_id")["enter_time"].min().sort_values().index
    ride_edges = t.groupby("traj_id")["uv"].apply(lambda s: set(s.unique()))

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
    _save(fig, "task4_coverage_saturation.png")


def fig_coverage_concentration(cov):
    """Lorenz-style curve: how few streets hold most of the recorded data."""
    tr = np.sort(cov["n_trav"].to_numpy())[::-1]
    n = len(tr)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(tr) / tr.sum()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], color="0.7", ls="--", lw=1, label="if coverage were uniform")
    ax.plot(x, y, color=BLUE, lw=2.4, label="observed")
    ax.fill_between(x, y, color=BLUE, alpha=0.08)
    for fx in (0.05, 0.10):
        fy = y[int(fx * n) - 1]
        ax.plot([fx, fx], [0, fy], color=MUTED, lw=0.8, ls=":")
        ax.annotate(f"top {fx:.0%} of edges\nhold {fy:.0%} of all data",
                    (fx, fy), (fx + 0.06, fy - 0.13), fontsize=11, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("edges, ranked from most- to least-recorded  →")
    ax.set_ylabel("cumulative share of recorded traversals")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.set_title(f"Recording effort is highly concentrated\nonly {cov['covered'].mean():.0%} "
                 f"of the network is ever recorded; the rest is a blank",
                 loc="left", fontsize=14, pad=10)
    _save(fig, "task4_coverage_concentration.png")


def fig_coverage_by_regime(cov):
    """Per regime, two aligned bars: breadth (% of streets ever recorded) and
    depth (mean recordings per recorded street). Sorted by breadth."""
    g = cov.groupby("edge_class", as_index=False).agg(
        n_edges=("u_lo", "size"), cov=("covered", "mean"))
    depth = cov[cov["covered"]].groupby("edge_class")["n_trav"].mean()
    g["depth"] = g["edge_class"].map(depth)
    g = g[g["n_edges"] >= 20].sort_values("cov")   # ascending -> best on top
    y = np.arange(len(g))
    cols = [REGIME_COLORS.get(c, BLUE) for c in g["edge_class"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True,
                                   gridspec_kw={"wspace": 0.06})
    axL.barh(y, g["cov"] * 100, color=cols, height=0.72, zorder=2)
    for yi, c in enumerate(g["cov"]):
        axL.text(c * 100 + 1.5, yi, f"{c:.0%}", va="center", fontsize=10.5,
                 color=INK, fontweight="bold")
    axL.set_xlim(0, 92)
    axL.invert_xaxis()                       # bars grow leftward, meet the labels
    axL.set_title("BREADTH\nshare of the regime's edges ever recorded",
                  loc="right", fontsize=11.5, color=MUTED, pad=8)

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
    for lab in axL.get_yticklabels():
        lab.set_color(INK)

    fig.subplots_adjust(top=0.84)
    fig.suptitle("Which regimes are data-rich, and which are starved\n"
                 "bicycle streets are recorded broadly and often; the big everyday "
                 "regimes — residential, service, paths — are barely touched",
                 x=0.5, y=1.05, fontsize=14, ha="center")
    _save(fig, "task4_coverage_by_regime.png")


def fig_revisit_by_regime(cov):
    """Temporal depth: of each regime's recorded streets, how many distinct months
    were they seen in (1 = one-off, 4+ = repeatedly across seasons)."""
    n_edges = cov.groupby("edge_class").size()
    keep = n_edges[n_edges >= 20].index
    rec = cov[cov["covered"] & cov["edge_class"].isin(keep)].copy()
    rec["band"] = pd.cut(rec["n_months"], [0, 1, 3, 999],
                         labels=["1 month", "2–3 months", "4+ months"])

    counts = (rec.groupby(["edge_class", "band"], observed=False).size()
              .unstack(fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    order = cov.groupby("edge_class")["covered"].mean().sort_values().index
    share = share.reindex([c for c in order if c in share.index])
    one_off = (rec["n_months"] <= 1).mean()

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
    ax.set_title(f"Most recorded edges are one-off visits  ({one_off:.0%} seen in a "
                 "single month)\ntemporal depth is thin — the revisits a rate estimate "
                 "needs are rare",
                 loc="left", fontsize=13.5, pad=10)
    _save(fig, "task4_revisit_by_regime.png")


def overtake_coverage(oracle, top=10):
    """How many overtakes each ridden edge carries — the sparsity that forces pooling.
    Log scale, because the zero bar dwarfs the rest. Unit = DIRECTED edge (u, v)."""
    obs = oracle[oracle["is_observed"]]
    n = obs["n_events"].to_numpy()
    never = int((~oracle["is_observed"]).sum())

    hist = [int((n == k).sum()) for k in range(top)] + [int((n >= top).sum())]
    labels = [str(k) for k in range(top)] + [f"{top}+"]
    _write(pd.DataFrame({"overtakes": labels, "edges": hist}), "task4_overtake_coverage.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(range(len(hist)), hist, color=[MUTED] + [BLUE] * (len(hist) - 1), alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(hist) * 4)
    for i, v in enumerate(hist):
        if v:
            ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("overtakes recorded on the edge")
    ax.set_ylabel("directed edges  (log scale)")
    ax.set_title(f"{(n == 0).mean():.0%} of ridden edges saw no overtake", fontsize=12, loc="left")
    ax.text(0.98, 0.93, f"{len(obs):,} ridden · {never:,} never ridden\n"
                        f"≥5 overtakes: {(n >= 5).sum()} edges · ≥10: {(n >= 10).sum()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "task4_overtake_coverage.png")
    print(f"  ridden {len(obs):,} | zero overtakes {(n == 0).sum():,} ({(n == 0).mean():.0%})"
          f" | >=5 {(n >= 5).sum():,} | >=10 {(n >= 10).sum():,} | max {int(n.max())}")


def forest_support_table(oracle, trav, ev):
    """The numbers behind fig_rate_predictors: events, exposure, rate, ratio-to-average,
    95% interval, and a support flag (solid >=100 events, ok 30-100, thin <30) for every
    bin the figure draws. Same bins and the same rider-HOUR exposure as the figure, so a
    row can always be traced to a bar."""
    obs = oracle[oracle["is_observed"]].copy()
    tr, e = trav.copy(), ev.copy()
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    e["time"] = pd.to_datetime(e["time"], utc=True, format="mixed")
    tr["hr"] = tr["on_edge_s"] / 3600.0
    overall = obs["n_events"].sum() / obs["rider_h"].sum()

    rows = []

    def add(group, label, n, expo):
        n, expo = float(n), float(expo)
        if n < 5 or expo <= 0:
            return
        rate, lo, hi = _poisson_ci(n, expo)
        flag = "solid" if n >= 100 else ("ok" if n >= 30 else "THIN")
        rows.append(dict(group=group, label=str(label), events=int(n),
                         rider_h=round(expo, 2), rate=round(rate, 3),
                         x_average=round(rate / overall, 2), lo=round(lo, 3),
                         hi=round(hi, 3), support=flag))

    def edge_bins(group, col, bins, labels):
        d = obs.dropna(subset=[col]).copy()
        d["_b"] = pd.cut(d[col], bins, labels=labels)
        g = d.groupby("_b", observed=True).agg(e=("n_events", "sum"), k=("rider_h", "sum"))
        for lab, r in g.iterrows():
            add(group, lab, r["e"], r["k"])

    g = (obs.groupby("edge_class").agg(e=("n_events", "sum"), k=("rider_h", "sum"))
         .sort_values("e", ascending=False))
    for lab, r in g[g["e"] >= 20].iterrows():
        add("riding regime", lab.replace("_", " "), r["e"], r["k"])
    edge_bins("bicycle accidents", "n_acc_bike", [-1, 0, 1, 3, 1000], ["none", "1", "2–3", "4+"])
    edge_bins("speed limit", "maxspeed_kmh", [0, 20, 30, 50, 200], ["<=20", "21-30", "31-50", ">50"])
    edge_bins("betweenness", "betweenness", [-1, 1e-5, 1e-4, 1e-3, 1],
              ["lowest", "low", "mid", "high"])
    edge_bins("measured AADT", "aadt_kfz", [0, 5000, 12000, 1e6], ["<5k", "5-12k", ">12k"])
    hb, hl = [-1, 6, 9, 15, 19, 24], ["night", "morning peak", "midday", "evening peak", "late"]
    ev_h = pd.cut(e["time"].dt.hour, hb, labels=hl).value_counts().reindex(hl, fill_value=0)
    hr_h = tr.groupby(pd.cut(tr["enter_time"].dt.hour, hb, labels=hl),
                      observed=False)["hr"].sum().reindex(hl)
    for lab in hl:
        add("time of day", lab, ev_h[lab], hr_h[lab])
    wmap = {False: "weekday", True: "weekend"}
    ev_w = (e["time"].dt.dayofweek >= 5).map(wmap).value_counts()
    hr_w = tr.groupby((tr["enter_time"].dt.dayofweek >= 5).map(wmap))["hr"].sum()
    for lab in ["weekday", "weekend"]:
        add("day type", lab, ev_w.get(lab, 0), hr_w.get(lab, 0))

    df = pd.DataFrame(rows)
    _write(df, "task4_forest_support_table.csv")
    thin = df[df["support"] == "THIN"]
    print(f"  overall {overall:.2f} /rider-hour | THIN rows (<30 overtakes): "
          f"{', '.join(thin['group'] + ':' + thin['label']) if len(thin) else 'none'}")
    return df


def intersection_robustness(oracle, buffer_m=15, buffers=(5, 10, 15, 20, 25, 30)):
    """Tests the assumption this table is built on: that an overtake belongs to an EDGE,
    not a junction. Many overtakes sit near junctions (turning/queuing cars) and
    map-matching is least certain there, so the regime ranking in fig_rate_predictors
    could be a junction artifact. We drop every event within a buffer of a REAL road
    junction and check whether the ranking survives — swept over buffers, since any
    single radius would be arbitrary.

    A 'real junction' is a node of degree >= 3 in the MOTOR-road subgraph — this
    drops the many degree-3 cycleway/path connectors that inflate the naive count."""
    import osmnx as ox
    from collections import Counter
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    MOTOR = {"primary", "primary_link", "secondary", "secondary_link", "tertiary",
             "tertiary_link", "trunk", "trunk_link", "unclassified", "residential",
             "living_street"}                       # service excluded: low cross-traffic

    def hw1(d):
        h = d.get("highway")
        return h[0] if isinstance(h, list) else h

    G = ox.load_graphml(GRAPH_PATH)
    deg = Counter()
    for u, v, d in ox.convert.to_undirected(G).edges(data=True):
        if hw1(d) in MOTOR:
            deg[u] += 1
            deg[v] += 1
    nodes = ox.graph_to_gdfs(G, edges=False).to_crs(25832)
    real = nodes[[deg.get(n, 0) >= 3 for n in nodes.index]]

    ev = gpd.read_file(MATCHED_EVENTS).to_crs(25832)
    ev = ev[ev["edge_class"].notna()].copy()
    tree = cKDTree(np.c_[real.geometry.x, real.geometry.y])
    dist, _ = tree.query(np.c_[ev.geometry.x, ev.geometry.y])

    obs = oracle[oracle["is_observed"]]
    hr = obs.groupby("edge_class")["rider_h"].sum()

    def regime_table(buf):
        """Relative regime rates with all events vs with near-junction events dropped.
        Exposure is unchanged — only events are removed — so this is a pure event test."""
        keep = dist > buf
        t = pd.DataFrame({"hr": hr,
                          "n_all": ev.groupby("edge_class").size(),
                          "n_mid": ev[keep].groupby("edge_class").size()})
        t = t.dropna(subset=["hr"]).fillna({"n_all": 0, "n_mid": 0})
        t = t[t["n_all"] >= 20].copy()
        t["share_near"] = 1 - t["n_mid"] / t["n_all"]
        t["rel_all"] = (t["n_all"] / t["hr"]) / (t["n_all"].sum() / t["hr"].sum())
        t["rel_mid"] = (t["n_mid"] / t["hr"]) / (t["n_mid"].sum() / t["hr"].sum())
        return t

    # sensitivity across buffers — one arbitrary radius would not settle the assumption
    sweep = []
    base = regime_table(0)["rel_mid"].sort_values(ascending=False)
    top0, bot0 = set(base.index[:3]), set(base.index[-3:])
    for b in buffers:
        t = regime_table(b)
        s = t["rel_mid"].sort_values(ascending=False)
        sweep.append(dict(buffer_m=b, share_events_removed=round(float((dist <= b).mean()), 3),
                          rank_rho=round(float(spearmanr(t["rel_all"], t["rel_mid"]).statistic), 3),
                          top3_kept=len(top0 & set(s.index[:3])),      # tier membership, not order
                          bottom3_kept=len(bot0 & set(s.index[-3:]))))
    sweep = pd.DataFrame(sweep)
    _write(sweep, "task4_intersection_buffer_sweep.csv")

    reg = regime_table(buffer_m).sort_values("rel_all", ascending=False).reset_index()
    rho = float(spearmanr(reg["rel_all"], reg["rel_mid"]).statistic)
    _write(reg.round(3), "task4_intersection_robustness.csv")
    print(f"  {len(real):,} real motor junctions | {(dist <= buffer_m).mean():.0%} of "
          f"{len(ev):,} events within {buffer_m} m | rank stability rho = {rho:.2f}")
    return sweep


def _binned_rate(events, exposure, labels):
    """Rate per unit exposure with a 95% Poisson interval, per bin."""
    n = np.asarray(events, float)
    e = np.asarray(exposure, float)
    rate = np.where(e > 0, n / e, np.nan)
    lo = np.where(e > 0, np.maximum(n - 1.96 * np.sqrt(n), 0) / e, np.nan)
    hi = np.where(e > 0, (n + 1.96 * np.sqrt(n)) / e, np.nan)
    return pd.DataFrame({"label": labels, "n": n, "rate": rate,
                         "lo": rate - lo, "hi": hi - rate})


def fig_rate_predictors(oracle, trav, ev):
    """One shared scale: how many times the average overtake rate does each group
    experience? Everything on one axis, reference line at 1x, so a predictor that
    matters visibly spreads and one that does not collapses onto the line."""
    obs = oracle[oracle["is_observed"]].copy()
    tr = trav.copy()
    e = ev.copy()
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    e["time"] = pd.to_datetime(e["time"], utc=True, format="mixed")
    tr["hr"] = tr["on_edge_s"] / 3600                  # exposure = time at risk
    overall = obs["n_events"].sum() / obs["rider_h"].sum()

    rows = []   # (group, label, ratio, lo, hi, n_events)

    def add(group, labels, events, hours):
        d = _binned_rate(events, hours, labels)
        for _, r in d.iterrows():
            if r["n"] >= 5 and np.isfinite(r["rate"]):      # skip empty bins
                rows.append((group, r["label"], r["rate"] / overall,
                             (r["rate"] - r["lo"]) / overall,
                             (r["rate"] + r["hi"]) / overall, int(r["n"])))

    def edge_bins(group, col, bins, labels):
        d = obs.dropna(subset=[col]).copy()
        d["_b"] = pd.cut(d[col], bins, labels=labels)
        g = d.groupby("_b", observed=True).agg(e=("n_events", "sum"), k=("rider_h", "sum"))
        add(group, list(g.index), g["e"], g["k"])

    # riding regime — the predictor that actually separates
    g = (obs.groupby("edge_class").agg(e=("n_events", "sum"), k=("rider_h", "sum"))
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
    add("time of day", hl,
        pd.cut(e["time"].dt.hour, hb, labels=hl).value_counts().reindex(hl, fill_value=0),
        tr.groupby(pd.cut(tr["enter_time"].dt.hour, hb, labels=hl),
                   observed=False)["hr"].sum().reindex(hl))
    wl = ["weekday", "weekend"]
    add("day type", wl,
        (e["time"].dt.dayofweek >= 5).map({False: "weekday", True: "weekend"})
        .value_counts().reindex(wl, fill_value=0),
        tr.groupby((tr["enter_time"].dt.dayofweek >= 5)
                   .map({False: "weekday", True: "weekend"}))["hr"].sum().reindex(wl))

    d = pd.DataFrame(rows, columns=["group", "label", "ratio", "lo", "hi", "n"])
    groups = list(dict.fromkeys(d["group"]))
    colors = {g: c for g, c in zip(groups, [BLUE, RED, "#eda100", "#1baf7a",
                                            "#4a3aa7", "#eb6834", "#52514e"])}

    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(d) + 1.8))
    y, ticks, labels, seps = 0, [], [], []
    for grp in groups:
        sub = d[d["group"] == grp]
        for _, r in sub.iterrows():
            ax.plot([r["lo"], r["hi"]], [y, y], color=colors[grp], lw=2, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["ratio"], y, "o", color=colors[grp], markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ticks.append(y)
            labels.append(f"{r['label']}   ({r['n']})")
            y += 1
        seps.append(y - 0.5)
        # group name pinned to the left edge in AXES x / DATA y, so it survives any xlim
        ax.text(0.012, y - len(sub) / 2 - 0.5, grp.upper(), fontsize=9,
                color=colors[grp], va="center", ha="left", fontweight="bold",
                transform=ax.get_yaxis_transform())
        y += 1

    for s in seps[:-1]:
        ax.axhline(s + 0.5, color="0.92", lw=1, zorder=0)
    ax.axvline(1, color=INK, ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log")
    lo_x, hi_x = d["lo"].min() / 1.5, d["hi"].max() * 1.15      # fit the data, no dead space
    keep = [(t, l) for t, l in zip([0.25, 0.5, 1, 2, 4], ["¼×", "½×", "average", "2×", "4×"])
            if lo_x <= t <= hi_x]
    ax.set_xticks([t for t, _ in keep])
    ax.set_xticklabels([l for _, l in keep], fontsize=10)
    ax.xaxis.set_minor_locator(plt.NullLocator())      # log minor ticks would overprint these
    ax.set_xlim(lo_x, hi_x)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_ylim(-1, y - 1)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("overtake rate per rider-hour ÷ network average   ·   (n) = overtakes",
                  fontsize=9.5)
    ax.set_title("Only street type separates the overtake rate\n"
                 "regime spans 6×; everything else within 0.9–1.4×  ·  univariate, unadjusted",
                 loc="left", fontsize=11.5, pad=10)
    _save(fig, "task4_rate_predictors.png")


def main():
    cov = load_covariates()
    inventory = build_inventory(cov)
    trav = build_traversals(cov=cov)
    ev = build_events()
    oracle = build_oracle(inventory, trav, ev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trav.to_csv(TRAVERSALS_CSV, index=False)
    ev.to_csv(EVENTS_CSV, index=False)
    oracle.to_csv(ORACLE_CSV, index=False)
    print(f"oracle: {len(oracle)} directed edges "
          f"({oracle['is_observed'].sum()} ever ridden, "
          f"{oracle['is_observed'].mean():.0%}) | traversal rows {len(trav)} | "
          f"event rows {len(ev)}")
    print(f"saved: {TRAVERSALS_CSV.name}, {EVENTS_CSV.name}, {ORACLE_CSV.name}")

    obs = oracle[oracle["is_observed"]]
    ne, km = obs["n_events"].sum(), obs["rider_km"].sum()
    print(f"exposure: {km:.0f} rider-km, {obs['rider_h'].sum():.0f} rider-h, {ne} overtakes"
          f"  |  overall {ne / km:.2f}/rider-km, {ne / obs['n_traversals'].sum():.3f}/traversal")
    print(f"on-edge time: median {trav['on_edge_s'].median():.0f} s/crossing  |  "
          f"weekend {obs['rider_km_weekend'].sum() / km:.0%} of km, "
          f"{obs['n_events_weekend'].sum() / ne:.0%} of overtakes")

    pd.set_option("display.width", 200, "display.max_columns", 60)
    print("\nhead — one row per directed edge:")
    print(oracle.head().to_string())

    cov_tbl = coverage_table(oracle, trav)
    overtake_coverage(oracle)
    fig_overtake_rate(cov_tbl)
    fig_coverage_saturation(trav)
    fig_coverage_concentration(cov_tbl)
    fig_coverage_by_regime(cov_tbl)
    fig_revisit_by_regime(cov_tbl)
    fig_rate_predictors(oracle, trav, ev)
    forest_support_table(oracle, trav, ev)      # the numbers behind that figure
    intersection_robustness(oracle)             # ...and whether its ranking survives


if __name__ == "__main__":
    main()
