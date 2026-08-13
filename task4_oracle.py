"""Build the per-edge oracle, the table the risk model screens.

One row per directed edge, both orientations present because contraflow is common and
never-ridden edges included as zeros. Exposure is offered as traversals, rider-km and
rider-hours, of which only rider-hours is measured.

Writes to output/task4_oracle/:
  task4_edge_traversals.csv             one row per directed edge and ride
  task4_edge_events.csv                 one row per overtake, with its edge
  task4_edge_oracle.csv                 one row per directed edge
  task4_overtake_coverage.csv           ridden edges by overtakes carried
  task4_rate_support.csv                the numbers behind task4_rate_predictors.png
  task4_intersection_robustness.csv     regime rates with near-junction events dropped
  task4_intersection_buffer_sweep.csv   the same, swept over buffer radius
  task4_*.png                           coverage, concentration, revisits, rate predictors
"""
from collections import Counter
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely.geometry import LineString

from task1_network import COVARIATES_CSV, EDGES_PATH, GRAPH_PATH
from task2b_overtake_events import EVENTS_PATH
from task3_mapmatching import MATCHED_POINTS_PATH

OUT_DIR = Path("output/task4_oracle")
TRAVERSALS_CSV = OUT_DIR / "task4_edge_traversals.csv"
EVENTS_CSV = OUT_DIR / "task4_edge_events.csv"
ORACLE_CSV = OUT_DIR / "task4_edge_oracle.csv"
OVERTAKE_COVERAGE_CSV = OUT_DIR / "task4_overtake_coverage.csv"
SUPPORT_CSV = OUT_DIR / "task4_rate_support.csv"
ROBUSTNESS_CSV = OUT_DIR / "task4_intersection_robustness.csv"
BUFFER_SWEEP_CSV = OUT_DIR / "task4_intersection_buffer_sweep.csv"

COVERAGE_MAP_FIG = OUT_DIR / "task4_coverage_map.png"
SATURATION_FIG = OUT_DIR / "task4_coverage_saturation.png"
CONCENTRATION_FIG = OUT_DIR / "task4_coverage_concentration.png"
BY_REGIME_FIG = OUT_DIR / "task4_coverage_by_regime.png"
REVISIT_FIG = OUT_DIR / "task4_revisit_by_regime.png"
OVERTAKE_COVERAGE_FIG = OUT_DIR / "task4_overtake_coverage.png"
RATE_PREDICTORS_FIG = OUT_DIR / "task4_rate_predictors.png"

GAP_CAP_S = 15
MOTOR = {"primary", "primary_link", "secondary", "secondary_link", "tertiary",
         "tertiary_link", "trunk", "trunk_link", "unclassified", "residential",
         "living_street"}


# ========= build the tables =========


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
        raise FileNotFoundError(f"{path} not found, run task1_network.py first")
    cov = pd.read_csv(path)
    cov["u_lo"] = np.minimum(cov["u"], cov["v"])
    cov["v_hi"] = np.maximum(cov["u"], cov["v"])
    return cov


def one_street_per_pair(cov):
    """One row per undirected pair. 31 pairs carry two ways the match cannot separate; n_ways flags them."""
    ranked = cov.sort_values("length_m", ascending=False)
    streets = ranked.drop_duplicates(subset=["u_lo", "v_hi"]).copy()
    n_ways = (cov.assign(_len=cov["length_m"].round(3))
              .groupby(["u_lo", "v_hi"])["_len"].nunique())
    streets["n_ways"] = streets.set_index(["u_lo", "v_hi"]).index.map(n_ways)
    return streets


def build_inventory(cov, streets):
    """Every directed edge in the network, carrying its street's covariates."""
    digitised = set(zip(cov["u"], cov["v"]))
    streets = streets.drop(columns=["u", "v"])
    fwd = streets.assign(u=streets["u_lo"], v=streets["v_hi"])
    rev = streets.assign(u=streets["v_hi"], v=streets["u_lo"])
    inv = pd.concat([fwd, rev], ignore_index=True)
    inv["is_digitised_dir"] = pd.MultiIndex.from_arrays([inv["u"], inv["v"]]).isin(digitised)
    return inv


def build_traversals(streets, points_path=MATCHED_POINTS_PATH):
    """One row per (directed edge, ride): when, how long, and the edge's full length."""
    pts = gpd.read_file(points_path, layer="task3_matched_points",
                        columns=["traj_id", "boxId", "createdAt", "u", "v"],
                        ignore_geometry=True)
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    pts = pts.sort_values(["traj_id", "createdAt"])
    nxt = pts.groupby("traj_id")["createdAt"].shift(-1)
    interval = (nxt - pts["createdAt"]).dt.total_seconds().clip(upper=GAP_CAP_S)
    pts["interval_s"] = interval.fillna(interval.median())
    pts = directed_key(pts)

    t = (pts.groupby(["u", "v", "traj_id"])
            .agg(boxId=("boxId", "first"), enter_time=("createdAt", "min"),
                 n_points=("createdAt", "size"), on_edge_s=("interval_s", "sum"),
                 u_lo=("u_lo", "first"), v_hi=("v_hi", "first"))
            .reset_index())
    t["is_weekend"] = t["enter_time"].dt.dayofweek >= 5

    return t.merge(streets[["u_lo", "v_hi", "length_m"]], on=["u_lo", "v_hi"], how="left")


def link_events(events, points_path=MATCHED_POINTS_PATH):
    """Each event on the edge of its anchor point, falling back to its first or last."""
    pts = gpd.read_file(points_path, layer="task3_matched_points",
                        columns=["point_id", "u", "v"], ignore_geometry=True)
    edge_of = pts.dropna(subset=["u"]).set_index("point_id")[["u", "v"]]

    linked = events.copy()
    linked[["u", "v"]] = np.nan
    linked["link_via"] = "none"
    for col, via in (("anchor_point_id", "anchor"), ("first_point_id", "first"),
                     ("last_point_id", "last")):
        todo = linked["u"].isna() & linked[col].isin(edge_of.index)
        if todo.any():
            linked.loc[todo, ["u", "v"]] = edge_of.loc[linked.loc[todo, col]].to_numpy()
            linked.loc[todo, "link_via"] = via
    via = linked["link_via"].value_counts().to_dict()
    print(f"[link] {linked['u'].notna().mean():.0%} of {len(linked)} events on an edge, {via}")
    return linked


def build_events(streets, events_path=EVENTS_PATH):
    """One row per overtake that reached an edge. edge_class comes from the covariates,
    the route the exposure uses too."""
    ev = directed_key(link_events(gpd.read_file(events_path)))
    ev["time"] = pd.to_datetime(ev["start"], utc=True)
    ev["is_weekend"] = ev["time"].dt.dayofweek >= 5

    ev = ev.merge(streets[["u_lo", "v_hi", "edge_class"]], on=["u_lo", "v_hi"], how="left")
    keep = ["u", "v", "u_lo", "v_hi", "event_uid", "traj_id", "boxId", "time",
            "is_weekend", "max_man_p", "min_clearance_cm", "edge_class", "link_via",
            "geometry"]
    return ev[[c for c in keep if c in ev.columns]]


def _rate(numerator, denominator):
    """Ratio where the denominator is positive, NaN where there is nothing to divide by."""
    return numerator.div(denominator).where(denominator > 0)


def build_oracle(inventory, traversals, events):
    """One row per directed edge: exposure, counts, rates, covariates, and the weekday,
    weekend and per-month splits of both."""
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
    counts = ["n_traversals", "n_boxes", "n_events", "n_months",
              "n_trav_weekend", "n_events_weekend"]
    o[counts] = o[counts].fillna(0).astype(int)
    o["rider_s"] = o["rider_s"].fillna(0.0)

    # a partial crossing is still booked the whole edge, so this overstates distance
    o["rider_km"] = o["n_traversals"] * o["length_m"] / 1000
    o["rider_h"] = o["rider_s"] / 3600
    o["is_observed"] = o["n_traversals"] > 0

    o["n_trav_weekday"] = o["n_traversals"] - o["n_trav_weekend"]
    o["n_events_weekday"] = o["n_events"] - o["n_events_weekend"]
    o["rider_km_weekday"] = o["n_trav_weekday"] * o["length_m"] / 1000
    o["rider_km_weekend"] = o["n_trav_weekend"] * o["length_m"] / 1000

    o["trav_per_month"] = _rate(o["n_traversals"], o["n_months"])
    o["events_per_month"] = _rate(o["n_events"], o["n_months"])
    o["sec_per_traversal"] = _rate(o["rider_s"], o["n_traversals"])
    o["rate_per_traversal"] = _rate(o["n_events"], o["n_traversals"])
    o["rate_per_rider_km"] = _rate(o["n_events"], o["rider_km"])
    o["rate_per_rider_h"] = _rate(o["n_events"], o["rider_h"])

    drop = ["rider_s", "is_sidepath", "is_cycling_street", "has_track_tag",
            "has_lane_tag", "n_accidents", "n_acc_severe", "n_accidents_recent",
            "n_acc_bike_recent", "aadt_hgv"]
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


# ========= analysis =========


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[csv] saved -> {path}")


def _poisson_ci(n, e):
    """95% interval on a rate n/e from the Poisson count n."""
    if e <= 0:
        return np.nan, np.nan, np.nan
    return n / e, max(n - 1.96 * np.sqrt(n), 0) / e, (n + 1.96 * np.sqrt(n)) / e


def coverage_table(oracle, trav):
    """Per undirected street: regime, traversals, overtakes, months seen."""
    net = (oracle.groupby(["u_lo", "v_hi"], as_index=False)
           .agg(edge_class=("edge_class", "first"),
                n_trav=("n_traversals", "sum"),
                n_events=("n_events", "sum")))
    t = trav[["u_lo", "v_hi", "enter_time"]].copy()
    t["month"] = pd.to_datetime(t["enter_time"], utc=True,
                                format="mixed").dt.strftime("%Y-%m")
    months = (t.groupby(["u_lo", "v_hi"], as_index=False)["month"].nunique()
              .rename(columns={"month": "n_months"}))
    full = net.merge(months, on=["u_lo", "v_hi"], how="left")
    full["n_months"] = full["n_months"].fillna(0).astype(int)
    full["covered"] = full["n_trav"] > 0
    return full


def overtake_coverage(oracle, top=10):
    """Directed ridden edges by how many overtakes they carry, zero included."""
    n = oracle.loc[oracle["is_observed"], "n_events"].to_numpy()
    counts = pd.DataFrame({"overtakes": [str(k) for k in range(top)] + [f"{top}+"],
                           "edges": [int((n == k).sum()) for k in range(top)]
                                    + [int((n >= top).sum())]})
    _write(counts, OVERTAKE_COVERAGE_CSV)
    print(f"  ridden {len(n):,} | zero overtakes {(n == 0).sum():,} ({(n == 0).mean():.0%})"
          f" | >=5 {(n >= 5).sum():,} | >=10 {(n >= 10).sum():,} | max {int(n.max())}")
    return counts


def rate_by_group(oracle, trav, ev):
    """Rate per rider-hour per bin, with a Poisson interval and a support flag.
    fig_rate_predictors draws these rows, so a bin edge cannot move for only one."""
    obs = oracle[oracle["is_observed"]]
    tr = trav.assign(enter_time=pd.to_datetime(trav["enter_time"], utc=True, format="mixed"))
    tr["hr"] = tr["on_edge_s"] / 3600            # exposure is time at risk
    e = ev.assign(time=pd.to_datetime(ev["time"], utc=True, format="mixed"))
    overall = obs["n_events"].sum() / obs["rider_h"].sum()

    rows = []

    def add(group, label, n, hours):
        n, hours = float(n), float(hours)
        if n < 5 or hours <= 0:  # too thin to say anything
            return
        rate, lo, hi = _poisson_ci(n, hours)
        rows.append(dict(group=group, label=str(label), events=int(n),
                         rider_h=round(hours, 2), rate=round(rate, 3),
                         x_average=round(rate / overall, 2),
                         lo=round(lo, 3), hi=round(hi, 3),
                         lo_x=round(lo / overall, 3), hi_x=round(hi / overall, 3),
                         support="solid" if n >= 100 else "ok" if n >= 30 else "THIN"))

    def edge_bins(group, col, bins, labels):
        d = obs.dropna(subset=[col])
        g = (d.groupby(pd.cut(d[col], bins, labels=labels), observed=True)
             .agg(e=("n_events", "sum"), k=("rider_h", "sum")))
        for label, r in g.iterrows():
            add(group, label, r["e"], r["k"])

    regimes = (obs.groupby("edge_class").agg(e=("n_events", "sum"), k=("rider_h", "sum"))
               .sort_values("e", ascending=False))
    for label, r in regimes[regimes["e"] >= 20].iterrows():
        add("riding regime", label.replace("_", " "), r["e"], r["k"])

    edge_bins("bicycle accidents", "n_acc_bike", [-1, 0, 1, 3, 1000],
              ["none", "1", "2-3", "4+"])
    edge_bins("speed limit", "maxspeed_kmh", [0, 20, 30, 50, 200],
              ["<=20 km/h", "21-30", "31-50", ">50"])
    edge_bins("betweenness", "betweenness", [-1, 1e-5, 1e-4, 1e-3, 1],
              ["lowest", "low", "mid", "high"])
    edge_bins("measured AADT", "aadt_kfz", [0, 5000, 12000, 1e6],
              ["<5k veh/day", "5-12k", ">12k"])

    edges, blocks = [-1, 6, 9, 15, 19, 24], ["night", "morning peak", "midday",
                                             "evening peak", "late"]
    by_block = pd.cut(e["time"].dt.hour, edges, labels=blocks).value_counts()
    hours_block = (tr.groupby(pd.cut(tr["enter_time"].dt.hour, edges, labels=blocks),
                              observed=False)["hr"].sum())
    for label in blocks:
        add("time of day", label, by_block.get(label, 0), hours_block.get(label, 0))

    day_type = {False: "weekday", True: "weekend"}
    by_day = (e["time"].dt.dayofweek >= 5).map(day_type).value_counts()
    hours_day = tr.groupby((tr["enter_time"].dt.dayofweek >= 5).map(day_type))["hr"].sum()
    for label in ("weekday", "weekend"):
        add("day type", label, by_day.get(label, 0), hours_day.get(label, 0))

    support = pd.DataFrame(rows)
    _write(support, SUPPORT_CSV)
    thin = support[support["support"] == "THIN"]
    print(f"  overall {overall:.2f} /rider-hour | THIN rows (<30 overtakes): "
          f"{', '.join(thin['group'] + ':' + thin['label']) if len(thin) else 'none'}")
    return support


def junction_nodes(graph_path=GRAPH_PATH):
    """Nodes where a motor road meets at least two other ways (regardless of the traffic regime)."""
    G = ox.convert.to_undirected(ox.load_graphml(graph_path))
    degree, motor = Counter(), Counter()
    for u, v, d in G.edges(data=True):
        highway = d.get("highway")
        highway = highway[0] if isinstance(highway, list) else highway
        for node in (u, v):
            degree[node] += 1
            motor[node] += highway in MOTOR
    nodes = ox.graph_to_gdfs(G, edges=False).to_crs(25832)
    return nodes[[degree[n] >= 3 and motor[n] >= 1 for n in nodes.index]]


def intersection_robustness(oracle, ev, buffer_m=15, buffers=(5, 10, 15, 20, 25, 30)):
    """Does the regime ranking survive dropping events near junctions?
     The buffer is the distance from a junction within which events are dropped."""
    junctions = junction_nodes()
    ev = ev[ev["edge_class"].notna()]
    dist, _ = cKDTree(np.c_[junctions.geometry.x, junctions.geometry.y]).query(
        np.c_[ev.geometry.x, ev.geometry.y])

    hours = oracle[oracle["is_observed"]].groupby("edge_class")["rider_h"].sum()

    def regime_table(buffer):
        t = pd.DataFrame({"hr": hours,
                          "n_all": ev.groupby("edge_class").size(),
                          "n_mid": ev[dist > buffer].groupby("edge_class").size()})
        t = t.dropna(subset=["hr"]).fillna({"n_all": 0, "n_mid": 0})
        t = t[t["n_all"] >= 20].copy()
        t["share_near"] = 1 - t["n_mid"] / t["n_all"]
        for col, n in (("rel_all", "n_all"), ("rel_mid", "n_mid")):
            t[col] = (t[n] / t["hr"]) / (t[n].sum() / t["hr"].sum())
        return t

    def top_bottom(t):
        ranked = t["rel_mid"].sort_values(ascending=False).index
        return set(ranked[:3]), set(ranked[-3:])

    top0, bottom0 = top_bottom(regime_table(0))
    sweep = []
    for buffer in buffers:
        t = regime_table(buffer)
        top, bottom = top_bottom(t)
        sweep.append(dict(buffer_m=buffer,
                          share_events_removed=round(float((dist <= buffer).mean()), 3),
                          rank_rho=round(float(spearmanr(t["rel_all"], t["rel_mid"]).statistic), 3),
                          top3_kept=len(top0 & top),        # tier membership, not order
                          bottom3_kept=len(bottom0 & bottom)))
    _write(pd.DataFrame(sweep), BUFFER_SWEEP_CSV)

    reg = regime_table(buffer_m).sort_values("rel_all", ascending=False).reset_index()
    _write(reg.round(3), ROBUSTNESS_CSV)
    rho = spearmanr(reg["rel_all"], reg["rel_mid"]).statistic
    print(f"  {len(junctions)} junctions, {(dist <= buffer_m).mean():.0%} of {len(ev)} events "
          f"within {buffer_m} m, rank rho {rho:.2f}")
    return pd.DataFrame(sweep)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {path}")


# ======== plotting ==========


def _despine(ax, sides=("top", "right")):
    for side in sides:
        ax.spines[side].set_visible(False)


def _regimes_by_coverage(cov, min_edges=20):
    """Regimes with enough streets to say anything, worst coverage first. Both regime
    figures use it, so they always show the same regimes in the same order."""
    g = cov.groupby("edge_class").agg(n_edges=("u_lo", "size"), cov=("covered", "mean"))
    return g[g["n_edges"] >= min_edges].sort_values("cov")


def _bar_labels(ax, values, fmt, pad):
    """The value written on each bar, since neither panel carries an axis to read it off."""
    for y, value in enumerate(values):
        ax.text(value + pad, y, fmt(value), va="center", fontsize=11,
                color="black", fontweight="bold")


def fig_coverage_map(ev, points_path=MATCHED_POINTS_PATH, path=COVERAGE_MAP_FIG):
    """Matched rides and overtakes over the network: how little of the city is reached."""
    pts = gpd.read_file(points_path, layer="task3_matched_points",
                        columns=["traj_id", "createdAt"])
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    lines = gpd.GeoDataFrame(
        [{"traj_id": traj_id,
          "geometry": LineString(ride.sort_values("createdAt").geometry.values)}
         for traj_id, ride in pts.groupby("traj_id") if len(ride) >= 2],
        geometry="geometry", crs=pts.crs)
    edges = gpd.read_file(EDGES_PATH)

    xlo, xhi = pts.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = pts.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    fig, ax = plt.subplots(figsize=(11, 11))
    edges.plot(ax=ax, color="lightgrey", linewidth=0.4, zorder=0)
    lines.plot(ax=ax, color="blue", linewidth=0.55, alpha=0.35, zorder=1)
    ev.plot(ax=ax, color="crimson", markersize=7, alpha=0.55, edgecolor="white",
            linewidth=0.2, zorder=2)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    handles = [
        Line2D([], [], color="lightgrey", lw=1.8, label="cyclable network"),
        Line2D([], [], color="blue", lw=1.8, alpha=0.7,
               label=f"ridden tracks ({lines['traj_id'].nunique()} rides, "
                     f"{lines.length.sum() / 1000:.0f} km)"),
        Line2D([], [], color="crimson", marker="o", ls="", markersize=8,
               markeredgecolor="white", markeredgewidth=0.4,
               label=f"overtake events ({len(ev)})"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=11,
              handletextpad=0.7, labelspacing=0.6)
    ax.set_title("Map-matched rides and overtake events", loc="left", fontsize=11)
    _save(fig, path)


def fig_coverage_saturation(trav):
    """Streets reaching 1 / 5 / 10 traversals as rides accumulate in collection order."""
    v = trav[["u_lo", "v_hi", "traj_id", "enter_time"]].copy()
    v["enter_time"] = pd.to_datetime(v["enter_time"], utc=True, format="mixed")
    first_seen = v.groupby("traj_id")["enter_time"].min().sort_values()
    rank = pd.Series(np.arange(1, len(first_seen) + 1), index=first_seen.index)
    v = (v.drop_duplicates(["u_lo", "v_hi", "traj_id"])
         .assign(rank=lambda d: d["traj_id"].map(rank))
         .sort_values("rank"))
    v["visit"] = v.groupby(["u_lo", "v_hi"]).cumcount() + 1
    series = {k: np.cumsum(np.bincount(v.loc[v["visit"] == k, "rank"],
                                       minlength=len(rank) + 1))[1:]
              for k in (1, 5, 10)}
    x = np.arange(1, len(rank) + 1)

    shades = dict(zip((1, 5, 10), plt.get_cmap("magma")(np.linspace(0.75, 0.2, 3))))
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
                fontsize=11, color="dimgrey",
                arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.8))
    ax.set_xlabel("rides collected  (in chronological order)")
    ax.set_ylabel("network edges reaching the threshold")
    ax.set_xlim(0, len(x) + 40)
    ax.set_ylim(0, series[1][-1] * 1.08)
    _despine(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_title("New coverage flattens as rides retread the same corridors",
                 loc="left", fontsize=11, pad=10)
    _save(fig, SATURATION_FIG)


def fig_coverage_concentration(cov):
    """Lorenz-style curve: how few streets hold most of the recorded data."""
    tr = np.sort(cov["n_trav"].to_numpy())[::-1]
    n = len(tr)
    x = np.arange(1, n + 1) / n
    y = np.cumsum(tr) / tr.sum()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.plot([0, 1], [0, 1], color="dimgrey", ls="--", lw=1, label="if coverage were uniform")
    ax.plot(x, y, color="crimson", lw=2.4, label="observed")
    ax.fill_between(x, y, color="crimson", alpha=0.08)
    for fx in (0.05, 0.10):
        fy = y[int(fx * n) - 1]
        ax.plot([fx, fx], [0, fy], color="dimgrey", lw=0.8, ls=":")
        ax.annotate(f"top {fx:.0%} of edges\nhold {fy:.0%} of all data",
                    (fx, fy), (fx + 0.06, fy - 0.13), fontsize=11, color="black",
                    arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.8))
    ax.set_xlabel("edges, ranked from most- to least-recorded")
    ax.set_ylabel("cumulative share of recorded traversals")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _despine(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.set_title(f"Only {cov['covered'].mean():.0%} of the network is ever recorded",
                 loc="left", fontsize=11, pad=10)
    _save(fig, CONCENTRATION_FIG)


def fig_coverage_by_regime(cov):
    """Per regime: share of its streets ever recorded, against recordings per street."""
    g = _regimes_by_coverage(cov)
    g["depth"] = cov[cov["covered"]].groupby("edge_class")["n_trav"].mean()
    y = np.arange(len(g))
    # sorted, so a regime keeps its colour between runs
    colors = dict(zip(sorted(cov["edge_class"].unique()), plt.get_cmap("tab20").colors))
    cols = [colors[c] for c in g.index]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True,
                                   gridspec_kw={"wspace": 0.06})
    axL.barh(y, g["cov"] * 100, color=cols, height=0.72, zorder=2)
    _bar_labels(axL, g["cov"] * 100, lambda v: f"{v / 100:.0%}", 1.5)
    axL.set_xlim(0, 92)
    axL.invert_xaxis()
    axL.set_title("BREADTH: edges ever recorded",
                  loc="right", fontsize=11, color="dimgrey", pad=8)

    axR.barh(y, g["depth"], color=cols, height=0.72, zorder=2)
    _bar_labels(axR, g["depth"], lambda v: f"{v:.1f}x", 0.15)
    axR.set_xlim(0, max(g["depth"]) * 1.18)
    axR.set_title("DEPTH: recordings per recorded edge",
                  loc="left", fontsize=11, color="dimgrey", pad=8)

    for ax in (axL, axR):
        _despine(ax, ("top", "right", "left", "bottom"))
        ax.tick_params(length=0)
        ax.set_xticks([])
    axL.set_yticks(y)
    axL.set_yticklabels([f"{c.replace('_', ' ')}\n{ne:,} edges"
                         for c, ne in zip(g.index, g["n_edges"])], fontsize=9)
    axL.tick_params(axis="y", pad=28)

    fig.suptitle("Which regimes are data-rich, and which are starved",
                 x=0.5, y=0.97, fontsize=11, ha="center")
    _save(fig, BY_REGIME_FIG)


def fig_revisit_by_regime(cov):
    """Distinct months each regime's recorded streets were seen in: 1 is a one-off."""
    regimes = _regimes_by_coverage(cov).index
    rec = cov[cov["covered"] & cov["edge_class"].isin(regimes)].copy()
    rec["band"] = pd.cut(rec["n_months"], [0, 1, 3, 999],
                         labels=["1 month", "2–3 months", "4+ months"])

    counts = (rec.groupby(["edge_class", "band"], observed=False).size()
              .unstack(fill_value=0))
    share = counts.div(counts.sum(axis=1), axis=0)
    share = share.reindex([c for c in regimes if c in share.index])
    one_off = (rec["n_months"] <= 1).mean()

    bands = ["1 month", "2–3 months", "4+ months"]
    shades = dict(zip(bands, plt.get_cmap("magma")(np.linspace(0.75, 0.2, len(bands)))))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(share))
    left = np.zeros(len(share))
    for band in bands:
        ax.barh(y, share[band].to_numpy(), left=left, color=shades[band],
                label=band, height=0.68)
        left += share[band].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in share.index], fontsize=11)
    ax.set_ylim(-0.6, len(share) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the regime's recorded edges", labelpad=6)
    _despine(ax, ("top", "right", "left"))
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3,
              frameon=False, fontsize=11,
              title="distinct months an edge was recorded")
    ax.set_title(f"{one_off:.0%} of recorded edges were seen in a single month",
                 loc="left", fontsize=11, pad=10)
    _save(fig, REVISIT_FIG)


def fig_overtake_coverage(oracle, counts, path=OVERTAKE_COVERAGE_FIG):
    """The same counts as bars. Log scale, because the zero bar dwarfs the rest."""
    n = oracle.loc[oracle["is_observed"], "n_events"].to_numpy()
    hist = counts["edges"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.bar(range(len(hist)), hist, color=["dimgrey"] + ["crimson"] * (len(hist) - 1),
           alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.7, hist.max() * 4)
    for i, v in enumerate(hist):
        if v:
            ax.text(i, v * 1.25, f"{v:,}", ha="center", fontsize=8, color="black")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts["overtakes"])
    ax.set_xlabel("overtakes recorded on the edge")
    ax.set_ylabel("directed edges  (log scale)")
    ax.set_title(f"{(n == 0).mean():.0%} of ridden edges saw no overtake",
                 fontsize=11, loc="left")
    _despine(ax)
    _save(fig, path)


def fig_rate_predictors(support, path=RATE_PREDICTORS_FIG):
    """Every predictor against the network average: one that matters spreads."""
    groups = list(dict.fromkeys(support["group"]))
    colors = dict(zip(groups, plt.get_cmap("tab10").colors))

    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(support) + 1.8))
    y, ticks, labels, seps = 0, [], [], []
    for group in groups:
        sub = support[support["group"] == group]
        for _, r in sub.iterrows():
            ax.plot([r["lo_x"], r["hi_x"]], [y, y], color=colors[group], lw=2, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["x_average"], y, "o", color=colors[group], markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ticks.append(y)
            labels.append(f"{r['label']}   ({r['events']})")
            y += 1
        seps.append(y - 0.5)
        # pinned in axes x, data y, so it survives any xlim
        ax.text(0.012, y - len(sub) / 2 - 0.5, group.upper(), fontsize=9,
                color=colors[group], va="center", ha="left", fontweight="bold",
                transform=ax.get_yaxis_transform())
        y += 1

    for s in seps[:-1]:
        ax.axhline(s + 0.5, color="lightgrey", lw=1, zorder=0)
    ax.axvline(1, color="black", ls="--", lw=1.2, zorder=1)
    ax.set_xscale("log")
    lo_x, hi_x = support["lo_x"].min() / 1.5, support["hi_x"].max() * 1.15
    ticks_x = [(t, l) for t, l in zip([0.25, 0.5, 1, 2, 4],
                                      ["1/4x", "1/2x", "average", "2x", "4x"])
               if lo_x <= t <= hi_x]
    ax.set_xticks([t for t, _ in ticks_x])
    ax.set_xticklabels([l for _, l in ticks_x], fontsize=11)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(lo_x, hi_x)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(-1, y - 1)
    ax.invert_yaxis()
    _despine(ax, ("top", "right", "left"))
    ax.tick_params(length=0)
    ax.set_xlabel("overtake rate per rider-hour / network average   (n) = overtakes",
                  fontsize=11)
    regime = support[support["group"] == "riding regime"]["x_average"]
    ax.set_title(f"Only street type separates the rate, spanning "
                 f"{regime.max() / regime.min():.0f}x", loc="left", fontsize=11, pad=10)
    _save(fig, path)


def main():
    cov = load_covariates()
    streets = one_street_per_pair(cov)
    inventory = build_inventory(cov, streets)
    trav = build_traversals(streets)
    ev = build_events(streets)
    oracle = build_oracle(inventory, trav, ev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(trav, TRAVERSALS_CSV)
    _write(ev.drop(columns="geometry"), EVENTS_CSV)
    _write(oracle, ORACLE_CSV)
    obs = oracle[oracle["is_observed"]]
    ne, km, hours = obs["n_events"].sum(), obs["rider_km"].sum(), obs["rider_h"].sum()
    print(f"{len(oracle)} directed edges, {len(obs)} ridden ({len(obs) / len(oracle):.0%}), "
          f"{len(trav)} traversals, {len(ev)} events")
    print(f"exposure {hours:.0f} rider-h and {km:.0f} rider-km, "
          f"median {trav['on_edge_s'].median():.0f} s per crossing")
    print(f"overall {ne / hours:.1f} overtakes per rider-hour, {ne / km:.2f} per km")

    cov_tbl = coverage_table(oracle, trav)
    counts = overtake_coverage(oracle)
    support = rate_by_group(oracle, trav, ev)
    intersection_robustness(oracle, ev)

    fig_coverage_map(ev)
    fig_overtake_coverage(oracle, counts)
    fig_coverage_saturation(trav)
    fig_coverage_concentration(cov_tbl)
    fig_coverage_by_regime(cov_tbl)
    fig_revisit_by_regime(cov_tbl)
    fig_rate_predictors(support)
    print("\nDONE")


if __name__ == "__main__":
    main()
