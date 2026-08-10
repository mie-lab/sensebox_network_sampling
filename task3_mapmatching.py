"""Map-match the kept rides to the street network and pin events onto edges.

Each ride is matched as an ordered path (Leuven HMM)
IN TRAVEL ORDER -- direction is preserved from here to the oracle.
Every event inherits the edge of its anchor point (falling back to its first/last
point); where several classified edges share a node pair (road vs parallel
sidepath, multigraph keys), the class comes from the geometry nearest the event.

Run AFTER task2b_overtake_events.py. If matched files already exist,
matching is skipped (delete them to force a re-match).

Writes to output/task3_matching/:
  task3_matched_points.gpkg   points with matched edge (u, v), travel order
  task3_matched_events.gpkg   events with edge (u, v) + edge_class + link_via
  task3_match_summary.csv     match/link/direction descriptives (for slides)
  task3_matched_per_box/      one validation page per box
and the coverage map for this stage to output/figures/.
"""
from pathlib import Path
import logging
import re

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString

logging.getLogger("leuvenmapmatching").setLevel(logging.WARNING)

from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from task1_network import GRAPH_PATH      # the graph is a task-1 artifact

OUT_DIR = Path("output/task3_matching")
FIG_DIR = Path("output/figures")
POINTS_PATH = Path("output/task2_trajectories/task2b_trajectory_points.gpkg")
EVENTS_PATH = Path("output/task2_trajectories/task2b_overtake_events.gpkg")
EDGES_PATH = Path("input/muenster_edges_classified.gpkg")
MATCHED_POINTS_PATH = OUT_DIR / "task3_matched_points.gpkg"
MATCHED_EVENTS_PATH = OUT_DIR / "task3_matched_events.gpkg"
MATCH_SUMMARY_CSV = OUT_DIR / "task3_match_summary.csv"
PER_BOX_MATCH_DIR = OUT_DIR / "task3_matched_per_box"

MAX_DIST = 50
OBS_NOISE = 15
MAX_DIST_INIT = 60
MIN_PROB_NORM = 0.001

VALUE_VMAX = 250  # clearance scale cap for point sizing

INK, NET = "#0b0b0b", "#d9d9d6"
RIDE, EVENT = "#2a78d6", "#e34948"


def build_inmem_map(graph_path=GRAPH_PATH):
    """InMemMap from the saved osmnx graph (already in CRS_METRIC)."""
    G = ox.load_graphml(graph_path)
    m = InMemMap("muenster", use_latlon=False, use_rtree=True, index_edges=True)
    for nid, data in G.nodes(data=True):
        m.add_node(int(nid), (float(data["x"]), float(data["y"])))
    # both directions are added even for oneways and without the reverse edge such rides would fail to match at all.
    # The direction check in match_summary reports how often travel goes against the digitized direction.
    for u, v, _ in G.edges(keys=True):
        u, v = int(u), int(v)
        m.add_edge(u, v)
        m.add_edge(v, u)
    m.purge()
    return m, G


def match_trajectory(matcher, t):
    """Match one time-ordered ride; return the travelled edge per point (or None).
    - Important: when the lattice dies mid-ride (GPS > max_dist from any network edge:
      off-network paths, tunnels, drift), Leuven stops and would leave the whole
      rest of the ride unplaced. We resume matching just after the dead point,
      so only the unmatchable stretch stays empty.
    """
    t = t.sort_values("createdAt")
    edges = pd.Series([None] * len(t), index=t.index, dtype=object)
    start = 0
    while start <= len(t) - 2:
        sub = t.iloc[start:]
        path = list(zip(sub.geometry.x.values, sub.geometry.y.values))
        states, last_idx = matcher.match(path)
        if states:
            for mm in matcher.lattice_best:
                if mm.is_emitting() and mm.edge_m is not None:
                    oi = start + int(mm.obs)
                    if oi < len(t):
                        edges.iloc[oi] = (int(mm.edge_m.l1), int(mm.edge_m.l2))
        start += max(int(last_idx), 0) + 1
    return edges


def match_all(points, m):
    matcher = DistanceMatcher(
        m, max_dist=MAX_DIST, max_dist_init=MAX_DIST_INIT, obs_noise=OBS_NOISE,
        min_prob_norm=MIN_PROB_NORM, non_emitting_states=True,
    )
    out = points.sort_values(["traj_id", "createdAt"]).copy()
    groups = list(out.groupby("traj_id", sort=False))
    n_total = len(groups)
    matched, n_ok = [], 0
    for i, (tid, t) in enumerate(groups, 1):
        edges = match_trajectory(matcher, t)
        ok = edges.notna().any()
        n_ok += ok
        matched.append(edges)
        if not ok or i % 50 == 0 or i == n_total:
            print(f"[match] {i}/{n_total}  {tid}  {len(t)} pts  "
                  f"{'matched' if ok else 'FAILED':7s}  ({edges.notna().mean():.0%} placed)")
    out["edge"] = pd.concat(matched)
    out["u"] = out["edge"].map(lambda e: e[0] if isinstance(e, tuple) else None)
    out["v"] = out["edge"].map(lambda e: e[1] if isinstance(e, tuple) else None)
    print(f"[match] done: {n_ok}/{n_total} trajectories matched, "
          f"{out['edge'].notna().mean():.0%} of all points placed")
    return out.drop(columns="edge")


def link_events(events, matched_points):
    """Give each event the edge of its anchor point, falling back to first/last in case it would not work.

    link_via records which point supplied the edge ('anchor'/'first'/'last') or
    'none' when the whole burst sat in an unmatched stretch."""
    edge_of = matched_points.set_index("point_id")[["u", "v"]]

    def pick(row):
        for col, via in (("closest_point_id", "anchor"), ("first_point_id", "first"),
                         ("last_point_id", "last")):
            pid = row.get(col)
            if pid in edge_of.index:
                u, v = edge_of.loc[pid, ["u", "v"]]
                if pd.notna(u):
                    return pd.Series({"u": u, "v": v, "link_via": via})
        return pd.Series({"u": None, "v": None, "link_via": "none"})

    uv = events.apply(pick, axis=1)
    ev = events.copy()
    ev["u"], ev["v"], ev["link_via"] = uv["u"], uv["v"], uv["link_via"]
    print(f"[link] {ev['u'].notna().mean():.0%} of events linked to an edge")
    return ev


def _uv_key(df):
    """Turn each row's (u, v) endpoint nodes into one lookup key that matches the
    same street regardless of travel direction."""
    return [tuple(sorted((int(a), int(b)))) if pd.notna(a) and pd.notna(b) else None
            for a, b in zip(df["u"], df["v"])]


def join_edge_class(events, edges_path=EDGES_PATH):
    """Attach edge_class via an undirected node pair (match may return either order).

    Several edges can share a node pair (directions, multigraph keys — notably a
    road and its parallel sidepath); resolve those by the edge geometry nearest
    to the event point instead of picking one arbitrarily.
    """
    if not Path(edges_path).exists():
        print(f"[class] {edges_path} not found -> skipping edge_class join")
        return events
    edges = gpd.read_file(edges_path)
    if not {"u", "v"}.issubset(edges.columns):
        print("[class] edges file lacks u/v columns -> skipping")
        return events
    edges = edges.assign(_uv=_uv_key(edges)).dropna(subset=["_uv"])
    groups = {k: g for k, g in edges.groupby("_uv")}

    def pick_class(row):
        cand = groups.get(row["_uv"])
        if cand is None:
            return None
        classes = cand["edge_class"].unique()
        if len(classes) == 1:
            return classes[0]
        return cand.loc[cand.distance(row.geometry).idxmin(), "edge_class"]

    ev = events.copy()
    ev["_uv"] = _uv_key(ev)
    ev["edge_class"] = ev.apply(pick_class, axis=1)
    n_multi = sum(1 for g in groups.values() if g["edge_class"].nunique() > 1)
    print(f"[class] {ev['edge_class'].notna().mean():.0%} of events classed "
          f"({n_multi} node pairs carry >1 class -> resolved by nearest geometry)")
    return ev.drop(columns="_uv")


def match_summary(matched_points, matched_events, G, path=MATCH_SUMMARY_CSV):
    """Descriptives of the matching step, saved as one CSV:
    how many rides/points/events got matched, where the loss concentrates,
    whether travel direction is valid, and how often the multigraph had to be
    disambiguated."""
    p, ev = matched_points, matched_events
    placed = p["u"].notna()
    per_ride = placed.groupby(p["traj_id"]).mean()

    # direction validity: does the travel-ordered (u, v) exist as a directed edge?
    directed = {(int(u), int(v)) for u, v, _ in G.edges(keys=True)}
    pp = p[placed]
    uv = list(zip(pp["u"].astype(int), pp["v"].astype(int)))
    forward = sum((a, b) in directed for a, b in uv)
    reverse_only = sum(((a, b) not in directed) and ((b, a) in directed) for a, b in uv)

    # loss concentration: boxes with the worst point-placement rates
    by_box = placed.groupby(p["boxName"]).agg(["mean", "size"])
    worst = by_box[by_box["size"] >= 500].sort_values("mean").head(3)

    linked = ev["u"].notna()
    via = ev["link_via"].value_counts()

    rows = [
        ("rides total", p["traj_id"].nunique()),
        ("rides fully matched (>=90% of points)", int((per_ride >= 0.9).sum())),
        ("rides partially matched (10-90%)", int(((per_ride >= 0.1) & (per_ride < 0.9)).sum())),
        ("rides failed (<10% of points)", int((per_ride < 0.1).sum())),
        ("points placed on an edge", f"{placed.mean():.0%}"),
        ("placed points: travel along digitized direction", f"{forward / len(uv):.0%}"),
        ("placed points: travel against oneway direction", f"{reverse_only / len(uv):.0%}"),
        ("events total", len(ev)),
        ("events linked to an edge", f"{linked.mean():.0%}"),
        ("  linked via anchor point", int(via.get("anchor", 0))),
        ("  linked via first/last fallback", int(via.get("first", 0) + via.get("last", 0))),
        ("  unlinked (burst in unmatched stretch)", int(via.get("none", 0))),
        ("events with an edge class", f"{ev['edge_class'].notna().mean():.0%}"
         if "edge_class" in ev else "n/a"),
    ]
    for name, r in worst.iterrows():
        rows.append((f"lowest placement: {name}", f"{r['mean']:.0%} of {int(r['size'])} pts"))

    s = pd.DataFrame(rows, columns=["metric", "value"])
    s.to_csv(path, index=False)
    print(f"[summary] -> {path}")
    return s


def _safe(s):
    return re.sub(r"[^\w.-]+", "_", str(s))[:60]


def _edge_geom_lookup(edges_path=EDGES_PATH):
    """Map sorted (u, v) -> edge geometry, or None if it can't be built."""
    if not Path(edges_path).exists():
        return None
    edges = gpd.read_file(edges_path)
    if not {"u", "v"}.issubset(edges.columns):
        return None
    g = gpd.GeoDataFrame({"_uv": _uv_key(edges)}, geometry=edges.geometry, crs=edges.crs)
    g = g.dropna(subset=["_uv"]).drop_duplicates(subset="_uv")
    return g.set_index("_uv")["geometry"]


def _matched_edges_for(t, edge_geom):
    """Matched edge geometries for one trajectory's points (deduped), or None."""
    if edge_geom is None:
        return None
    seen = list(dict.fromkeys(
        k for k in _uv_key(t) if k is not None and k in edge_geom.index))
    if not seen:
        return None
    return gpd.GeoSeries([edge_geom[k] for k in seen], crs=edge_geom.crs)


def plot_match_per_box(matched_points, edges_path=EDGES_PATH, out_dir=PER_BOX_MATCH_DIR):
    """One page per box: raw GPS points sized by value + matched edges on top."""
    edge_geom = _edge_geom_lookup(edges_path)
    if edge_geom is None:
        print("[plot] no edge geometry (missing file or u/v) -> point-recolour fallback")

    out_dir.mkdir(parents=True, exist_ok=True)
    for bid, b in matched_points.groupby("boxId", sort=False):
        name = b["boxName"].iloc[0] if "boxName" in b else ""
        trajs = sorted(b.groupby("traj_id", sort=False), key=lambda kv: -len(kv[1]))
        ncols = 4
        nrows = -(-len(trajs) // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for ax, (tid, t) in zip(axes, trajs):
            t = t.sort_values("createdAt")
            placed = t["u"].notna().mean()
            ax.plot(t.geometry.x, t.geometry.y, color="0.8", linewidth=0.5, zorder=0)
            det = t[t["value"] > 0]
            if not det.empty:
                vv = det["value"].to_numpy(float)
                s = 3 + 40 * np.sqrt(np.clip(vv, 0, VALUE_VMAX) / VALUE_VMAX)
                ax.scatter(det.geometry.x, det.geometry.y, s=s, c="0.5",
                           alpha=0.5, edgecolors="none", zorder=1)
            me = _matched_edges_for(t, edge_geom)
            if me is not None:
                me.plot(ax=ax, color="crimson", linewidth=1.5, alpha=0.8, zorder=2)
            else:
                mp = t[t["u"].notna()]
                ax.scatter(mp.geometry.x, mp.geometry.y, s=8, c="crimson",
                           alpha=0.7, edgecolors="none", zorder=2)
            ax.set_title(f"{str(tid)[-4:]} | {len(t)} pts | {placed:.0%} placed", fontsize=8)
            ax.set_aspect("equal")
            ax.set_axis_off()

        for ax in axes[len(trajs):]:
            ax.set_visible(False)
        fig.suptitle(f"{name}  [{bid}]  — matched edges (crimson) vs GPS", fontsize=13, y=1.005)
        p = out_dir / f"{_safe(name)}__{_safe(bid)}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"[plot] per-box match validation -> {out_dir}/")


def _ride_lines(points):
    """One LineString per ride, in travel order — the ridden tracks for the map."""
    recs = []
    for tid, t in points.groupby("traj_id"):
        t = t.sort_values("createdAt")
        if len(t) >= 2:
            recs.append({"traj_id": tid, "geometry": LineString(t.geometry.values)})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=points.crs)


def fig_coverage_map(matched_points, matched_events):
    """Matched rides and overtake events over the whole cyclable network — how much
    of the city the collection actually reaches."""
    edges = gpd.read_file(EDGES_PATH)
    lines = _ride_lines(matched_points)

    xlo, xhi = matched_points.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = matched_points.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)

    fig, ax = plt.subplots(figsize=(11, 11))
    edges.plot(ax=ax, color=NET, linewidth=0.4, zorder=0)
    lines.plot(ax=ax, color=RIDE, linewidth=0.55, alpha=0.35, zorder=1)
    matched_events.plot(ax=ax, color=EVENT, markersize=7, alpha=0.55,
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
               label=f"overtake events ({len(matched_events):,})"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=12.5,
                    handletextpad=0.7, borderpad=0.9, labelspacing=0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.85)
    ax.set_title("Map-matched rides and car-overtake events — Münster",
                 fontsize=17, color=INK, loc="left", pad=12)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / "task3_coverage_map.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {p}")


def main():
    if MATCHED_POINTS_PATH.exists() and MATCHED_EVENTS_PATH.exists():
        print(f"[skip] {MATCHED_POINTS_PATH.name} exists -> summary + plots only "
              "(delete it to force a re-match)")
        matched_points = gpd.read_file(MATCHED_POINTS_PATH)
        matched_points["createdAt"] = pd.to_datetime(matched_points["createdAt"], utc=True)
        matched_events = gpd.read_file(MATCHED_EVENTS_PATH)
        G = ox.load_graphml(GRAPH_PATH)
    else:
        points = gpd.read_file(POINTS_PATH)
        points["createdAt"] = pd.to_datetime(points["createdAt"], utc=True)
        events = gpd.read_file(EVENTS_PATH)
        print(f"[load] {points['traj_id'].nunique()} trajectories, "
              f"{len(points)} points, {len(events)} events")

        m, G = build_inmem_map()
        matched_points = match_all(points, m)
        matched_events = join_edge_class(link_events(events, matched_points))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for p in (MATCHED_POINTS_PATH, MATCHED_EVENTS_PATH):
            p.unlink(missing_ok=True)   # avoid stale extra layers in the gpkg
        matched_points.to_file(MATCHED_POINTS_PATH, driver="GPKG")
        matched_events.to_file(MATCHED_EVENTS_PATH, driver="GPKG")
        print(f"saved: {MATCHED_POINTS_PATH.name}, {MATCHED_EVENTS_PATH.name}")

    match_summary(matched_points, matched_events, G)
    plot_match_per_box(matched_points)
    fig_coverage_map(matched_points, matched_events)


if __name__ == "__main__":
    main()