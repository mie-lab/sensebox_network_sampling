"""Map-match good trajectories to the bike network with Leuven, attach each
overtake event to the edge it happened on, and plot per-box validation.

Run AFTER trajectory_diagnostics.py and senseboxbike_preprocessing.py.
If matched_points/matched_events already exist, matching is skipped and only
the validation plots are (re)drawn; delete them to force a re-match.

A trajectory is an ordered path, so the matcher resolves which of two parallel
edges (road vs sidepath) each point belongs to; an event inherits the edge of
its closest-approach point, falling back to its first/last point.

Outputs (output/):
  matched_points.gpkg   points with matched edge (u, v)
  matched_events.gpkg   events with edge (u, v) + edge_class
  matched_per_box/      one validation page per box
"""
from pathlib import Path
import logging
import re

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger("leuvenmapmatching").setLevel(logging.WARNING)

from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap

from senseboxbike_preprocessing import GRAPH_PATH, CRS_METRIC

OUT_DIR = Path("output")
POINTS_PATH = OUT_DIR / "trajectory_points.gpkg"
EVENTS_PATH = OUT_DIR / "overtake_events.gpkg"
EDGES_PATH = Path("input/muenster_edges_classified.gpkg")
MATCHED_POINTS_PATH = OUT_DIR / "matched_points.gpkg"
MATCHED_EVENTS_PATH = OUT_DIR / "matched_events.gpkg"
PER_BOX_MATCH_DIR = OUT_DIR / "matched_per_box"

# Leuven tuning for ~1 Hz urban cycling GPS. max_dist: furthest an observation
# may sit from its edge; obs_noise: GPS sigma in metres; non_emitting handles
# stretches with no nearby node between observations.
MAX_DIST = 50
OBS_NOISE = 15
MAX_DIST_INIT = 60
MIN_PROB_NORM = 0.001

VALUE_VMAX = 250  # clearance scale cap for point sizing


def build_inmem_map(graph_path=GRAPH_PATH):
    """InMemMap from the saved osmnx graph (already in CRS_METRIC)."""
    G = ox.load_graphml(graph_path)
    m = InMemMap("muenster", use_latlon=False, use_rtree=True, index_edges=True)
    for nid, data in G.nodes(data=True):
        m.add_node(int(nid), (float(data["x"]), float(data["y"])))
    for u, v, _ in G.edges(keys=True):
        u, v = int(u), int(v)
        m.add_edge(u, v)
        m.add_edge(v, u)
    m.purge()
    return m, G


def match_trajectory(matcher, t):
    """Match one time-ordered trajectory; return edge id per point (or None)."""
    t = t.sort_values("createdAt")
    path = list(zip(t.geometry.x.values, t.geometry.y.values))
    states, _ = matcher.match(path)
    if not states:
        return pd.Series([None] * len(t), index=t.index)
    edges = [(matcher.lattice_best[i].edge_m.l1, matcher.lattice_best[i].edge_m.l2)
             if i < len(matcher.lattice_best) else None
             for i in range(len(t))]
    return pd.Series(edges, index=t.index)


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
        print(f"[match] {i}/{n_total}  {tid}  {len(t)} pts  "
              f"{'matched' if ok else 'FAILED':7s}  ({edges.notna().mean():.0%} placed)")
    out["edge"] = pd.concat(matched)
    out["u"] = out["edge"].map(lambda e: e[0] if isinstance(e, tuple) else None)
    out["v"] = out["edge"].map(lambda e: e[1] if isinstance(e, tuple) else None)
    print(f"[match] done: {n_ok}/{n_total} trajectories matched, "
          f"{out['edge'].notna().mean():.0%} of all points placed")
    return out.drop(columns="edge")


def link_events(events, matched_points):
    """Give each event the edge of its closest point, falling back to first/last."""
    edge_of = matched_points.set_index("point_id")[["u", "v"]]

    def pick(row):
        for col in ("closest_point_id", "first_point_id", "last_point_id"):
            pid = row.get(col)
            if pid in edge_of.index:
                u, v = edge_of.loc[pid, ["u", "v"]]
                if pd.notna(u):
                    return pd.Series({"u": u, "v": v})
        return pd.Series({"u": None, "v": None})

    uv = events.apply(pick, axis=1)
    ev = events.copy()
    ev["u"], ev["v"] = uv["u"], uv["v"]
    print(f"[link] {ev['u'].notna().mean():.0%} of events linked to an edge")
    return ev


def _uv_key(df):
    return [tuple(sorted((a, b))) if pd.notna(a) and pd.notna(b) else None
            for a, b in zip(df["u"], df["v"])]


def join_edge_class(events, edges_path=EDGES_PATH):
    """Attach edge_class via an undirected node pair (match may return either order)."""
    if not Path(edges_path).exists():
        print(f"[class] {edges_path} not found -> skipping edge_class join")
        return events
    edges = gpd.read_file(edges_path)
    if not {"u", "v"}.issubset(edges.columns):
        print("[class] edges file lacks u/v columns -> skipping")
        return events
    cls = edges.assign(_uv=_uv_key(edges))[["_uv", "edge_class"]].dropna(subset=["_uv"])
    cls = cls.drop_duplicates(subset="_uv")
    ev = events.assign(_uv=_uv_key(events)).merge(cls, on="_uv", how="left")
    return ev.drop(columns="_uv")


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
    """One page per box: raw GPS points sized by value + matched edges on top.

    Draws matched edge lines when geometry can be reconstructed from (u, v);
    otherwise falls back to recolouring the matched GPS points.
    """
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


if __name__ == "__main__":
    if MATCHED_POINTS_PATH.exists() and MATCHED_EVENTS_PATH.exists():
        print(f"[skip] {MATCHED_POINTS_PATH.name} exists -> plotting only "
              "(delete it to re-match)")
        matched_points = gpd.read_file(MATCHED_POINTS_PATH)
        matched_points["createdAt"] = pd.to_datetime(matched_points["createdAt"], utc=True)
    else:
        points = gpd.read_file(POINTS_PATH)
        points["createdAt"] = pd.to_datetime(points["createdAt"], utc=True)
        events = gpd.read_file(EVENTS_PATH)
        print(f"[load] {points['traj_id'].nunique()} trajectories, "
              f"{len(points)} points, {len(events)} events")

        m, _ = build_inmem_map()
        matched_points = match_all(points, m)
        matched_events = join_edge_class(link_events(events, matched_points))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        matched_points.to_file(MATCHED_POINTS_PATH, driver="GPKG")
        matched_events.to_file(MATCHED_EVENTS_PATH, driver="GPKG")
        print(f"saved: {MATCHED_POINTS_PATH.name}, {MATCHED_EVENTS_PATH.name}")

        if "edge_class" in matched_events.columns:
            print("\novertakes per riding regime:")
            print(matched_events.groupby("edge_class").size()
                  .sort_values(ascending=False).to_string())

    plot_match_per_box(matched_points)