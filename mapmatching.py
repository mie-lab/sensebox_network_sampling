"""Map-match good trajectories to the directed bike network with Leuven, then
attach each overtake event to the edge it happened on.

Pipeline position: run AFTER trajectory_diagnostics.py (defines good trajectories)
and senseboxbike_preprocessing.py (writes scoped points + events with point ids).

Matching is on the DIRECTED graph: a trajectory is an ordered, directed path, so
Leuven resolves which of two parallel edges (road vs sidepath, or the two
directions of a street) each observation belongs to. Each matched observation
gets an edge_id; an event inherits the edge of its closest-approach point
(closest_point_id), falling back to its first/last point if that one was dropped.

Outputs (output/):
  matched_points.gpkg   points with (u, v, key) edge id of the matched edge
  matched_events.gpkg   events with edge id + edge_class joined from the network
"""


from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import logging
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

# Leuven tuning for ~1 Hz urban cycling GPS. max_dist: furthest an observation
# may sit from its edge; obs_noise: GPS sigma in metres; non_emitting handles
# stretches with no nearby node between observations.
MAX_DIST = 50
OBS_NOISE = 15
MAX_DIST_INIT = 60
MIN_PROB_NORM = 0.001


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
    # matcher.lattice_best holds one matched node-pair (edge) per observation
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


def join_edge_class(events, edges_path=EDGES_PATH):
    """Attach edge_class via an undirected node pair (match may return either order)."""
    if not Path(edges_path).exists():
        print(f"[class] {edges_path} not found -> skipping edge_class join")
        return events
    edges = gpd.read_file(edges_path)
    if not {"u", "v"}.issubset(edges.columns):
        print("[class] edges file lacks u/v columns -> skipping")
        return events

    def uv_key(df):
        return [tuple(sorted((a, b))) if pd.notna(a) and pd.notna(b) else None
                for a, b in zip(df["u"], df["v"])]

    cls = edges.assign(_uv=uv_key(edges))[["_uv", "edge_class"]].dropna(subset=["_uv"])
    cls = cls.drop_duplicates(subset="_uv")
    ev = events.assign(_uv=uv_key(events)).merge(cls, on="_uv", how="left")
    return ev.drop(columns="_uv")


if __name__ == "__main__":
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
