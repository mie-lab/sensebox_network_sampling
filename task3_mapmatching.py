"""Map-match the kept rides onto the street network.

Each ride is matched as an ordered path (Leuven HMM) in travel order, so the (u, v) on every
point is the direction it was ridden. Only geometry is used, every attribute an edge carries
is joined later, in task 4.

Matching is skipped if the matched points already exist; delete the file to force a re-match.

Writes to output/task3_matching/:
  task3_matched_points.gpkg      points with their matched edge (u, v), in travel order
  task3_match_summary.csv        match and direction descriptives
  task3_placement_per_box.csv    one row per box, worst total loss first
  task3_unplaced_points.png      where matching failed, over the network
  task3_matched_per_box/         one validation page per box
"""

import logging
from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd

from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
from task1_network import GRAPH_PATH
from task2a_ride_quality import SEG_POINTS_PATH

logging.getLogger("leuvenmapmatching").setLevel(logging.WARNING)

OUT_DIR = Path("output/task3_matching")
MATCHED_POINTS_PATH = OUT_DIR / "task3_matched_points.gpkg"
MATCH_SUMMARY_CSV = OUT_DIR / "task3_match_summary.csv"
PLACEMENT_CSV = OUT_DIR / "task3_placement_per_box.csv"
UNPLACED_FIG = OUT_DIR / "task3_unplaced_points.png"
PER_BOX_DIR = OUT_DIR / "task3_matched_per_box"

# Leuven housekeeping
MAX_DIST = 50
MAX_DIST_INIT = 60
OBS_NOISE = 15
MIN_PROB_NORM = 0.001


def load_kept_points(points_path=SEG_POINTS_PATH):
    """The points of the rides task 2a passed, already in the CRS the graph uses."""
    points = gpd.read_file(points_path)
    points = points[points["keep"]].reset_index(drop=True)
    points["createdAt"] = pd.to_datetime(points["createdAt"], utc=True)
    print(f"[load] {points['traj_id'].nunique()} rides, {len(points)} points "
          f"from {points_path.name}")
    return points


def build_inmem_map(graph_path=GRAPH_PATH):
    """InMemMap from the saved osmnx graph.
    Both directions go in even for oneways, or a contraflow ride would not match at all."""
    G = ox.load_graphml(graph_path)
    road_map = InMemMap("muenster", use_latlon=False, use_rtree=True, index_edges=True)
    for node_id, data in G.nodes(data=True):
        road_map.add_node(int(node_id), (float(data["x"]), float(data["y"])))
    for u, v, _ in G.edges(keys=True):
        road_map.add_edge(int(u), int(v))
        road_map.add_edge(int(v), int(u))
    road_map.purge()
    return road_map, G


def match_trajectory(matcher, ride):
    """The travelled edge per point, or None where the ride could not be placed.
    When the lattice dies mid-ride, matching resumes at the last point that did match."""
    ride = ride.sort_values("createdAt")
    edges = pd.Series([None] * len(ride), index=ride.index, dtype=object)
    start = 0
    while start <= len(ride) - 2:
        rest = ride.iloc[start:]
        states, last_idx = matcher.match(list(zip(rest.geometry.x, rest.geometry.y)))
        if states:
            for step in matcher.lattice_best:
                if step.is_emitting() and step.edge_m is not None:
                    i = start + int(step.obs)
                    edges.iloc[i] = (int(step.edge_m.l1), int(step.edge_m.l2))
        # leuven returns the index one before the last match, so the +1 lands back on it
        start += max(int(last_idx), 0) + 1
    return edges


def match_all(points, road_map):
    """Every ride matched in travel order, with (u, v) columns on each point."""
    matcher = DistanceMatcher(
        road_map, max_dist=MAX_DIST, max_dist_init=MAX_DIST_INIT, obs_noise=OBS_NOISE,
        min_prob_norm=MIN_PROB_NORM, non_emitting_states=True,
    )
    out = points.sort_values(["traj_id", "createdAt"])
    rides = list(out.groupby("traj_id", sort=False))
    print(f"Matching {len(rides)} rides...", flush=True)

    matched, n_ok = [], 0
    for i, (traj_id, ride) in enumerate(rides, 1):
        edges = match_trajectory(matcher, ride)
        ok = edges.notna().any()
        n_ok += ok
        matched.append(edges)
        if not ok or i % 50 == 0 or i == len(rides):
            print(f"  {i}/{len(rides)} {traj_id} {len(ride)} pts, "
                  f"{'matched' if ok else 'FAILED'} ({edges.notna().mean():.0%} placed)",
                  flush=True)

    edge = pd.concat(matched).reindex(out.index)   # comes back in ride order, not row order
    print(f"{n_ok}/{len(rides)} rides matched, "
          f"{edge.notna().mean():.0%} of all points placed")
    return out.assign(u=[e[0] if e is not None else None for e in edge],
                      v=[e[1] if e is not None else None for e in edge])


def match_summary(points, G, path=MATCH_SUMMARY_CSV):
    """How much matched, where the loss concentrates, and whether travel direction holds."""
    full, failed = 0.9, 0.1
    placed = points["u"].notna()
    per_ride = placed.groupby(points["traj_id"]).mean()

    directed = {(int(u), int(v)) for u, v, _ in G.edges(keys=True)}
    uv = zip(points.loc[placed, "u"].astype(int), points.loc[placed, "v"].astype(int))
    oneway = [(a, b) for a, b in uv if (a, b) not in directed or (b, a) not in directed]
    against = sum((a, b) not in directed for a, b in oneway)

    rows = [
        ("rides total", points["traj_id"].nunique()),
        (f"rides fully matched (>={full:.0%} of points)", int((per_ride >= full).sum())),
        (f"rides partially matched ({failed:.0%}-{full:.0%})",
         int(((per_ride >= failed) & (per_ride < full)).sum())),
        (f"rides failed (<{failed:.0%} of points)", int((per_ride < failed).sum())),
        ("points placed on an edge", f"{placed.mean():.0%}"),
        ("placed points: contraflow on a oneway",
         f"{against / len(oneway):.0%} of {len(oneway)} points on oneway edges"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)
    print(f"[csv] saved -> {path}")


def placement_per_box(points, path=PLACEMENT_CSV):
    """Every box's placement, worst total loss first."""
    placed = points["u"].notna()
    per_box = placed.groupby(points["boxId"]).agg(n_points="size", n_placed="sum")
    per_box["n_unplaced"] = per_box["n_points"] - per_box["n_placed"]
    per_box["placed_share"] = (per_box["n_placed"] / per_box["n_points"]).round(3)
    per_box["n_rides"] = points.groupby("boxId")["traj_id"].nunique()
    per_box["boxName"] = points.groupby("boxId")["boxName"].first()   # labels only, not unique

    cols = ["boxName", "n_rides", "n_points", "n_placed", "n_unplaced", "placed_share"]
    per_box.sort_values("n_unplaced", ascending=False)[cols].to_csv(path)
    print(f"[csv] saved -> {path}")


# ========= plotting =========


def _save(fig, path, announce=True):
    """Write and close a figure; the per-box pages are announced once, in main()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if announce:
        print(f"[fig] saved -> {path}")


def _scatter_ride(ax, ride):
    """A ride's GPS path."""
    ride = ride.sort_values("createdAt")
    ax.plot(ride.geometry.x, ride.geometry.y, color="lightgrey", linewidth=0.5, zorder=0)
    det = ride[ride["value"] > 0]
    ax.scatter(det.geometry.x, det.geometry.y, s=8, color="dimgrey", alpha=0.5,
               edgecolors="none", zorder=1)


def _uv_key(df):
    """Each row's (u, v) as one key that matches the street whichever way it was ridden."""
    return [tuple(sorted((int(a), int(b)))) if pd.notna(a) and pd.notna(b) else None
            for a, b in zip(df["u"], df["v"])]


def _edge_geometry(G):
    """Sorted (u, v) -> one edge geometry per node pair, from the graph the match ran on."""
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()
    keyed = edges.assign(_uv=_uv_key(edges)).dropna(subset=["_uv"])
    return keyed.drop_duplicates(subset="_uv").set_index("_uv")["geometry"]


def _matched_edges(ride, edge_geometry):
    """The distinct edge geometries one ride was matched onto, or None."""
    keys = list(dict.fromkeys(
        k for k in _uv_key(ride) if k is not None and k in edge_geometry.index))
    return gpd.GeoSeries([edge_geometry[k] for k in keys], crs=edge_geometry.crs) if keys else None


def fig_unplaced_points(matched_points, G, path=UNPLACED_FIG):
    """Where matching failed, over the network and the riding that produced it."""
    unplaced = matched_points[matched_points["u"].isna()]
    edges = ox.graph_to_gdfs(G, nodes=False)

    fig, ax = plt.subplots(figsize=(11, 11))
    edges.plot(ax=ax, color="lightgrey", linewidth=0.4, zorder=0)
    ax.scatter(matched_points.geometry.x, matched_points.geometry.y, s=2, color="blue",
               alpha=0.15, edgecolors="none", zorder=1)
    ax.scatter(unplaced.geometry.x, unplaced.geometry.y, s=8, color="crimson",
               alpha=0.6, edgecolors="none", zorder=2)

    xlo, xhi = matched_points.geometry.x.quantile([0.01, 0.99])
    ylo, yhi = matched_points.geometry.y.quantile([0.01, 0.99])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"Unplaced points in crimson over the ridden points in blue "
                 f"({len(unplaced)} of {len(matched_points)}, "
                 f"{len(unplaced) / len(matched_points):.1%})", loc="left", fontsize=11)
    _save(fig, path)


def fig_match_per_box(box, edge_geometry, out_dir=PER_BOX_DIR, ncols=4):
    """One page per box, one panel per ride: the GPS track with its matched edges on top."""
    box_id, name = box["boxId"].iloc[0], box["boxName"].iloc[0]
    rides = sorted(box.groupby("traj_id", sort=False), key=lambda kv: -len(kv[1]))
    nrows = -(-len(rides) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (traj_id, ride) in zip(axes.flat, rides):
        _scatter_ride(ax, ride)
        matched = _matched_edges(ride, edge_geometry)
        if matched is not None:
            matched.plot(ax=ax, color="crimson", linewidth=1.5, alpha=0.8, zorder=2)
        ax.set_title(f"ride ..{str(traj_id)[-4:]} | {len(ride)} pts | "
                     f"{ride['u'].notna().mean():.0%} placed", fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()
    for ax in axes.flat[len(rides):]:
        ax.set_visible(False)

    fig.suptitle(f"{name}  [{box_id}] - {len(rides)} rides, matched edges in crimson",
                 fontsize=11, y=1.005)
    stem = "__".join(re.sub(r"[^\w.-]+", "_", str(s))[:60] for s in (name, box_id))
    _save(fig, out_dir / f"{stem}.png", announce=False)


def main():
    if MATCHED_POINTS_PATH.exists():
        print(f"{MATCHED_POINTS_PATH.name} exists, summary and figures only "
              "(delete it to force a re-match)")
        matched_points = gpd.read_file(MATCHED_POINTS_PATH)
        matched_points["createdAt"] = pd.to_datetime(matched_points["createdAt"], utc=True)
        G = ox.load_graphml(GRAPH_PATH)
    else:
        road_map, G = build_inmem_map()
        matched_points = match_all(load_kept_points(), road_map)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        MATCHED_POINTS_PATH.unlink(missing_ok=True)  
        matched_points.to_file(MATCHED_POINTS_PATH, driver="GPKG")
        print(f"[gpkg] saved -> {MATCHED_POINTS_PATH}")

    match_summary(matched_points, G)
    placement_per_box(matched_points)

    fig_unplaced_points(matched_points, G)

    edge_geometry = _edge_geometry(G)
    for _, box in matched_points.groupby("boxId", sort=False):
        fig_match_per_box(box, edge_geometry)
    print(f"[fig] saved -> {PER_BOX_DIR}/ ({matched_points['boxId'].nunique()} per-box pages)")
    print("\nDONE")


if __name__ == "__main__":
    main()
