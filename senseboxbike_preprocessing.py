"""senseBox overtaking points -> per-rider trajectories, overtake events, summary.

Points are split into trajectories by boxId then temporal gap. GPS glitches are
dropped by clipping to the saved network graph's bbox. Overtake events collapse
the ~1Hz bursts into one row per pass, linked to trajectories by traj_id.
"""
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString

# --- config -----------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:25832"            # UTM 32N, same as the network graph
BOX_ID_COL = "boxId"

GRAPH_PATH = Path("input/muenster_bike.graphml")
INPUT_CSV = Path("input/opensensemap_overtaking_distances.csv")
OUT_DIR = Path("output")
PLOT_PATH = OUT_DIR / "trajectories.png"
POINTS_PATH = OUT_DIR / "trajectory_points.gpkg"
SUMMARY_PATH = OUT_DIR / "trajectory_summary.csv"
EVENTS_PATH = OUT_DIR / "overtake_events.gpkg"

GAP_MINUTES = 10                     # gap that starts a new trajectory
MIN_POINTS = 5                       # drop trajectories shorter than this
OVERTAKE_THRESHOLD = 0               # value (cm) above which a point is an overtake
CLOSE_PASS_CM = 150                  # legal minimum clearance in town (DE)
MERGE_GAP_S = 3                      # gap below which two bursts are one event
BBOX_PAD_M = 2000
MAX_LEGEND = 15
MIN_LENGTH_KM = 0.2                  # below this a per-km rate is meaningless


def bbox_from_graph(graph_path=GRAPH_PATH, pad_m=BBOX_PAD_M):
    """Study-area bbox (minx, miny, maxx, maxy) in metres from the saved graph."""
    G = ox.load_graphml(graph_path)
    nodes, _ = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = nodes.total_bounds
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def load_points(csv_path=INPUT_CSV, bbox=None):
    """Read the CSV into a projected, time-sorted GeoDataFrame; clip to bbox."""
    df = pd.read_csv(csv_path)
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "createdAt", BOX_ID_COL])

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        n_before = len(gdf)
        gdf = gdf.cx[minx:maxx, miny:maxy]
        if n_before - len(gdf):
            print(f"[load] dropped {n_before - len(gdf)} points outside study area")

    return gdf.sort_values([BOX_ID_COL, "createdAt"]).reset_index(drop=True)


def segment_trajectories(gdf, gap_minutes=GAP_MINUTES, min_points=MIN_POINTS):
    """Split each box's stream into trajectories on temporal gaps; drop short ones."""
    g = gdf.copy()
    dt = g.groupby(BOX_ID_COL)["createdAt"].diff()
    new_traj = dt.isna() | (dt > pd.Timedelta(minutes=gap_minutes))
    g["traj_id"] = (
        g[BOX_ID_COL].astype(str) + "_"
        + new_traj.groupby(g[BOX_ID_COL]).cumsum().astype(str)
    )
    counts = g.groupby("traj_id")["traj_id"].transform("size")
    return g[counts >= min_points].reset_index(drop=True)


def bearing_along(gdf_traj):
    """Per-point travel bearing (0-360, 0=N) from the previous point."""
    x = gdf_traj.geometry.x.values
    y = gdf_traj.geometry.y.values
    east = np.diff(x, prepend=x[0])
    north = np.diff(y, prepend=y[0])
    bearing = np.degrees(np.arctan2(east, north)) % 360  # east/north -> compass bearing
    bearing[0] = bearing[1] if len(bearing) > 1 else 0.0
    return bearing


def add_bearings(gdf):
    """Attach a per-point travel bearing, computed within each trajectory."""
    out = gdf.copy().sort_values(["traj_id", "createdAt"])
    parts = [pd.Series(bearing_along(t), index=t.index)
             for _, t in out.groupby("traj_id", sort=False)]
    out["bearing"] = pd.concat(parts)
    return out


def trajectory_lines(gdf, color_by=BOX_ID_COL):
    """One LineString per trajectory, with its colour key (built in one pass)."""
    records = []
    for traj_id, t in gdf.groupby("traj_id"):
        t = t.sort_values("createdAt")
        if len(t) < 2:
            continue
        records.append({
            "traj_id": traj_id,
            color_by: t[color_by].iloc[0],
            "geometry": LineString(t.geometry.values),
        })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)


def extract_overtake_events(gdf, overtake_threshold=OVERTAKE_THRESHOLD,
                            merge_gap_s=MERGE_GAP_S):
    """Collapse nonzero-reading bursts into one row per overtake event.

    Linked to its trajectory by traj_id; identified globally by event_uid.
    geometry is the point of closest approach (min clearance).
    """
    g = gdf[gdf["value"] > overtake_threshold].copy()
    g = g.sort_values(["traj_id", "createdAt"])
    cols = ["event_uid", "traj_id", "event_id", "boxId", "start", "end",
            "duration_s", "n_samples", "min_clearance_cm", "mean_clearance_cm",
            "is_close", "geometry"]
    if g.empty:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=gdf.crs)

    dt = g.groupby("traj_id")["createdAt"].diff().dt.total_seconds()
    new_event = dt.isna() | (dt > merge_gap_s)
    g["event_id"] = new_event.groupby(g["traj_id"]).cumsum()

    def one_event(e):
        closest = e.loc[e["value"].idxmin()]
        return pd.Series({
            "boxId": e["boxId"].iloc[0],
            "start": e["createdAt"].min(),
            "end": e["createdAt"].max(),
            "duration_s": (e["createdAt"].max() - e["createdAt"].min()).total_seconds(),
            "n_samples": len(e),
            "min_clearance_cm": e["value"].min(),
            "mean_clearance_cm": e["value"].mean(),
            "is_close": bool(e["value"].min() < CLOSE_PASS_CM),
            "geometry": closest.geometry,
        })

    events = (g.groupby(["traj_id", "event_id"], group_keys=True)
              .apply(one_event, include_groups=False).reset_index())
    events["event_uid"] = events["traj_id"] + "_ev" + events["event_id"].astype(str)
    return gpd.GeoDataFrame(events[cols], geometry="geometry", crs=gdf.crs)


def summarise(gdf, events=None, min_length_km=MIN_LENGTH_KM):
    """One row per trajectory: geometry/speed plus event-based overtake stats.

    Counts are EVENTS, not raw samples. overtake_rate_per_km is NaN for
    trajectories shorter than min_length_km.
    """
    if events is None:
        events = extract_overtake_events(gdf)

    def length_km(t):
        pts = t.sort_values("createdAt").geometry.values
        return LineString(pts).length / 1000 if len(pts) >= 2 else 0.0

    grp = gdf.groupby("traj_id")
    summary = pd.DataFrame({
        "boxId": grp["boxId"].first(),
        "boxName": grp["boxName"].first(),
        "start": grp["createdAt"].min(),
        "end": grp["createdAt"].max(),
        "n_points": grp.size(),
        "length_km": grp.apply(length_km),
    })
    summary["duration_min"] = (summary["end"] - summary["start"]).dt.total_seconds() / 60
    summary["mean_speed_kmh"] = (
        summary["length_km"] / (summary["duration_min"] / 60).replace(0, np.nan)
    )

    if len(events):
        eg = events.groupby("traj_id")
        ev_stats = pd.DataFrame({
            "n_overtakes": eg.size(),
            "n_close_passes": eg["is_close"].sum(),
            "min_overtake_cm": eg["min_clearance_cm"].min(),
            "mean_overtake_cm": eg["min_clearance_cm"].mean(),  # mean of per-event minima
            "mean_event_duration_s": eg["duration_s"].mean(),
        })
    else:
        ev_stats = pd.DataFrame(columns=[
            "n_overtakes", "n_close_passes", "min_overtake_cm",
            "mean_overtake_cm", "mean_event_duration_s"])

    summary = summary.join(ev_stats)
    for c in ["n_overtakes", "n_close_passes"]:
        summary[c] = summary[c].fillna(0).astype(int)

    usable = summary["length_km"].where(summary["length_km"] >= min_length_km)
    summary["overtake_rate_per_km"] = summary["n_overtakes"] / usable
    return summary.sort_values("overtake_rate_per_km", ascending=False)


def plot_trajectories(gdf, color_by=BOX_ID_COL, path=PLOT_PATH, edges=None, max_legend=MAX_LEGEND):
    """Coverage plot: one LineString per trajectory, coloured by color_by."""
    lines = trajectory_lines(gdf, color_by=color_by)
    print(f"[plot] {len(lines)} trajectories | bounds {lines.total_bounds}")

    cats = lines[color_by].value_counts().index
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(12, 12))
    if edges is not None:
        edges.plot(ax=ax, color="0.9", linewidth=0.3, zorder=0)

    for i, cat in enumerate(cats):
        sub = lines[lines[color_by] == cat]
        label = f"{str(cat)[:12]} ({len(sub)})" if i < max_legend else None
        sub.plot(ax=ax, color=cmap(i % cmap.N), linewidth=0.7, alpha=0.7,
                 label=label, zorder=1)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"senseBox trajectory coverage by {color_by}")
    extra = len(cats) - min(len(cats), max_legend)
    legend_title = color_by + (f"  (+{extra} more)" if extra > 0 else "")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
              title=legend_title, frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")
    return path


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_CSV
    bbox = bbox_from_graph()

    pts = add_bearings(segment_trajectories(load_points(csv, bbox=bbox)))
    events = extract_overtake_events(pts)
    summ = summarise(pts, events=events)

    print(f"{pts['traj_id'].nunique()} trajectories from "
          f"{pts[BOX_ID_COL].nunique()} boxes, {len(pts)} points")
    print(f"{len(events)} overtake events ({int(events['is_close'].sum())} close)")
    print(summ.head(15).to_string())

    orphan = ~events["traj_id"].isin(summ.index)
    if orphan.any():
        print(f"[warn] {orphan.sum()} events have no matching trajectory")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_trajectories(pts)
    pts.to_file(POINTS_PATH, driver="GPKG")
    events.to_file(EVENTS_PATH, driver="GPKG")
    summ.to_csv(SUMMARY_PATH)
    print(f"saved: {POINTS_PATH.name}, {EVENTS_PATH.name}, {SUMMARY_PATH.name}")

    bbox = bbox_from_graph()
    pts = segment_trajectories(load_points(bbox=bbox))
    v = pts["value"]

    print("=" * 60)
    print("1. RAW VALUE DISTRIBUTION (all points)")
    print("=" * 60)
    print(f"total points        : {len(v):,}")
    print(f"value == 0          : {(v == 0).sum():,}  ({(v == 0).mean():.1%})")
    print(f"value  > 0          : {(v > 0).sum():,}  ({(v > 0).mean():.1%})")
    print(f"\nnonzero value stats (cm):")
    nz = v[v > 0]
    print(nz.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).to_string())

    print("\n" + "=" * 60)
    print("2. LOW-END DETAIL (is there a noise floor?)")
    print("=" * 60)
    # how many nonzero readings sit at each low cm value
    low = nz[nz <= 30]
    print(f"nonzero readings <= 30 cm: {len(low):,} ({len(low) / len(nz):.1%} of nonzero)")
    print("counts per cm, 1..30:")
    print(low.round().value_counts().sort_index().to_string())

    print("\n" + "=" * 60)
    print("3. STUCK-SENSOR CHECK (overtake share per trajectory)")
    print("=" * 60)
    # share of points that are nonzero, per trajectory
    share = pts.assign(nz=(pts["value"] > 0)).groupby("traj_id")["nz"].mean()
    print("distribution of nonzero-share across trajectories:")
    print(share.describe(percentiles=[.5, .75, .9, .95, .99]).to_string())
    for thr in (0.3, 0.5, 0.7, 0.9):
        n = (share > thr).sum()
        print(f"  trajectories with >{thr:.0%} nonzero points: {n} "
              f"({n / len(share):.1%})")

    print("\n" + "=" * 60)
    print("4. EVENT DURATION (are 'events' single blips or real passes?)")
    print("=" * 60)
    ev = extract_overtake_events(pts)
    print(f"total events: {len(ev):,}")
    print("event duration (s) distribution:")
    print(ev["duration_s"].describe(percentiles=[.5, .9, .99]).to_string())
    print(f"\nevents lasting 0 s (single sample): {(ev['duration_s'] == 0).sum():,} "
          f"({(ev['duration_s'] == 0).mean():.1%})")
    print("n_samples per event:")
    print(ev["n_samples"].describe(percentiles=[.5, .9, .99]).to_string())