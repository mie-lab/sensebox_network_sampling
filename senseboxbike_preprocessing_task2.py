"""Extract overtake events from the quality-checked rides.

Reads the segmented points and the keep verdict written by diagnostics_task2.py
(run that first), keeps only approved rides, and collapses bursts of high
"Overtaking Manoeuvre" probability (man_p >= MAN_P_TAU, gaps <= MERGE_GAP_S
merged) into one event per car pass.

Each event carries first/last/anchor point ids so mapmatching_task3.py can pin it onto a matched edge;
the anchor is the highest-probability point (best guess for the moment the car was alongside).

Writes to output/task2_trajectories/:
  trajectory_points_task2.gpkg   points of the kept rides (input to map-matching)
  overtake_events_task2.gpkg     one row per overtake event
  trajectory_summary_task2.csv   one row per ride
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString

from diagnostics_task2 import BOX_ID_COL, MAN_P_TAU, QUALITY_CSV, SEG_POINTS_PATH

OUT_DIR = Path("output/task2_trajectories")
POINTS_PATH = OUT_DIR / "trajectory_points_task2.gpkg"
EVENTS_PATH = OUT_DIR / "overtake_events_task2.gpkg"
SUMMARY_PATH = OUT_DIR / "trajectory_summary_task2.csv"
PLOT_PATH = OUT_DIR / "trajectories_task2.png"

CLOSE_PASS_CM = 150                  # ordinal descriptor threshold; NOT meters-true clearance
MERGE_GAP_S = 5                      # gap below which two bursts are one event
                                     # (the app max-pools man_p over 2 s, which can split one pass)
MIN_LENGTH_KM = 0.2
MAX_LEGEND = 15


def load_kept_points(points_path=SEG_POINTS_PATH, quality_csv=QUALITY_CSV):
    """Segmented points of the rides that passed diagnostics."""
    pts = gpd.read_file(points_path)
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    q = pd.read_csv(quality_csv)
    good = set(q.loc[q["keep"], "traj_id"])
    kept = pts[pts["traj_id"].isin(good)].reset_index(drop=True)
    print(f"[load] {kept['traj_id'].nunique()}/{pts['traj_id'].nunique()} rides kept "
          f"({len(kept)} points) from {points_path.name}")
    return kept


def extract_overtake_events(gdf, man_p_tau=MAN_P_TAU, merge_gap_s=MERGE_GAP_S):
    """Collapse high-manoeuvre-probability bursts into one row per overtake event."""
    g = gdf[gdf["man_p"] >= man_p_tau].copy()
    g = g.sort_values(["traj_id", "createdAt"])
    cols = ["event_uid", "traj_id", "event_id", "boxId", "start", "end",
            "duration_s", "n_samples", "max_man_p", "min_clearance_cm",
            "mean_clearance_cm", "is_close", "first_point_id", "last_point_id",
            "closest_point_id", "geometry"]
    if g.empty:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=gdf.crs)

    dt = g.groupby("traj_id")["createdAt"].diff().dt.total_seconds()
    new_event = dt.isna() | (dt > merge_gap_s)
    g["event_id"] = new_event.groupby(g["traj_id"]).cumsum()

    def one_event(e):
        e = e.sort_values("createdAt")
        anchor = e.loc[e["man_p"].idxmax()]
        nz = e.loc[e["value"] > 0, "value"]
        min_d = nz.min() if len(nz) else np.nan
        return pd.Series({
            "boxId": e["boxId"].iloc[0],
            "start": e["createdAt"].min(),
            "end": e["createdAt"].max(),
            "duration_s": (e["createdAt"].max() - e["createdAt"].min()).total_seconds(),
            "n_samples": len(e),
            "max_man_p": e["man_p"].max(),
            "min_clearance_cm": min_d,
            "mean_clearance_cm": nz.mean() if len(nz) else np.nan,
            "is_close": bool(min_d < CLOSE_PASS_CM) if pd.notna(min_d) else False,
            "first_point_id": e["point_id"].iloc[0],
            "last_point_id": e["point_id"].iloc[-1],
            "closest_point_id": anchor["point_id"],
            "geometry": anchor.geometry,
        })

    events = (g.groupby(["traj_id", "event_id"], group_keys=True)
              .apply(one_event, include_groups=False).reset_index())
    events["event_uid"] = events["traj_id"] + "_ev" + events["event_id"].astype(str)
    return gpd.GeoDataFrame(events[cols], geometry="geometry", crs=gdf.crs)


def summarise(gdf, events, min_length_km=MIN_LENGTH_KM):
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
            "mean_overtake_cm": eg["min_clearance_cm"].mean(),
            "mean_event_duration_s": eg["duration_s"].mean(),
        })
        summary = summary.join(ev_stats)
    for c in ["n_overtakes", "n_close_passes"]:
        summary[c] = summary.get(c, pd.Series(0, index=summary.index)).fillna(0).astype(int)

    usable = summary["length_km"].where(summary["length_km"] >= min_length_km)
    summary["overtake_rate_per_km"] = summary["n_overtakes"] / usable
    return summary.sort_values("overtake_rate_per_km", ascending=False)


def plot_trajectories(gdf, color_by=BOX_ID_COL, path=PLOT_PATH, max_legend=MAX_LEGEND):
    records = []
    for traj_id, t in gdf.groupby("traj_id"):
        t = t.sort_values("createdAt")
        if len(t) >= 2:
            records.append({"traj_id": traj_id, color_by: t[color_by].iloc[0],
                            "geometry": LineString(t.geometry.values)})
    lines = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)

    cats = lines[color_by].value_counts().index
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, cat in enumerate(cats):
        sub = lines[lines[color_by] == cat]
        label = f"{str(cat)[:12]} ({len(sub)})" if i < max_legend else None
        sub.plot(ax=ax, color=cmap(i % cmap.N), linewidth=0.7, alpha=0.7,
                 label=label, zorder=1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"senseBox trajectory coverage by {color_by}")
    extra = len(cats) - min(len(cats), max_legend)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
              title=color_by + (f"  (+{extra} more)" if extra > 0 else ""), frameon=False)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")


if __name__ == "__main__":
    pts = load_kept_points()
    events = extract_overtake_events(pts)
    summ = summarise(pts, events)
    print(f"[events] {len(events)} overtake events "
          f"({int(events['is_close'].sum())} with proximity < {CLOSE_PASS_CM} cm) "
          f"over {summ['length_km'].sum():.0f} km "
          f"-> {len(events) / summ['length_km'].sum():.2f} events/km")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (POINTS_PATH, EVENTS_PATH):
        p.unlink(missing_ok=True)   # a leftover file would keep stale layers alongside the new one
    pts.to_file(POINTS_PATH, driver="GPKG")
    events.to_file(EVENTS_PATH, driver="GPKG")
    summ.to_csv(SUMMARY_PATH)
    print(f"saved: {POINTS_PATH.name}, {EVENTS_PATH.name}, {SUMMARY_PATH.name}")
    try:
        plot_trajectories(pts)
    except Exception as e:                      # plots must never block the data outputs
        print(f"[plot] FAILED ({e}) -> data outputs are saved, only the figure is missing")
