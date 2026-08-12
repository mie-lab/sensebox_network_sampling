"""Cut the raw points into rides, and decide which rides are usable.

A ride is dropped on any of four flags: blocked distance sensor, missing classifier
channel, non-cycling speed, or too short to measure a rate on.

Writes to output/task2_diagnostics/:
  task2a_segmented_points.gpkg     every point with its ride id, read by task 2b
  task2a_trajectory_quality.csv    one row per ride: stats, flags, keep verdict
  task2a_diagnostics_summary.csv   headline numbers and threshold sensitivity
  task2a_trajectories.png          the kept rides, coloured by box
  task2a_per_box/                  one page per box, one panel per ride
"""

from collections import namedtuple
from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString

from task1_network import GRAPH_PATH

DISTANCE_CSV = Path("input/muenster_overtaking_distance_2024-08_2026-08.csv")
MANOEUVRE_CSV = Path("input/muenster_overtaking_manoeuvre_2024-08_2026-08.csv")

OUT_DIR = Path("output/task2_diagnostics")
SEG_POINTS_PATH = OUT_DIR / "task2a_segmented_points.gpkg"
QUALITY_CSV = OUT_DIR / "task2a_trajectory_quality.csv"
SUMMARY_CSV = OUT_DIR / "task2a_diagnostics_summary.csv"
TRAJECTORIES_FIG = OUT_DIR / "task2a_trajectories.png"
PER_BOX_DIR = OUT_DIR / "task2a_per_box"

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:25832"    # UTM 32N, same as the network graph
BOX_ID_COL = "boxId"

BBOX_PAD_M = 2000
GAP_MINUTES = 5
MIN_POINTS = 5
MAN_P_TAU = 0.5

# thresholds of the four drop rules
SPEED_MIN_KMH, SPEED_MAX_KMH = 3.0, 40.0
STUCK_MIN_NONZERO = 20
STUCK_VARIATION_CM = 5
STUCK_IN_RANGE_MIN = 0.9
MAN_MISSING_MAX = 0.5
MIN_LENGTH_KM = 0.2

FLAG_COLS = ["f_speed", "f_stuck", "f_no_manoeuvre", "f_too_short"]

# one probe pair per threshold, looser first.
Rule = namedtuple("Rule", "flag kwarg loose strict")
RULES = [
    Rule("f_speed", "speed", (2, 50), (5, 30)),
    Rule("f_stuck", "min_nonzero", 40, 10),
    Rule("f_stuck", "var", 3, 8),
    Rule("f_stuck", "in_range", 0.99, 0.8),
    Rule("f_no_manoeuvre", "miss", 0.9, 0.3),
    Rule("f_too_short", "min_km", 0.1, 0.5),
]

VALUE_VMIN, VALUE_VMAX = 0, 250
MAX_LEGEND = 15


# ========= load sensor data =========


def bbox_from_graph(graph_path=GRAPH_PATH, pad_m=BBOX_PAD_M):
    """The network's own extent, padded, so a ride may stray outside the city."""
    G = ox.load_graphml(graph_path)
    nodes, _ = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = nodes.total_bounds
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def load_distance_channel(bbox, csv_path=DISTANCE_CSV):
    """The Overtaking Distance channel as points, clipped to bbox and ordered per box in time."""
    df = pd.read_csv(csv_path)
    n_raw = len(df)

    # one box labels every reading '%'; its values match the cm distribution but we drop it.
    df = df[df["unit"] == "cm"]
    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "createdAt", BOX_ID_COL])

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    minx, miny, maxx, maxy = bbox
    gdf = gdf.cx[minx:maxx, miny:maxy]
    print(f"[load] {n_raw} rows, {len(gdf)} points kept, {n_raw - len(gdf)} removed")
    return gdf.sort_values([BOX_ID_COL, "createdAt"]).reset_index(drop=True)


def add_manoeuvre_channel(gdf, csv_path=MANOEUVRE_CSV):
    """The classifier's 0-1 car-pass probability per point, NaN where it said nothing."""
    m = pd.read_csv(csv_path)
    m["createdAt"] = pd.to_datetime(m["createdAt"], utc=True)
    m = (m.rename(columns={"value": "man_p"})[[BOX_ID_COL, "createdAt", "man_p"]]
         .drop_duplicates(subset=[BOX_ID_COL, "createdAt"]))

    out = gdf.merge(m, on=[BOX_ID_COL, "createdAt"], how="left")
    print(f"[load] {out['man_p'].isna().mean():.0%} of points without a manoeuvre reading")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def segment_trajectories(gdf, gap_minutes=GAP_MINUTES, min_points=MIN_POINTS):
    """Cut each box's stream into rides; expects the points sorted per box in time."""
    g = gdf.copy()
    dt = g.groupby(BOX_ID_COL)["createdAt"].diff()
    new_traj = dt.isna() | (dt > pd.Timedelta(minutes=gap_minutes))
    g["traj_id"] = (
        g[BOX_ID_COL].astype(str) + "_"
        + new_traj.groupby(g[BOX_ID_COL]).cumsum().astype(str)
    )
    counts = g.groupby("traj_id")["traj_id"].transform("size")
    return g[counts >= min_points].reset_index(drop=True)


def add_point_id(gdf):
    """Stable per-point id so events can reference their points after matching."""
    g = gdf.sort_values(["traj_id", "createdAt"]).reset_index(drop=True)
    g["point_id"] = g["traj_id"] + "_p" + g.groupby("traj_id").cumcount().astype(str)
    return g


# ========= ride quality =========


def _dist_variation_cm(values):
    """Median absolute deviation of the nonzero readings, in cm."""
    nz = values[values > 0]
    return float(np.median(np.abs(nz - np.median(nz)))) if len(nz) else np.nan


def per_ride_stats(gdf):
    """One row per ride from raw values + geometry."""
    def stats(ride):
        ride = ride.sort_values("createdAt")   # LineString takes the points in row order
        v = ride["value"].to_numpy(dtype=float)
        in_range = v > 0
        length_km = LineString(ride.geometry.values).length / 1000 if len(ride) >= 2 else 0.0
        dur_h = (ride["createdAt"].max() - ride["createdAt"].min()).total_seconds() / 3600
        return pd.Series({
            "boxId": ride[BOX_ID_COL].iloc[0],
            "boxName": ride["boxName"].iloc[0],
            "n_points": len(v),
            "n_nonzero": int(in_range.sum()),
            "in_range_share": float(in_range.mean()),
            "dist_variation_cm": _dist_variation_cm(v),
            "man_missing_share": ride["man_p"].isna().mean(),
            "n_ot_points": int((ride["man_p"] >= MAN_P_TAU).sum()),
            "length_km": length_km,
            "mean_speed_kmh": length_km / dur_h if dur_h > 0 else np.nan,
        })

    rides = gdf.groupby("traj_id", sort=False).apply(stats, include_groups=False).reset_index()
    usable = rides["length_km"].where(rides["length_km"] >= MIN_LENGTH_KM)  # no rate off a stub
    rides["ot_per_km"] = rides["n_ot_points"] / usable
    return rides


def apply_quality_rules(stats, speed=(SPEED_MIN_KMH, SPEED_MAX_KMH),
                        min_nonzero=STUCK_MIN_NONZERO,
                        var=STUCK_VARIATION_CM,
                        in_range=STUCK_IN_RANGE_MIN,
                        miss=MAN_MISSING_MAX,
                        min_km=MIN_LENGTH_KM):
    """One row per ride, its four flags and the keep verdict. A NaN stat fails every comparison, so it never flags."""
    q = stats.copy()
    q["f_speed"] = (q["mean_speed_kmh"] < speed[0]) | (q["mean_speed_kmh"] > speed[1])
    q["f_stuck"] = ((q["n_nonzero"] >= min_nonzero)
                    & (q["dist_variation_cm"] < var)
                    & (q["in_range_share"] >= in_range))
    q["f_no_manoeuvre"] = q["man_missing_share"] > miss
    q["f_too_short"] = q["length_km"] < min_km

    q["n_flags"] = q[FLAG_COLS].sum(axis=1)
    q["flag_reasons"] = q[FLAG_COLS].apply(
        lambda r: ",".join(c[2:] for c in FLAG_COLS if r[c]) or "ok", axis=1)
    q["keep"] = q["n_flags"] == 0

    front = ["traj_id", "boxId", "boxName", "keep", "n_flags", "flag_reasons",
             "n_points", "length_km", "mean_speed_kmh",
             "dist_variation_cm", "in_range_share", "man_missing_share",
             "n_ot_points", "ot_per_km"]
    rest = [c for c in q.columns if c not in front]
    return q[front + rest]


def rule_sensitivity(stats, verdicts, path=SUMMARY_CSV):
    """Headline counts, plus what each threshold is worth: rides kept when it is loosened
    or tightened, and rides it alone decides."""
    def judged(**thresholds):
        return apply_quality_rules(stats, **thresholds)

    kept = verdicts[verdicts["keep"]]
    rows = [
        ("rides total", len(verdicts)),
        ("rides kept", len(kept)),
        ("rides kept share", f"{verdicts['keep'].mean():.0%}"),
        ("boxes total", verdicts["boxId"].nunique()),
        ("boxes with >=1 kept ride", kept["boxId"].nunique()),
        ("km total", round(verdicts["length_km"].sum(), 1)),
        ("km kept", round(kept["length_km"].sum(), 1)),
        ("dropped: too short/slow ride", int(verdicts["f_too_short"].sum())),
        ("dropped: implausible speed", int(verdicts["f_speed"].sum())),
        ("dropped: blocked/stuck distance sensor", int(verdicts["f_stuck"].sum())),
        ("dropped: no manoeuvre stream (no events measurable)",
         int(verdicts["f_no_manoeuvre"].sum())),
    ]
    for r in RULES:
        loose, strict = (judged(**{r.kwarg: p}) for p in (r.loose, r.strict))
        rows += [
            (f"sensitivity: {r.kwarg} at {r.loose}", int(loose["keep"].sum())),
            (f"sensitivity: {r.kwarg} at {r.strict}", int(strict["keep"].sum())),
            (f"borderline: rides {r.kwarg} alone decides",
             int((loose[r.flag] != strict[r.flag]).sum())),
        ]

    rows += [("sensitivity: all loose",
              int(judged(**{r.kwarg: r.loose for r in RULES})["keep"].sum())),
             ("sensitivity: all strict",
              int(judged(**{r.kwarg: r.strict for r in RULES})["keep"].sum()))]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)
    print(f"[csv] saved -> {path}")


# ========= plotting =========


def _save(fig, path, announce=True):
    """Write and close a figure; the per-box pages are announced once, in main()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if announce:
        print(f"[fig] saved -> {path}")


def _scatter_ride(ax, ride, size_scale=1.0):
    """A ride's GPS path in grey, with a dot per second the sensor saw something."""
    ride = ride.sort_values("createdAt")
    ax.plot(ride.geometry.x, ride.geometry.y, color="lightgrey", linewidth=0.5, zorder=0)
    det = ride[ride["value"] > 0]
    v = det["value"].to_numpy(dtype=float)
    ax.scatter(det.geometry.x, det.geometry.y, c=v, cmap="magma",
               s=(3 + 50 * np.sqrt(np.clip(v, 0, VALUE_VMAX) / VALUE_VMAX)) * size_scale,
               vmin=VALUE_VMIN, vmax=VALUE_VMAX, alpha=0.7, edgecolors="none", zorder=1)


def fig_trajectories(gdf, color_by=BOX_ID_COL, max_legend=MAX_LEGEND):
    """Every kept ride as a line, coloured by box, to show where the data comes from."""
    lines = gpd.GeoDataFrame(
        [{color_by: ride[color_by].iloc[0],
          "geometry": LineString(ride.sort_values("createdAt").geometry.values)}
         for _, ride in gdf.groupby("traj_id") if len(ride) >= 2],
        geometry="geometry", crs=gdf.crs)

    cats = lines[color_by].value_counts().index
    cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, cat in enumerate(cats):
        subset = lines[lines[color_by] == cat]
        subset.plot(ax=ax, color=cmap(i % cmap.N), linewidth=0.7, alpha=0.7, zorder=1,
                    label=f"{str(cat)[:12]} ({len(subset)})" if i < max_legend else None)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"senseBox trajectory coverage by {color_by}", fontsize=11)
    extra = len(cats) - min(len(cats), max_legend)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, frameon=False,
              title=color_by + (f"  (+{extra} more)" if extra > 0 else ""))
    _save(fig, OUT_DIR / "task2a_trajectories.png")


def fig_per_box(gdf_box, out_dir=PER_BOX_DIR, ncols=4):
    """One page per box, one panel per ride, for spotting the stuck-sensor rides."""
    bid, name = gdf_box[BOX_ID_COL].iloc[0], gdf_box["boxName"].iloc[0]
    rides = sorted(gdf_box.groupby("traj_id", sort=False), key=lambda kv: -len(kv[1]))
    nrows = -(-len(rides) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (traj_id, ride) in zip(axes.flat, rides):
        _scatter_ride(ax, ride)
        v = ride["value"].to_numpy(dtype=float)
        ax.set_title(f"ride ..{str(traj_id)[-4:]} | {len(ride)} seconds\n"
                     f"{(v > 0).mean():.0%} object in range | varies {_dist_variation_cm(v):.0f} cm",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()
    for ax in axes.flat[len(rides):]:
        ax.set_visible(False)

    bar = fig.colorbar(plt.cm.ScalarMappable(cmap="magma",
                                             norm=plt.Normalize(VALUE_VMIN, VALUE_VMAX)),
                       ax=axes.ravel().tolist(), shrink=0.5, location="right", pad=0.01)
    bar.set_label("distance reading (cm)", fontsize=11)
    bar.ax.tick_params(labelsize=11)
    fig.suptitle(f"{name}  [{bid}] - {len(rides)} rides, one map each",
                 fontsize=12, y=1.005)
    stem = "__".join(re.sub(r"[^\w.-]+", "_", str(s))[:60] for s in (name, bid))
    _save(fig, out_dir / f"{stem}.png", announce=False)


def main():
    pts = add_point_id(segment_trajectories(add_manoeuvre_channel(
        load_distance_channel(bbox_from_graph()))))
    print(f"{pts['traj_id'].nunique()} rides from {pts[BOX_ID_COL].nunique()} boxes, "
          f"{len(pts)} points")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_POINTS_PATH.unlink(missing_ok=True)
    pts.to_file(SEG_POINTS_PATH, driver="GPKG")
    print(f"[gpkg] saved -> {SEG_POINTS_PATH}")

    stats = per_ride_stats(pts)
    verdicts = apply_quality_rules(stats)
    (verdicts.sort_values(["keep", "n_flags"], ascending=[True, False])   # worst first
             .to_csv(QUALITY_CSV, index=False))
    print(f"[csv] saved -> {QUALITY_CSV}")
    print(f"kept {verdicts['keep'].sum()}/{len(verdicts)} rides "
          f"({verdicts['keep'].mean():.0%})")
    rule_sensitivity(stats, verdicts)

    fig_trajectories(pts[pts["traj_id"].isin(verdicts.loc[verdicts["keep"], "traj_id"])])
    for _, box in pts.groupby(BOX_ID_COL, sort=False):
        fig_per_box(box)
    print(f"[fig] saved -> {PER_BOX_DIR}/ ({pts[BOX_ID_COL].nunique()} per-box pages)")
    print("\nDONE")


if __name__ == "__main__":
    main()
