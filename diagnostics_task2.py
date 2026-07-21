"""Data intake + ride quality check: build clean rides once, decide which may proceed.

Intake: load the distance channel, drop mislabeled/duplicate rows, join the
"Overtaking Manoeuvre" classifier channel by (box, timestamp), clip to the
network area, cut into rides on >10 min gaps, give every point a stable id.

Quality: flag a ride as bad when
  f_stuck         distance readings barely vary over the whole ride -> sensor was blocked
  f_no_manoeuvre  the classifier channel is missing -> overtakes cannot be counted
  f_speed         average speed is not cycling (below 3 or above 40 km/h)
  f_too_short     ride too short to use (under 5 points or 0.2 km)
A ride with zero flags is kept.

Writes to output/task2_diagnostics/:
  segmented_points_task2.gpkg    ALL rides' points (senseboxbike_preprocessing_task2.py
                                 reads this -- no recomputation downstream)
  trajectory_quality_task2.csv   one row per ride: stats, flags, keep verdict
  diagnostics_summary_task2.csv  headline numbers + threshold sensitivity, for slides
  box_overview + per-box PNGs    maps for eyeballing each box's rides
"""
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

CRS_WGS84 = "EPSG:4326"
CRS_METRIC = "EPSG:25832"            # UTM 32N, same as the network graph
BOX_ID_COL = "boxId"

GRAPH_PATH = Path("input/muenster_bike.graphml")
INPUT_CSV = Path("input/muenster_overtaking_distance_2024-07_2026-07.csv")
MANOEUVRE_CSV = Path("input/muenster_overtaking_manoeuvre_2024-07_2026-07.csv")

OUT_DIR = Path("output/task2_diagnostics")
SEG_POINTS_PATH = OUT_DIR / "segmented_points_task2.gpkg"
QUALITY_CSV = OUT_DIR / "trajectory_quality_task2.csv"
SUMMARY_CSV = OUT_DIR / "diagnostics_summary_task2.csv"
OVERVIEW_PATH = OUT_DIR / "box_overview_task2.png"
PER_BOX_DIR = OUT_DIR / "per_box_task2"

BBOX_PAD_M = 2000
GAP_MINUTES = 10           # silence longer than this starts a new ride
MIN_POINTS = 5
MAN_P_TAU = 0.5            # classifier probability gate; sensitivity: 0.2 / 0.8

SPEED_MIN_KMH = 3.0
SPEED_MAX_KMH = 40.0
STUCK_VARIATION_CM = 3     # readings varying less than the sensor's own noise = blocked sensor
STUCK_MIN_NONZERO = 20     # only judge variation when there are enough nonzero readings
MAN_MISSING_MAX = 0.5      # >50% of points without a manoeuvre reading -> events not measurable
MIN_LENGTH_KM = 0.2
FLAG_COLS = ["f_speed", "f_stuck", "f_no_manoeuvre", "f_too_short"]

VALUE_VMIN, VALUE_VMAX = 0, 250
CMAP = "viridis"


# ---------------------------------------------------------------- intake

def bbox_from_graph(graph_path=GRAPH_PATH, pad_m=BBOX_PAD_M):
    G = ox.load_graphml(graph_path)
    nodes, _ = ox.graph_to_gdfs(G)
    minx, miny, maxx, maxy = nodes.total_bounds
    return (minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)


def load_points(csv_path=INPUT_CSV, bbox=None):
    df = pd.read_csv(csv_path)
    if "unit" in df.columns:         # a few boxes report a mislabeled '%' phenomenon here
        n = len(df)
        df = df[df["unit"] == "cm"]
        if n - len(df):
            print(f"[load] dropped {n - len(df)} non-cm rows")
    df = df.drop_duplicates()        # the bulk-download endpoint returns duplicated rows
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


def add_manoeuvre(gdf, manoeuvre_csv=MANOEUVRE_CSV):
    """Join the Overtaking Manoeuvre channel (0-1 car-pass probability, max-pooled
    over 2 s by the app) onto the points by (boxId, createdAt); missing -> 0."""
    m = pd.read_csv(manoeuvre_csv)
    m["createdAt"] = pd.to_datetime(m["createdAt"], utc=True)
    m = (m.rename(columns={"value": "man_p"})[[BOX_ID_COL, "createdAt", "man_p"]]
         .drop_duplicates(subset=[BOX_ID_COL, "createdAt"]))
    out = gdf.merge(m, on=[BOX_ID_COL, "createdAt"], how="left")
    out["man_p"] = pd.to_numeric(out["man_p"], errors="coerce")
    out["man_p_missing"] = out["man_p"].isna()   # no manoeuvre reading at this timestamp
    out["man_p"] = out["man_p"].fillna(0.0)
    print(f"[man] {(out['man_p'] > 0).mean():.0%} of points have a nonzero manoeuvre probability "
          f"({out['man_p_missing'].mean():.0%} missing a reading)")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def segment_trajectories(gdf, gap_minutes=GAP_MINUTES, min_points=MIN_POINTS):
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


# ---------------------------------------------------------------- quality

def dist_variation_cm(values):
    """How many cm the nonzero readings typically vary around their usual value
    (median absolute deviation). Healthy riding: tens of cm (cars, hedges, gaps).
    Below the sensor's own noise (~3 cm): it stared at one fixed object."""
    v = np.asarray(values, dtype=float)
    nz = v[v > 0]
    if len(nz) == 0:
        return np.nan
    return float(np.median(np.abs(nz - np.median(nz))))


def per_trajectory_stats(gdf):
    """One row per ride from raw values + geometry. No events."""
    def agg(t):
        t = t.sort_values("createdAt")
        v = t["value"].to_numpy(dtype=float)
        pts = t.geometry.values
        length_km = LineString(pts).length / 1000 if len(pts) >= 2 else 0.0
        dur_min = (t["createdAt"].max() - t["createdAt"].min()).total_seconds() / 60
        return pd.Series({
            "boxId": t[BOX_ID_COL].iloc[0],
            "boxName": t["boxName"].iloc[0],
            "n_points": len(v),
            "n_nonzero": int((v > 0).sum()),
            "dist_variation_cm": dist_variation_cm(v),
            "man_missing_share": t["man_p_missing"].mean(),
            "n_ot_points": int((t["man_p"] >= MAN_P_TAU).sum()),
            "length_km": length_km,
            "mean_speed_kmh": length_km / (dur_min / 60) if dur_min > 0 else np.nan,
        })

    s = gdf.groupby("traj_id", sort=False).apply(agg, include_groups=False).reset_index()
    usable = s["length_km"].where(s["length_km"] >= MIN_LENGTH_KM)
    s["ot_per_km"] = s["n_ot_points"] / usable
    return s


def build_quality_table(stats):
    m = stats.copy()
    m["f_speed"] = (m["mean_speed_kmh"] < SPEED_MIN_KMH) | (m["mean_speed_kmh"] > SPEED_MAX_KMH)
    m["f_stuck"] = ((m["n_nonzero"] >= STUCK_MIN_NONZERO)
                    & (m["dist_variation_cm"] < STUCK_VARIATION_CM))
    m["f_no_manoeuvre"] = m["man_missing_share"] > MAN_MISSING_MAX
    m["f_too_short"] = (m["n_points"] < MIN_POINTS) | (m["length_km"] < MIN_LENGTH_KM)

    m["n_flags"] = m[FLAG_COLS].sum(axis=1)
    m["flag_reasons"] = m[FLAG_COLS].apply(
        lambda r: ",".join(c[2:] for c in FLAG_COLS if r[c]) or "ok", axis=1)
    m["keep"] = m["n_flags"] == 0

    front = ["traj_id", "boxId", "boxName", "keep", "n_flags", "flag_reasons",
             "n_points", "length_km", "mean_speed_kmh",
             "dist_variation_cm", "man_missing_share", "n_ot_points", "ot_per_km"]
    rest = [c for c in m.columns if c not in front]
    return m[front + rest].sort_values(["keep", "n_flags"], ascending=[True, False])


def diagnostic_summary(q, path=SUMMARY_CSV):
    """One CSV for slides: headline numbers, per-flag drop counts, and the
    threshold-sensitivity check (kept counts under loosened/tightened rules)."""
    def kept(speed=(SPEED_MIN_KMH, SPEED_MAX_KMH), var=STUCK_VARIATION_CM,
             miss=MAN_MISSING_MAX, min_km=MIN_LENGTH_KM):
        bad = ((q["mean_speed_kmh"] < speed[0]) | (q["mean_speed_kmh"] > speed[1])
               | ((q["n_nonzero"] >= STUCK_MIN_NONZERO) & (q["dist_variation_cm"] < var))
               | (q["man_missing_share"] > miss)
               | (q["n_points"] < MIN_POINTS) | (q["length_km"] < min_km))
        return int((~bad).sum())

    k = q[q["keep"]]
    rows = [
        ("rides total", len(q)),
        ("rides kept", len(k)),
        ("rides kept share", f"{q['keep'].mean():.0%}"),
        ("boxes total", q["boxId"].nunique()),
        ("boxes with >=1 kept ride", k["boxId"].nunique()),
        ("km total", round(q["length_km"].sum(), 1)),
        ("km kept", round(k["length_km"].sum(), 1)),
        ("dropped: too short/slow ride", int(q["f_too_short"].sum())),
        ("dropped: implausible speed", int(q["f_speed"].sum())),
        ("dropped: blocked/stuck distance sensor", int(q["f_stuck"].sum())),
        ("dropped: no manoeuvre stream (no events measurable)", int(q["f_no_manoeuvre"].sum())),
        ("sensitivity: speed bounds 2-50 km/h", kept(speed=(2, 50))),
        ("sensitivity: speed bounds 5-30 km/h", kept(speed=(5, 30))),
        ("sensitivity: stuck variation < 2 cm", kept(var=2)),
        ("sensitivity: stuck variation < 5 cm", kept(var=5)),
        ("sensitivity: classifier missing > 30%", kept(miss=0.3)),
        ("sensitivity: classifier missing > 70%", kept(miss=0.7)),
        ("sensitivity: min ride length 0.1 km", kept(min_km=0.1)),
        ("sensitivity: min ride length 0.5 km", kept(min_km=0.5)),
        ("sensitivity: all loose", kept(speed=(2, 50), var=2, miss=0.7, min_km=0.1)),
        ("sensitivity: all strict", kept(speed=(5, 30), var=5, miss=0.3, min_km=0.5)),
    ]
    s = pd.DataFrame(rows, columns=["metric", "value"])
    s.to_csv(path, index=False)
    return s


# ---------------------------------------------------------------- plots

def _safe(s):
    return re.sub(r"[^\w.-]+", "_", str(s))[:60]


def box_label(t):
    return f"{t['boxName'].iloc[0]} [{str(t[BOX_ID_COL].iloc[0])[-6:]}]"


def _scatter_traj(ax, t, size_scale=1.0):
    t = t.sort_values("createdAt")
    ax.plot(t.geometry.x, t.geometry.y, color="0.85", linewidth=0.5, zorder=0)
    det = t[t["value"] > 0]
    if det.empty:
        return None
    v = det["value"].to_numpy(dtype=float)
    s = (3 + 50 * np.sqrt(np.clip(v, 0, VALUE_VMAX) / VALUE_VMAX)) * size_scale
    return ax.scatter(det.geometry.x, det.geometry.y, s=s, c=v, cmap=CMAP,
                      vmin=VALUE_VMIN, vmax=VALUE_VMAX, alpha=0.7,
                      edgecolors="none", zorder=1)


def plot_box_overview(gdf, path=OVERVIEW_PATH):
    boxes = sorted(gdf.groupby(BOX_ID_COL, sort=False), key=lambda kv: -len(kv[1]))
    n, ncols = len(boxes), 6
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    sc = None

    for ax, (bid, b) in zip(axes, boxes):
        v = b["value"].to_numpy(dtype=float)
        nz_share = (v > 0).mean()
        for _, t in b.groupby("traj_id", sort=False):
            _sc = _scatter_traj(ax, t, size_scale=0.5)
            if _sc is not None:
                sc = _sc
        n_ot = int((b["man_p"] >= MAN_P_TAU).sum())
        ax.set_title(f"{box_label(b)}\n{b['traj_id'].nunique()} rides | {len(b)} seconds\n"
                     f"{nz_share:.0%} object in range | {n_ot} overtake-seconds", fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.4, location="right",
                     pad=0.01).set_label("distance reading (cm, capped at scale max)")
    fig.suptitle("Per-box overview -- one map per box, all its rides\n"
                 "grey line = GPS path; dots = seconds with an object in sensor range "
                 "(bigger/brighter = farther); overtake-seconds = classifier confident a car passed",
                 fontsize=13, y=1.005)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")


def plot_one_box(gdf_box, out_dir=PER_BOX_DIR):
    bid = gdf_box[BOX_ID_COL].iloc[0]
    name = gdf_box["boxName"].iloc[0]
    trajs = sorted(gdf_box.groupby("traj_id", sort=False), key=lambda kv: -len(kv[1]))
    ncols = 4
    nrows = -(-len(trajs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    sc = None

    for ax, (tid, t) in zip(axes, trajs):
        _sc = _scatter_traj(ax, t)
        if _sc is not None:
            sc = _sc
        v = t["value"].to_numpy(dtype=float)
        ax.set_title(f"ride ..{str(tid)[-4:]} | {len(t)} seconds\n"
                     f"{(v > 0).mean():.0%} object in range | varies {dist_variation_cm(v):.0f} cm",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()

    for ax in axes[len(trajs):]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.5, location="right", pad=0.01)
    fig.suptitle(f"{name}  [{bid}]  --  {len(trajs)} rides, one map each\n"
                 "grey line = GPS path; dots = seconds with an object in sensor range; "
                 "'varies X cm' = typical variation of the readings; near 0 = sensor stared "
                 "at one object (stuck filter)",
                 fontsize=12, y=1.005)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{_safe(name)}__{_safe(bid)}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    pts = add_point_id(segment_trajectories(add_manoeuvre(
        load_points(INPUT_CSV, bbox=bbox_from_graph()))))
    print(f"[diag] {pts['traj_id'].nunique()} trajectories from "
          f"{pts[BOX_ID_COL].nunique()} boxes, {len(pts)} points")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_POINTS_PATH.unlink(missing_ok=True)  # avoid stale extra layers in the gpkg
    pts.to_file(SEG_POINTS_PATH, driver="GPKG")
    print(f"[diag] segmented points -> {SEG_POINTS_PATH}")

    q = build_quality_table(per_trajectory_stats(pts))
    q.to_csv(QUALITY_CSV, index=False)
    diagnostic_summary(q)
    print(f"[diag] kept {q['keep'].sum()}/{len(q)} rides ({q['keep'].mean():.0%}) -> {QUALITY_CSV}")
    print(f"[diag] summary -> {SUMMARY_CSV}")

    plot_box_overview(pts)
    for _, b in pts.groupby(BOX_ID_COL, sort=False):
        plot_one_box(b)
    print(f"[plot] per-box pages -> {PER_BOX_DIR}/")
