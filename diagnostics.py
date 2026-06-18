"""Trajectory diagnostics: per-trajectory quality table + per-box plots.

Works on RAW distance values only (no events): nonzero share, longest nonzero
run, raw nonzero readings per km. Outputs under output/diagnostics/:
trajectory_quality.csv, box_overview.png, per_box/<name>__<id>.png.
keep == (n_flags == 0); loosen with n_flags <= 1 or edit thresholds.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from senseboxbike_preprocessing import (
    load_points, segment_trajectories, bbox_from_graph,
    BOX_ID_COL, INPUT_CSV,
)

OUT_DIR = Path("output/diagnostics")
PER_BOX_DIR = OUT_DIR / "per_box"
OVERVIEW_PATH = OUT_DIR / "box_overview.png"
QUALITY_CSV = OUT_DIR / "trajectory_quality.csv"

USE_SAVED_GPKG = None  # Path("output/trajectory_points.gpkg") to reuse; None reruns

# shared colour scale; values clip at ~400 (sensor max range) so cap below that
VALUE_VMIN, VALUE_VMAX = 0, 250
CMAP = "viridis"

SPEED_MIN_KMH = 3.0
SPEED_MAX_KMH = 40.0
NZ_SHARE_MAX = 0.40
RUN_FRAC_MAX = 0.30    # longest nonzero run / trajectory length; stuck ~0.67, real wall-follows ~0.04
RAW_PER_KM_MAX = 250   # nonzero readings per km; at 1 Hz / 15 km/h every reading nonzero is ~240/km
MIN_POINTS = 5
MIN_LENGTH_KM = 0.2


def _safe(s):
    return re.sub(r"[^\w.-]+", "_", str(s))[:60]


def longest_nonzero_run(values):
    nz = (np.asarray(values) > 0).astype(int)
    if nz.sum() == 0:
        return 0
    best = run = 0
    for v in nz:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def per_trajectory_stats(gdf):
    """One row per trajectory from raw values + geometry. No events."""
    def agg(t):
        t = t.sort_values("createdAt")
        v = t["value"].to_numpy(dtype=float)
        pts = t.geometry.values
        length_km = LineString(pts).length / 1000 if len(pts) >= 2 else 0.0
        dur_min = (t["createdAt"].max() - t["createdAt"].min()).total_seconds() / 60
        return pd.Series({
            "boxId": t[BOX_ID_COL].iloc[0],
            "boxName": t["boxName"].iloc[0] if "boxName" in t else "",
            "n_points": len(v),
            "n_nonzero": int((v > 0).sum()),
            "nonzero_share": (v > 0).mean(),
            "value_max": v.max(),
            "longest_nonzero_run": longest_nonzero_run(v),
            "length_km": length_km,
            "mean_speed_kmh": length_km / (dur_min / 60) if dur_min > 0 else np.nan,
        })

    s = gdf.groupby("traj_id", sort=False).apply(agg, include_groups=False).reset_index()
    usable = s["length_km"].where(s["length_km"] >= MIN_LENGTH_KM)
    s["raw_per_km"] = s["n_nonzero"] / usable
    return s


def build_quality_table(stats):
    m = stats.copy()
    m["f_speed"] = (m["mean_speed_kmh"] < SPEED_MIN_KMH) | (m["mean_speed_kmh"] > SPEED_MAX_KMH)
    m["f_nz_share"] = m["nonzero_share"] > NZ_SHARE_MAX
    m["f_long_run"] = (m["longest_nonzero_run"] / m["n_points"]) > RUN_FRAC_MAX
    m["f_rate"] = m["raw_per_km"] > RAW_PER_KM_MAX
    m["f_too_short"] = (m["n_points"] < MIN_POINTS) | (m["length_km"] < MIN_LENGTH_KM)

    flag_cols = ["f_speed", "f_nz_share", "f_long_run", "f_rate", "f_too_short"]
    m["n_flags"] = m[flag_cols].sum(axis=1)

    reason_map = {"f_speed": "speed", "f_nz_share": "nz_share", "f_long_run": "long_run",
                  "f_rate": "high_rate", "f_too_short": "too_short"}
    m["flag_reasons"] = m[flag_cols].apply(
        lambda r: ",".join(reason_map[c] for c in flag_cols if r[c]) or "ok", axis=1)
    m["keep"] = m["n_flags"] == 0

    front = ["traj_id", "boxId", "boxName", "keep", "n_flags", "flag_reasons",
             "n_points", "length_km", "mean_speed_kmh", "nonzero_share",
             "longest_nonzero_run", "raw_per_km", "value_max"]
    rest = [c for c in m.columns if c not in front]
    return m[front + rest].sort_values(["keep", "n_flags"], ascending=[True, False])


def good_traj_ids(quality_table):
    return set(quality_table.loc[quality_table["keep"], "traj_id"])


def print_report(q):
    print(f"\n=== trajectory quality: {len(q)} trajectories ===")
    print(f"keep (0 flags): {q['keep'].sum()}  ({q['keep'].mean():.0%})")
    print("\nflag counts (a trajectory can trip several):")
    for c in ["f_speed", "f_nz_share", "f_long_run", "f_rate", "f_too_short"]:
        print(f"  {c:12s}: {int(q[c].sum()):4d}")
    print("\nn_flags distribution:")
    print(q["n_flags"].value_counts().sort_index().to_string())

    print("\n=== per-box keep rate (worst first) ===")
    by_box = (q.groupby(["boxId", "boxName"])
                .agg(n_traj=("traj_id", "size"), keep_rate=("keep", "mean"))
                .reset_index().sort_values("keep_rate"))
    by_box["keep_rate"] = by_box["keep_rate"].round(2)
    print(by_box.head(15).to_string(index=False))


def box_label(t):
    name = t["boxName"].iloc[0] if "boxName" in t else ""
    return f"{name} [{str(t[BOX_ID_COL].iloc[0])[-6:]}]"


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
        flag = "X" if nz_share > NZ_SHARE_MAX else ""
        ax.set_title(f"{flag}{box_label(b)}\n{b['traj_id'].nunique()} traj | {len(b)} pts\n"
                     f"nz {nz_share:.0%} | max {v.max():.0f}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.4, location="right",
                     pad=0.01).set_label("clearance value (capped at scale max)")
    fig.suptitle(f"Per-box overview\nX = mean nonzero-share > {NZ_SHARE_MAX:.0%} "
                 "(likely stuck sensor)", fontsize=13, y=1.005)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved -> {path}")


def plot_one_box(gdf_box, out_dir=PER_BOX_DIR):
    bid = gdf_box[BOX_ID_COL].iloc[0]
    name = gdf_box["boxName"].iloc[0] if "boxName" in gdf_box else ""
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
        ax.set_title(f"{str(tid)[-4:]} | {len(t)} pts\nnz {(v > 0).mean():.0%} | "
                     f"run {longest_nonzero_run(v)} | max {v.max():.0f}", fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()

    for ax in axes[len(trajs):]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.5, location="right", pad=0.01)
    fig.suptitle(f"{name}  [{bid}]  —  {len(trajs)} trajectories", fontsize=13, y=1.005)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{_safe(name)}__{_safe(bid)}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    if USE_SAVED_GPKG and Path(USE_SAVED_GPKG).exists():
        pts = gpd.read_file(USE_SAVED_GPKG)
        pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    else:
        pts = segment_trajectories(load_points(INPUT_CSV, bbox=bbox_from_graph()))

    print(f"[diag] {pts['traj_id'].nunique()} trajectories from "
          f"{pts[BOX_ID_COL].nunique()} boxes, {len(pts)} points")

    stats = per_trajectory_stats(pts)
    q = build_quality_table(stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q.to_csv(QUALITY_CSV, index=False)
    print_report(q)
    print(f"\nsaved -> {QUALITY_CSV}  ({q['keep'].sum()} good trajectories)")

    plot_box_overview(pts)
    PER_BOX_DIR.mkdir(parents=True, exist_ok=True)
    for bid, b in pts.groupby(BOX_ID_COL, sort=False):
        p = plot_one_box(b)
        print(f"  [box] {box_label(b):32s} {b['traj_id'].nunique():3d} traj -> {p.name}")
    print(f"[diag] per-box pages in {PER_BOX_DIR}/")