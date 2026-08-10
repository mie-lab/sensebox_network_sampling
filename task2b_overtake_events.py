"""Extract overtake events from the quality-checked rides.

Reads the segmented points and the keep verdict written by task2a_ride_quality.py
(run that first), keeps only approved rides, and collapses bursts of high
"Overtaking Manoeuvre" probability (man_p >= MAN_P_TAU, gaps <= MERGE_GAP_S
merged) into one event per car pass.

Each event carries first/last/anchor point ids so task3_mapmatching.py can pin it onto a matched edge;
the anchor is the highest-probability point (best guess for the moment the car was alongside).

Writes to output/task2_trajectories/:
  task2b_trajectory_points.gpkg   points of the kept rides (input to map-matching)
  task2b_overtake_events.gpkg     one row per overtake event
  task2b_trajectory_summary.csv   one row per ride
and the two slide figures for this stage to output/figures/.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString

from task2a_ride_quality import BOX_ID_COL, MAN_P_TAU, QUALITY_CSV, SEG_POINTS_PATH

OUT_DIR = Path("output/task2_trajectories")
FIG_DIR = Path("output/figures")
POINTS_PATH = OUT_DIR / "task2b_trajectory_points.gpkg"
EVENTS_PATH = OUT_DIR / "task2b_overtake_events.gpkg"
SUMMARY_PATH = OUT_DIR / "task2b_trajectory_summary.csv"
PLOT_PATH = OUT_DIR / "task2b_trajectories.png"

CLOSE_PASS_CM = 150                  # ordinal descriptor threshold; NOT meters-true clearance
MERGE_GAP_S = 5                      # gap below which two bursts are one event
                                     # (the app max-pools man_p over 2 s, which can split one pass)
MIN_LENGTH_KM = 0.2
MAX_LEGEND = 15

INK, MUTED = "#0b0b0b", "#52514e"
RIDE, EVENT = "#2a78d6", "#e34948"
plt.rcParams.update({"font.size": 12, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


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


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = FIG_DIR / name
    fig.savefig(p, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {p}")


def fig_box_activity(summary):
    """One row per box: a faint lifespan bar (first→last ride) with a tick per ride.
    Sorted by first activity, so boxes appearing over time form a staircase."""
    s = summary.copy()
    s["start_num"] = mdates.date2num(s["start"])
    first = s.groupby("boxName")["start_num"].min()
    last = s.groupby("boxName")["start_num"].max()
    order = first.sort_values(ascending=False).index
    ypos = {name: i for i, name in enumerate(order)}

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("white")
    for name in order:
        y = ypos[name]
        ax.plot([first[name], last[name]], [y, y], color="0.86", lw=3,
                solid_capstyle="round", zorder=1)
    ax.scatter(s["start_num"], [ypos[n] for n in s["boxName"]],
               s=12, color=RIDE, alpha=0.8, marker="|", linewidths=1.1, zorder=2)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n[:22] for n in order], fontsize=7)
    ax.set_ylim(-1, len(order))
    ax.set_xlim(s["start_num"].min() - 15, s["start_num"].max() + 15)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", which="minor", length=0)
    ax.tick_params(axis="x", which="major", labelsize=10, colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"When each senseBox was active  ({s['boxName'].nunique()} boxes, "
                 f"{len(s)} rides)\nCollection is episodic — most boxes ride briefly, "
                 "a few carry the dataset",
                 loc="left", fontsize=14, pad=12)
    _save(fig, "task2b_box_activity.png")


def fig_overtake_extraction(pts, traj_id="67226da749d0900007ca343c_40",
                            win_start_s=60, win_end_s=210):
    """One ride's classifier confidence over time; gated bursts become events —
    the picture of what extract_overtake_events() does."""
    t = pts[pts["traj_id"] == traj_id].sort_values("createdAt").copy()
    if t.empty:
        print(f"[fig] skipped task2b_overtake_extraction: traj_id {traj_id} not in the kept rides")
        return
    t0 = t["createdAt"].min()
    t["s"] = (t["createdAt"] - t0).dt.total_seconds()
    w = t[(t["s"] >= win_start_s) & (t["s"] <= win_end_s)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    # shade gated bursts (man_p >= tau), merged if <= MERGE_GAP_S apart
    gated = w[w["man_p"] >= MAN_P_TAU]
    if len(gated):
        brk = gated["s"].diff().gt(MERGE_GAP_S).cumsum()
        for _, g in gated.groupby(brk):
            for ax in (ax1, ax2):
                ax.axvspan(g["s"].min() - 0.5, g["s"].max() + 0.5,
                           color=EVENT, alpha=0.13, zorder=0)

    ax1.plot(w["s"], w["man_p"], color=RIDE, lw=1.6)
    ax1.axhline(MAN_P_TAU, color=MUTED, ls="--", lw=1)
    ax1.text(w["s"].min(), MAN_P_TAU + 0.03, f"gate  τ = {MAN_P_TAU}",
             color=MUTED, fontsize=10)
    ax1.set_ylabel("classifier\nconfidence")
    ax1.set_ylim(-0.03, 1.03)

    ax2.plot(w["s"], w["value"].where(w["value"] > 0), color="0.55", lw=1.2)
    ax2.set_ylabel("distance\n(cm)")
    ax2.set_xlabel("seconds into ride")

    n_ev = 0 if not len(gated) else (gated["s"].diff().gt(MERGE_GAP_S).sum() + 1)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    ax1.set_title("From sensor stream to overtake events\n"
                  f"red bands = seconds the classifier flags a passing car → "
                  f"{int(n_ev)} events in this 2.5-min window",
                  loc="left", fontsize=14, pad=10)
    _save(fig, "task2b_overtake_extraction.png")


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


def main():
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
    try:                                        # plots must never block the data outputs
        plot_trajectories(pts)
        fig_box_activity(summ)
        fig_overtake_extraction(pts)
    except Exception as e:
        print(f"[plot] FAILED ({e}) -> data outputs are saved, only the figures are missing")


if __name__ == "__main__":
    main()
