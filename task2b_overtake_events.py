"""Extract overtake events from the rides that passed quality control.

A burst of high manoeuvre probability becomes one event per car pass.
Each event carries first, last and anchor point ids; the anchor is the highest-probability point.

Writes to output/task2_overtakes/:
  task2b_overtake_events.gpkg      one row per overtake
  task2b_trajectory_summary.csv    one row per ride
  task2b_event_sensitivity.csv     what the event count rests on
  task2b_overtakes_per_box.png     overtakes against rider hours, one point per box
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from task2a_ride_quality import QUALITY_CSV, SEG_POINTS_PATH

OUT_DIR = Path("output/task2_overtakes")
EVENTS_PATH = OUT_DIR / "task2b_overtake_events.gpkg"
SUMMARY_PATH = OUT_DIR / "task2b_trajectory_summary.csv"
EVENT_SENSITIVITY_CSV = OUT_DIR / "task2b_event_sensitivity.csv"

CLOSE_PASS_CM = 150
MAN_PROB_THRESHOLD = 0.5
MERGE_GAP_S = 4

# probes for sensitivity analysis
MERGE_GAP_PROBES = [1, 3, 8, 15]
MAN_PROB_PROBES = [0.2, 0.8]

EVENT_COLS = ["event_uid", "traj_id", "event_id", "boxId", "start", "end",
              "duration_s", "n_seconds", "max_man_p", "min_clearance_cm",
              "mean_clearance_cm", "is_close", "first_point_id", "last_point_id",
              "anchor_point_id", "geometry"]


def load_kept_rides(points_path=SEG_POINTS_PATH, quality_csv=QUALITY_CSV):
    """Load the rides task 2a passed."""
    quality = pd.read_csv(quality_csv, parse_dates=["start", "end"])
    rides = quality[quality["keep"]].set_index("traj_id")
    points = gpd.read_file(points_path)
    points["createdAt"] = pd.to_datetime(points["createdAt"], utc=True)
    points = points[points["keep"]].reset_index(drop=True)
    print(f"[load] {len(rides)}/{len(quality)} rides kept ({len(points)} points) "
          f"from {points_path.name}")
    return points, rides


def extract_overtake_events(points, man_p_tau=MAN_PROB_THRESHOLD, merge_gap_s=MERGE_GAP_S):
    """Collapse each burst of high manoeuvre probability into one row per overtake."""
    gated = points[points["man_p"] >= man_p_tau].sort_values(["traj_id", "createdAt"])
    if gated.empty:
        return gpd.GeoDataFrame(columns=EVENT_COLS, geometry="geometry", crs=points.crs)

    gap_s = gated.groupby("traj_id")["createdAt"].diff().dt.total_seconds()
    gated["event_id"] = (gap_s.isna() | (gap_s > merge_gap_s)).groupby(gated["traj_id"]).cumsum()

    def one_event(burst):
        anchor = burst.loc[burst["man_p"].idxmax()]
        clearances = burst.loc[burst["value"] > 0, "value"]
        closest = clearances.min()  # NaN when nothing was ever in range
        start, end = burst["createdAt"].min(), burst["createdAt"].max()
        return pd.Series({
            "boxId": burst["boxId"].iloc[0],
            "start": start,
            "end": end,
            "duration_s": (end - start).total_seconds(),
            "n_seconds": len(burst),
            "max_man_p": burst["man_p"].max(),
            "min_clearance_cm": closest,
            "mean_clearance_cm": clearances.mean(),
            "is_close": bool(closest < CLOSE_PASS_CM),
            "first_point_id": burst["point_id"].iloc[0],
            "last_point_id": burst["point_id"].iloc[-1],
            "anchor_point_id": anchor["point_id"],
            "geometry": anchor.geometry,
        })

    events = (gated.groupby(["traj_id", "event_id"], group_keys=True)
              .apply(one_event, include_groups=False).reset_index())
    events["event_uid"] = events["traj_id"] + "_ev" + events["event_id"].astype(str)
    return gpd.GeoDataFrame(events[EVENT_COLS], geometry="geometry", crs=points.crs)


def summarise(rides, events):
    """Each kept ride as task 2a measured it, plus the overtake rates and descriptives found on it."""
    summary = rides.copy()
    if len(events):
        by_event = events.groupby("traj_id")
        summary = summary.join(pd.DataFrame({
            "n_overtakes": by_event.size(),
            "n_close_passes": by_event["is_close"].sum(),
            "min_overtake_cm": by_event["min_clearance_cm"].min(),
            "mean_overtake_cm": by_event["min_clearance_cm"].mean(),
            "mean_event_duration_s": by_event["duration_s"].mean(),
        }))
    for c in ("n_overtakes", "n_close_passes"):
        summary[c] = summary.get(c, pd.Series(0, index=summary.index)).fillna(0).astype(int)

    hours = summary["duration_min"] / 60          # task 2a already dropped the stubs
    summary["overtake_rate_per_km"] = summary["n_overtakes"] / summary["length_km"]
    summary["overtake_rate_per_h"] = summary["n_overtakes"] / hours.where(hours > 0)
    return summary.sort_values("overtake_rate_per_h", ascending=False)


def event_sensitivity(points, path=EVENT_SENSITIVITY_CSV):
    """How many overtakes the two event-defining constants are worth."""
    def n_events(**thresholds):
        return len(extract_overtake_events(points, **thresholds))

    rows = [("gated seconds", int((points["man_p"] >= MAN_PROB_THRESHOLD).sum())),
            (f"events at merge gap {MERGE_GAP_S} s, gate {MAN_PROB_THRESHOLD}", n_events())]
    rows += [(f"events at merge gap {gap} s", n_events(merge_gap_s=gap))
             for gap in MERGE_GAP_PROBES]
    rows += [(f"events at gate {tau}", n_events(man_p_tau=tau))
             for tau in MAN_PROB_PROBES]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(path, index=False)
    print(f"[csv] saved -> {path}")


# ========= plotting =========


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {path}")


def fig_overtakes_per_box(summary):
    """Overtakes against rider-hours, one point per box.
    Spread around the dashed pooled rate is the between-rider variation."""
    hours = summary["duration_min"] / 60
    per_box = (summary.assign(hours=hours)
               .groupby("boxId").agg(hours=("hours", "sum"),
                                     overtakes=("n_overtakes", "sum")))
    pooled = summary["n_overtakes"].sum() / hours.sum()
    span = [0, per_box["hours"].max()]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(span, [pooled * x for x in span], color="dimgrey", ls="--", lw=1.2, zorder=1)
    ax.scatter(per_box["hours"], per_box["overtakes"], s=24, color="crimson",
               alpha=0.6, edgecolors="none", zorder=2)
    ax.annotate(f"pooled rate {pooled:.1f}/h", xy=(span[1], pooled * span[1]),
                xytext=(-8, 10), textcoords="offset points", ha="right",
                color="dimgrey", fontsize=11)
    ax.set_xlabel("rider hours", fontsize=11)
    ax.set_ylabel("overtakes recorded", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Overtakes against exposure ({len(per_box)} boxes)",
                 loc="left", fontsize=11)
    _save(fig, OUT_DIR / "task2b_overtakes_per_box.png")


def main():
    points, rides = load_kept_rides()
    events = extract_overtake_events(points)
    summary = summarise(rides, events)
    km = summary["length_km"].sum()
    print(f"{len(events)} overtake events "
          f"({int(events['is_close'].sum())} closer than {CLOSE_PASS_CM} cm) "
          f"over {km:.0f} km -> {len(events) / km:.2f} events/km")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.unlink(missing_ok=True)
    events.to_file(EVENTS_PATH, driver="GPKG")
    print(f"[gpkg] saved -> {EVENTS_PATH}")

    summary.to_csv(SUMMARY_PATH)
    print(f"[csv] saved -> {SUMMARY_PATH}")

    event_sensitivity(points)

    fig_overtakes_per_box(summary)
    print("\nDONE")


if __name__ == "__main__":
    main()
