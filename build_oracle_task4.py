"""Build the per-edge oracle: the table the risk model screens.

Unit = a DIRECTED edge (u -> v). Both orientations of every street are present,
because a cyclist can ride either way (contraflow on one-ways is common, and ~4%
of matched points travel against the digitised direction). The undirected key
(u_lo, v_hi) is carried so the model can pool directions where data is thin.

Every edge in the network is present, including those never ridden
(n_traversals = 0) — screening covers the whole network, so "never observed" has
to be a row, not a missing one.

Exposure: a traversal is not a fixed amount of exposure. Median time on an edge
is ~5 s but some traversals last many minutes (a rider stopped while traffic
passes), so each traversal carries
  distance = the edge's OSM length (assumes the whole edge was ridden)
  time     = sum of per-point dwell (each point's gap to the next GPS point in the
             ride), so even a single-point traversal gets its real enter->leave
             seconds instead of 0.
Rates are offered per traversal / rider-km / rider-hour; which denominator the
estimand uses is a modelling decision, not one made here.

Outputs (output/task4_oracle/):
  edge_traversals  one row per (directed edge, ride)
  edge_events      one row per overtake event
  edge_oracle      one row per directed edge: exposure, counts, covariates
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

OUT_DIR = Path("output/task4_oracle")
MATCHED_POINTS = Path("output/task3_matching/matched_points_task3.gpkg")
MATCHED_EVENTS = Path("output/task3_matching/matched_events_task3.gpkg")
COVARIATES_CSV = Path("input/muenster_edge_covariates.csv")

TRAVERSALS_CSV = OUT_DIR / "edge_traversals_task4.csv"
EVENTS_CSV = OUT_DIR / "edge_events_task4.csv"
ORACLE_CSV = OUT_DIR / "edge_oracle_task4.csv"

GAP_CAP_S = 15


def directed_key(df):
    """Travel-order (u, v) as ints, plus the undirected key (u_lo, v_hi)."""
    d = df.dropna(subset=["u", "v"]).copy()
    d["u"] = d["u"].astype(int)
    d["v"] = d["v"].astype(int)
    d["u_lo"] = np.minimum(d["u"], d["v"])
    d["v_hi"] = np.maximum(d["u"], d["v"])
    return d


def load_covariates(path=COVARIATES_CSV):
    """Static per-street covariates (run network_covariates_task1.py first)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run network_covariates_task1.py first")
    cov = pd.read_csv(path)
    cov["u_lo"] = np.minimum(cov["u"], cov["v"])
    cov["v_hi"] = np.maximum(cov["u"], cov["v"])
    return cov


def build_inventory(cov):
    """Every directed edge in the network: both orientations of every street,
    carrying that street's covariates."""
    digitised = set(zip(cov["u"], cov["v"]))
    streets = cov.drop_duplicates(subset=["u_lo", "v_hi"]).drop(columns=["u", "v"])
    fwd = streets.assign(u=streets["u_lo"], v=streets["v_hi"])
    rev = streets.assign(u=streets["v_hi"], v=streets["u_lo"])
    inv = pd.concat([fwd, rev], ignore_index=True)
    inv["is_digitised_dir"] = [(u, v) in digitised for u, v in zip(inv["u"], inv["v"])]
    return inv


def build_traversals(points_path=MATCHED_POINTS, cov=None):
    """One row per (directed edge, ride): when the ride was on the edge, how long,
    and the edge's length as distance ridden."""
    pts = gpd.read_file(points_path, layer="matched_points_task3",
                        columns=["traj_id", "boxId", "createdAt", "u", "v"],
                        ignore_geometry=True)
    pts["createdAt"] = pd.to_datetime(pts["createdAt"], utc=True)
    pts = pts.sort_values(["traj_id", "createdAt"])
    nxt = pts.groupby("traj_id")["createdAt"].shift(-1)
    pts["dwell_s"] = (nxt - pts["createdAt"]).dt.total_seconds().clip(upper=GAP_CAP_S)
    pts["dwell_s"] = pts["dwell_s"].fillna(pts["dwell_s"].median())
    pts = directed_key(pts)

    t = (pts.groupby(["u", "v", "traj_id"])
            .agg(boxId=("boxId", "first"), enter_time=("createdAt", "min"),
                 exit_time=("createdAt", "max"), n_points=("createdAt", "size"),
                 on_edge_s=("dwell_s", "sum"),
                 u_lo=("u_lo", "first"), v_hi=("v_hi", "first"))
            .reset_index())
    t["is_weekend"] = t["enter_time"].dt.dayofweek >= 5

    lengths = cov.drop_duplicates(subset=["u_lo", "v_hi"])[["u_lo", "v_hi", "length_m"]]
    return t.merge(lengths, on=["u_lo", "v_hi"], how="left")


def build_events(events_path=MATCHED_EVENTS):
    """One row per overtake event that got an edge."""
    ev = directed_key(gpd.read_file(events_path, layer="matched_events_task3",
                                    ignore_geometry=True))
    ev["start"] = pd.to_datetime(ev["start"], utc=True)
    ev["is_weekend"] = ev["start"].dt.dayofweek >= 5
    keep = ["u", "v", "u_lo", "v_hi", "event_uid", "traj_id", "boxId", "start",
            "is_weekend", "max_man_p", "min_clearance_cm", "edge_class", "link_via"]
    return ev[[c for c in keep if c in ev.columns]].rename(columns={"start": "time"})


def build_oracle(inventory, traversals, events):
    """One row per directed edge: exposure, counts, rates, covariates, and the
    temporal structure — weekday/weekend split of exposure & events,
    plus per-month traversal/event intensity."""
    trav = (traversals.groupby(["u", "v"])
            .agg(n_traversals=("traj_id", "size"), n_boxes=("boxId", "nunique"),
                 first_visit=("enter_time", "min"), last_visit=("enter_time", "max"),
                 rider_s=("on_edge_s", "sum"),
                 n_months=("enter_time",
                           lambda s: s.dt.tz_convert(None).dt.to_period("M").nunique()),
                 n_trav_weekend=("is_weekend", "sum"))
            .reset_index())
    ev = (events.groupby(["u", "v"])
          .agg(n_events=("time", "size"), n_events_weekend=("is_weekend", "sum"))
          .reset_index())

    o = (inventory.merge(trav, on=["u", "v"], how="left")
                  .merge(ev, on=["u", "v"], how="left"))
    for c, fill in (("n_traversals", 0), ("n_boxes", 0), ("n_events", 0),
                    ("rider_s", 0.0), ("n_months", 0), ("n_trav_weekend", 0),
                    ("n_events_weekend", 0)):
        o[c] = o[c].fillna(fill)
    for c in ("n_traversals", "n_events", "n_trav_weekend", "n_events_weekend"):
        o[c] = o[c].astype(int)

    o["rider_km"] = o["n_traversals"] * o["length_m"] / 1000
    o["rider_h"] = o["rider_s"] / 3600
    o["is_observed"] = o["n_traversals"] > 0

    # ---- temporal structure -------------------------------------------------
    o["n_trav_weekday"] = o["n_traversals"] - o["n_trav_weekend"]
    o["n_events_weekday"] = o["n_events"] - o["n_events_weekend"]
    o["rider_km_weekday"] = o["n_trav_weekday"] * o["length_m"] / 1000
    o["rider_km_weekend"] = o["n_trav_weekend"] * o["length_m"] / 1000
    # traversals / overtakes per active month (how often it is revisited)
    with np.errstate(divide="ignore", invalid="ignore"):
        o["trav_per_month"] = np.where(o["n_months"] > 0,
                                       o["n_traversals"] / o["n_months"], np.nan)
        o["events_per_month"] = np.where(o["n_months"] > 0,
                                         o["n_events"] / o["n_months"], np.nan)
        # mean seconds spent crossing the edge (exit_time - entry_time)
        o["sec_per_traversal"] = np.where(o["n_traversals"] > 0,
                                          o["rider_s"] / o["n_traversals"], np.nan)

        o["rate_per_traversal"] = np.where(o["n_traversals"] > 0,
                                           o["n_events"] / o["n_traversals"], np.nan)
        o["rate_per_rider_km"] = np.where(o["rider_km"] > 0,
                                          o["n_events"] / o["rider_km"], np.nan)
        o["rate_per_rider_h"] = np.where(o["rider_h"] > 0,
                                         o["n_events"] / o["rider_h"], np.nan)

    drop = ["rider_s", "is_sidepath", "is_cycling_street", "has_track_tag",
            "has_lane_tag", "n_accidents", "n_acc_severe", "n_accidents_recent",
            "n_acc_bike_recent", "aadt_hgv", "is_digitised_dir"]
    o = o.drop(columns=[c for c in drop if c in o.columns])

    front = ["u", "v", "u_lo", "v_hi", "edge_class", "is_observed",
             "n_traversals", "n_events", "rider_km", "rider_h", "n_boxes",
             "n_months", "trav_per_month", "events_per_month", "sec_per_traversal",
             "n_trav_weekday", "n_trav_weekend", "n_events_weekday",
             "n_events_weekend", "rider_km_weekday", "rider_km_weekend",
             "rate_per_traversal", "rate_per_rider_km", "rate_per_rider_h",
             "first_visit", "last_visit"]
    rest = [c for c in o.columns if c not in front]
    return o[front + rest]


if __name__ == "__main__":
    cov = load_covariates()
    inventory = build_inventory(cov)
    trav = build_traversals(cov=cov)
    ev = build_events()
    oracle = build_oracle(inventory, trav, ev)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trav.to_csv(TRAVERSALS_CSV, index=False)
    ev.to_csv(EVENTS_CSV, index=False)
    oracle.to_csv(ORACLE_CSV, index=False)
    print(f"oracle: {len(oracle)} directed edges "
          f"({oracle['is_observed'].sum()} ever ridden, "
          f"{oracle['is_observed'].mean():.0%}) | traversal rows {len(trav)} | "
          f"event rows {len(ev)}")
    print(f"saved: {TRAVERSALS_CSV.name}, {EVENTS_CSV.name}, {ORACLE_CSV.name}")

    obs = oracle[oracle["is_observed"]]
    ne, km = obs["n_events"].sum(), obs["rider_km"].sum()
    print(f"exposure: {km:.0f} rider-km, {obs['rider_h'].sum():.0f} rider-h, {ne} overtakes"
          f"  |  overall {ne / km:.2f}/rider-km, {ne / obs['n_traversals'].sum():.3f}/traversal")
    print(f"on-edge time: median {trav['on_edge_s'].median():.0f} s/crossing  |  "
          f"weekend {obs['rider_km_weekend'].sum() / km:.0%} of km, "
          f"{obs['n_events_weekend'].sum() / ne:.0%} of overtakes")

    pd.set_option("display.width", 200, "display.max_columns", 60)
    print("\nhead — one row per directed edge:")
    print(oracle.head().to_string())
