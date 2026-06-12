"""Acquire, classify, and plot the cyclable street network of Münster."""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("input/muenster_bike.graphml")
EDGES_PATH = Path("input/muenster_edges_classified.gpkg")
PLOT_PATH = Path("output/network_overview.png")
INSPECT_DIR = Path("output/inspection")

CRS_METRIC = "EPSG:25832"  # UTM 32N

EXTRA_TAGS = [
    "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
    "bicycle", "cyclestreet", "bicycle_road",
    "oneway:bicycle", "traffic_calming",
    "maxspeed", "surface", "lanes",
    "is_sidepath", "segregated", "footway",  # sidewalk/sidepath signals (DE tagging)
]
ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(EXTRA_TAGS))


def get_graph(force_download: bool = False):
    if GRAPH_PATH.exists() and not force_download:
        return ox.load_graphml(GRAPH_PATH)
    G = ox.graph_from_place(
        "Münster, North Rhine-Westphalia, Germany",
        network_type="bike", simplify=True,
    )
    G = ox.project_graph(G, to_crs=CRS_METRIC)
    GRAPH_PATH.parent.mkdir(exist_ok=True)
    ox.save_graphml(G, GRAPH_PATH)
    return G


def _norm(name, e):
    """Clean string series for a tag; resolves simplification list-values.

    Audit note: list values affect <0.1% of cycleway/bicycle tags and 2.6% of highway (mostly *_link merges).
    """
    if name not in e.columns:
        return pd.Series("", index=e.index)

    def resolve(v):
        if isinstance(v, list):
            vals = [x for x in v if x not in (None, "no", "none")]
            return vals[0] if vals else "no"
        return v

    return e[name].map(resolve).fillna("").astype(str)


def _main_bearing(geom):
    """Bearing (0-180 deg) of the longest segment of a LineString.

    More robust than endpoint bearing for curved edges (e.g. sidepaths rounding a junction corner).
    """
    coords = list(geom.coords)
    best_len, best_bearing = -1.0, 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg_len = np.hypot(x1 - x0, y1 - y0)
        if seg_len > best_len:
            best_len = seg_len
            best_bearing = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
    return best_bearing


def flag_sidepaths(edges, near_dist=12, max_dist=30, max_angle=25):
    """Flag car-free edges running parallel to a road carrying motor traffic.

    Two-band logic:
      - within near_dist m of a road: sidepath, unconditionally (perpendicular crossings that close are short and rare)
      - within near_dist..max_dist m: sidepath only if roughly parallel (bearing of longest segment within max_angle degrees of the road's)
    Combined (OR) with the explicit German OSM tag is_sidepath=yes.
    """
    # Double check whether the German tag assumption makes sense.

    e = edges.copy()
    hw = _norm("highway", e)
    CAR_FREE = {"cycleway", "path", "footway", "pedestrian", "track", "bridleway"}
    MOTOR = {"primary", "primary_link", "secondary", "secondary_link",
             "tertiary", "tertiary_link", "trunk", "trunk_link",
             "unclassified", "residential"}

    car_free = e[hw.isin(CAR_FREE)]
    roads = e[hw.isin(MOTOR)][["geometry"]].reset_index(drop=True)

    joined = gpd.sjoin_nearest(
        car_free[["geometry"]], roads, max_distance=max_dist,
        distance_col="dist_to_road", how="left",
    )
    joined = joined[~joined.index.duplicated(keep="first")]

    cf_bearing = joined.geometry.map(_main_bearing)
    rd_bearing = roads.geometry.map(_main_bearing).reindex(joined["index_right"]).values
    angle = (cf_bearing - rd_bearing).abs() % 180
    angle = np.minimum(angle, 180 - angle)

    near = joined["dist_to_road"] <= near_dist
    parallel_band = (joined["dist_to_road"] > near_dist) & (angle <= max_angle)
    geom_sidepath = (near | parallel_band) & joined["dist_to_road"].notna()

    e["is_sidepath"] = (
        geom_sidepath.reindex(e.index).fillna(False)
        | (_norm("is_sidepath", e) == "yes")
    )
    e["dist_to_road"] = joined["dist_to_road"].reindex(e.index)
    return e


def classify_edge_type(edges):
    """Assign one riding-regime class per edge, OSM-native and sidepath-aware.

    Precedence (np.select: first match wins), most specific first:
    dedicated bike infrastructure > on-carriageway infrastructure > calmed/low-traffic road types > generic road type.
    Overlaps are intentional and resolved by this order; the audit step reports how often precedence actually decided.
    """
    e = edges.copy()
    hw = _norm("highway", e)
    cw = pd.concat(
        [_norm(t, e) for t in
         ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]],
        axis=1,
    )
    LANE_VALS = {"lane", "shared_lane", "share_busway", "opposite_lane"}
    TRACK_VALS = {"track", "opposite_track"}
    MAIN = {"primary", "primary_link", "secondary", "secondary_link",
            "tertiary", "tertiary_link", "trunk", "trunk_link"} # unclassified not here. To think whether include.

    sidepath = e["is_sidepath"]
    shared_path = hw.isin(["path", "footway"])

    # covariate booleans (kept for modelling, independent of edge_class)
    e["is_cycling_street"] = (
        (_norm("cyclestreet", e) == "yes") | (_norm("bicycle_road", e) == "yes")
    )
    e["has_track_tag"] = cw.isin(TRACK_VALS).any(axis=1)
    e["has_lane_tag"] = cw.isin(LANE_VALS).any(axis=1)

    separated_geom = (hw == "cycleway") | shared_path
    e["separate_geometry"] = separated_geom

    conditions = [
        e["is_cycling_street"],                     # Fahrradstrasse: cars are guests
        separated_geom & sidepath,                  # separated track beside a road
        e["has_track_tag"],                         # same thing, mapped as road tag
        separated_geom & ~sidepath,                 # path away from motor traffic
        e["has_lane_tag"],                          # painted lane on the carriageway
        hw == "pedestrian",                         # riding among pedestrians
        hw.isin(["residential", "living_street"]),  # low-speed residential streets
        hw == "service",                            # access/parking/service ways
        hw.isin(["track", "bridleway"]),            # unpaved field/forest tracks
        hw == "busway",                             # bus lanes open to bikes
        hw == "unclassified",                       # minor connector roads
        hw.isin(MAIN),                              # arterial road, mixed traffic
    ]
    choices = [
        "bicycle_street",                           # Fahrradstrasse
        "roadside_track",                           # separated track along a road (red tiles etc.)
        "roadside_track",  # tag-mapped variant (has_track_tag). Duplicate is intentional bcs 12 conditions 12 choices.
        "independent_path",                         # cycleway/path away from motor traffic
        "painted_lane",                             # on-carriageway cycle lane (a.k.a. advisory lane)
        "pedestrian_zone",
        "residential_street",
        "service_way",
        "offroad_track",
        "bus_lane",                                 # bus and bikes share the carriageway
        "minor_road_shared",                        # mixed traffic, low volume
        "main_road_shared",                         # mixed traffic on an arterial road
    ]
    e["edge_class"] = np.select(conditions, choices, default="other")
    e.attrs["class_conditions"] = list(zip(choices, conditions))  # for audit
    return e


def inspect_classification(edges, column="edge_class", out_dir=INSPECT_DIR):
    """Diagnostics: how raw OSM tags were transformed into edge classes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    e = edges

    # 1. highway tag vs assigned class
    ct = pd.crosstab(_norm("highway", e), e[column], margins=True)
    ct.to_csv(out_dir / "highway_vs_label.csv")
    print("\n=== highway tag vs assigned class ===")
    print(ct.to_string())

    # 2. precedence audit: how many edges matched >1 condition?
    named_conditions = e.attrs.get("class_conditions", [])
    if named_conditions:
        cond_matrix = np.column_stack([c.values for _, c in named_conditions])
        n_multi = (cond_matrix.sum(axis=1) > 1).sum()
        print(f"\n=== precedence audit ===")
        print(f"edges matching >1 condition: {n_multi} ({n_multi / len(e):.1%})")
        names = [n for n, _ in named_conditions]
        overlap = pd.DataFrame(
            cond_matrix.T.astype(int) @ cond_matrix.astype(int),
            index=names, columns=names,
        )
        overlap.to_csv(out_dir / "condition_overlap.csv")

    # 3. 'other' should be ~empty; if not, a condition is missing
    n_other = (e[column] == "other").sum()
    print(f"\nedges classified 'other': {n_other}")
    if n_other:
        print(_norm("highway", e)[e[column] == "other"].value_counts().head(10).to_string())

    # 4. sidepath flag coverage
    if "is_sidepath" in e.columns:
        print(f"\nsidepath-flagged edges: {e['is_sidepath'].sum()} "
              f"({e['is_sidepath'].mean():.1%})")

    # 5. length-weighted shares
    km = e.groupby(column)["length"].sum() / 1000
    summary = pd.DataFrame({"km": km.round(1), "share": (km / km.sum()).round(3)})
    summary = summary.sort_values("km", ascending=False)
    summary.to_csv(out_dir / "length_by_label.csv")
    print("\n=== share of network by length (km) ===")
    print(summary.to_string())

    # 6. small multiples (classes with >=10 edges), titled with km not counts
    km_by_cat = e.groupby(column)["length"].sum() / 1000
    cats = [c for c in e[column].value_counts().index
            if (e[column] == c).sum() >= 10]
    ncols = 4
    nrows = -(-len(cats) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    for ax, cat in zip(axes.flat, cats):
        e.plot(ax=ax, color="0.9", linewidth=0.3)
        e[e[column] == cat].plot(ax=ax, color="crimson", linewidth=0.7)
        ax.set_title(f"{cat} ({km_by_cat[cat]:.0f} km)", fontsize=10)
        ax.set_axis_off()
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    fig.savefig(out_dir / "labels_small_multiples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_network(edges, column="edge_class", path=PLOT_PATH):
    categories = edges[column].value_counts().index
    cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(12, 12))
    for i, cat in enumerate(categories):
        subset = edges[edges[column] == cat]
        subset.plot(ax=ax, color=cmap(i), linewidth=0.6,
                    label=f"{cat} ({len(subset)})")
    ax.set_title("Münster cyclable network by riding regime")
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=8)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    G = get_graph()
    nodes, edges = ox.graph_to_gdfs(G)
    edges = flag_sidepaths(edges)
    edges = classify_edge_type(edges)

    print(edges["edge_class"].value_counts())
    inspect_classification(edges)

    # GPKG can't store lists/attrs: stringify residual list columns
    out = edges.drop(columns=["dist_to_road"], errors="ignore").copy()
    for c in out.columns:
        if c != "geometry" and out[c].map(lambda v: isinstance(v, list)).any():
            out[c] = out[c].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    out.to_file(EDGES_PATH, driver="GPKG")

    plot_network(edges)