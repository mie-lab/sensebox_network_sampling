"""Build the street network and attach its static covariates in one pass.

The network: download Muenster's cyclable graph from OSM (cached), flag sidepaths, and
give every edge one of 13 riding regimes.

The covariates: length, speed limit, lanes, betweenness, bike accidents and AADT.
This is the slow half, so it is cached. Accidents (Unfallatlas) and AADT (Strassen.NRW)
have no direction, so both are joined per street on (u_lo, v_hi) and copied to both
directions (!!!).

Writes to input/:
  muenster_edges_classified.gpkg   every edge with its riding regime
  muenster_edge_covariates.csv     one row per directed edge
and to output/task1_network/task1_inspection/: the classification audit and its figures.
"""

from pathlib import Path
import re
import time
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("input/muenster_bike.graphml")
EDGES_PATH = Path("input/muenster_edges_classified.gpkg")
COVARIATES_CSV = Path("input/muenster_edge_covariates.csv")
ACCIDENT_DIR = Path("input/accidents")
TRAFFIC_ZIP = Path("input/traffic/Verkehrswerte.zip")
INSPECT_DIR = Path("output/task1_network/task1_inspection")

CRS_METRIC = "EPSG:25832"  # UTM 32N

AADT_SNAP_M = 25
AADT_MAX_ANGLE = 30
SIDEPATH_NEAR_M = 12          # this close to a road it is a sidepath whatever its bearing
SIDEPATH_MAX_M = 30
SIDEPATH_MAX_ANGLE = 25
BETWEENNESS_SAMPLES = 500     # source nodes sampled; higher = smoother but slower.
SEED = 42
ACCIDENT_SNAP_M = 30          # furthest an accident may sit from its edge. 2026 is not published yet.
ACCIDENT_RECENT_FROM = 2024

SEPARATED_HIGHWAY = {"cycleway", "path", "footway"}
EXTRA_TAGS = [
    "cycleway", "cycleway:left", "cycleway:right", "cycleway:both",
    "bicycle", "cyclestreet", "bicycle_road",
    "oneway:bicycle", "traffic_calming",
    "maxspeed", "surface", "lanes",
    "is_sidepath", "segregated", "footway",  # sidewalk/sidepath signals (DE tagging)
]
ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(EXTRA_TAGS))


def get_graph():
    """Muenster's cyclable graph from OSM, cached to GRAPH_PATH; delete it to re-download."""
    if GRAPH_PATH.exists():
        print(f"Loading cached graph {GRAPH_PATH}", flush=True)
        return ox.load_graphml(GRAPH_PATH)

    print("Downloading the cyclable network from OSM...", flush=True)
    G = ox.graph_from_place(
        "Münster, North Rhine-Westphalia, Germany",
        network_type="bike", simplify=True,
    )
    G = ox.project_graph(G, to_crs=CRS_METRIC)
    GRAPH_PATH.parent.mkdir(exist_ok=True)
    ox.save_graphml(G, GRAPH_PATH)
    print(f"[graph] saved -> {GRAPH_PATH} "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


def _norm(name, e):
    """One OSM tag as a plain string Series: no NaN, no lists, safe to compare.
    Merged ways can disagree on a tag; the first non-negative value wins.
    A tag that was never downloaded returns empty, so it matches nothing.
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
    More robust than endpoint bearing for curved edges.
    """
    coords = list(geom.coords)
    best_len, best_bearing = -1.0, 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg_len = np.hypot(x1 - x0, y1 - y0)
        if seg_len > best_len:
            best_len = seg_len
            best_bearing = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
    return best_bearing


def _angle_between(a, b):
    """Smallest angle (0-90 deg) between two bearings, so opposite headings read as 0."""
    d = (a - b).abs() % 180
    return np.minimum(d, 180 - d)


def flag_sidepaths(edges, near_dist=SIDEPATH_NEAR_M, max_dist=SIDEPATH_MAX_M, max_angle=SIDEPATH_MAX_ANGLE):
    """Flag separated cycling geometry running alongside a road carrying motor traffic.
    Near the road any bearing counts, further out only a roughly parallel one.
    """
    e = edges.copy()
    hw = _norm("highway", e)
    MOTOR = {"primary", "primary_link", "secondary", "secondary_link",
             "tertiary", "tertiary_link", "trunk", "trunk_link",
             "unclassified", "residential"}

    car_free = e[hw.isin(SEPARATED_HIGHWAY)]
    roads = e[hw.isin(MOTOR)][["geometry"]].reset_index(drop=True)

    joined = gpd.sjoin_nearest(
        car_free[["geometry"]], roads, max_distance=max_dist,
        distance_col="dist_to_road", how="left",
    )
    # a path equidistant from two roads comes back once per road; keep first
    joined = joined[~joined.index.duplicated(keep="first")]

    cf_bearing = joined.geometry.map(_main_bearing)
    rd_bearing = roads.geometry.map(_main_bearing).reindex(joined["index_right"]).values
    angle = _angle_between(cf_bearing, rd_bearing)

    # unmatched edges have a NaN distance, and NaN fails both comparisons
    dist = joined["dist_to_road"]
    geom_sidepath = (dist <= near_dist) | ((dist > near_dist) & (angle <= max_angle))

    e["is_sidepath"] = (
        geom_sidepath.reindex(e.index).fillna(False)
        | (_norm("is_sidepath", e) == "yes")
    )
    return e


def classify_edge_type(edges):
    """Assign one riding regime per edge.
    Where a road also carries cycling provision the provision wins,
    which decides 2% of edges; the other 98% match a single rule.
    """
    e = edges.copy()
    hw, svc, name = _norm("highway", e), _norm("service", e), _norm("name", e)
    cw = pd.concat(
        [_norm(t, e) for t in
         ["cycleway", "cycleway:left", "cycleway:right", "cycleway:both"]],
        axis=1,
    )
    LANE_VALS = {"lane", "shared_lane", "share_busway", "opposite_lane"}
    TRACK_VALS = {"track", "opposite_track"}
    MAIN = {"primary", "primary_link", "secondary", "secondary_link",
            "tertiary", "tertiary_link", "trunk", "trunk_link"}  # unclassified not here. To think whether include.
    ACCESS_VALS = {"driveway", "parking_aisle", "alley"}

    sidepath = e["is_sidepath"]

    # covariate booleans (kept for modelling, independent of edge_class)
    e["is_cycling_street"] = ((_norm("cyclestreet", e) == "yes") | (_norm("bicycle_road", e) == "yes"))
    e["has_track_tag"] = cw.isin(TRACK_VALS).any(axis=1)
    e["has_lane_tag"] = cw.isin(LANE_VALS).any(axis=1)
    separated_geom = hw.isin(SEPARATED_HIGHWAY)
    e["separate_geometry"] = separated_geom
    shares_busway = cw.isin(["share_busway"]).any(axis=1)  # bus_lane has to outrank painted_lane below

    rules = [
        ("promenade",          name == "Promenade"),                        # Muenster's car-free ring
        ("bicycle_street",     e["is_cycling_street"]),                     # Fahrradstrasse
        ("roadside_track",     separated_geom & sidepath),                  # its own way, beside a road
        ("roadside_track",     e["has_track_tag"]),                         # the same, tagged on the carriageway
        ("independent_path",   separated_geom & ~sidepath),                 # its own way, away from any road
        ("bus_lane",           (hw == "busway") | shares_busway),
        ("painted_lane",       e["has_lane_tag"]),
        ("pedestrian_zone",    hw == "pedestrian"),
        ("residential_street", hw.isin(["residential", "living_street"])),  # residential street, calm
        ("access_way",         (hw == "service") & svc.isin(ACCESS_VALS)),  # destination only
        ("service_way",        (hw == "service") & ~svc.isin(ACCESS_VALS)), # ridden as a through route
        ("offroad_track",      hw.isin(["track", "bridleway"])),            # Feldweg: farm traffic, etc.
        ("minor_road_shared",  hw == "unclassified"),                       # OSM's minor through road, below tertiary
        ("major_road_shared",  hw.isin(MAIN)),
    ]
    e["edge_class"] = np.select([c for _, c in rules], [r for r, _ in rules], default="other")
    e.attrs["class_conditions"] = rules   # read by inspect_classification
    return e


# ========= inspection =========


def inspect_classification(edges, column="edge_class", out_dir=INSPECT_DIR):
    """Diagnostics: how raw OSM tags were transformed into edge classes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    e = edges

    ct = pd.crosstab(_norm("highway", e), e[column], margins=True)
    print("\n=== highway tag vs assigned class ===")
    print(ct.to_string())
    ct.to_csv(out_dir / "task1_highway_vs_label.csv")
    print(f"[csv] saved -> {out_dir / 'task1_highway_vs_label.csv'}")

    # a GPKG round-trip loses .attrs, so the precedence audit is only available in memory
    named_conditions = e.attrs.get("class_conditions", [])
    if named_conditions:
        cond_matrix = np.column_stack([c.values for _, c in named_conditions])
        n_multi = (cond_matrix.sum(axis=1) > 1).sum()
        print(f"\nedges matching >1 condition: {n_multi} ({n_multi / len(e):.1%})")
        names = [n for n, _ in named_conditions]
        overlap = pd.DataFrame(cond_matrix.T.astype(int) @ cond_matrix.astype(int),
                               index=names, columns=names)
        overlap.to_csv(out_dir / "task1_condition_overlap.csv")
        print(f"[csv] saved -> {out_dir / 'task1_condition_overlap.csv'}")

    n_other = (e[column] == "other").sum()          # anything here means a rule is missing
    print(f"\nedges classified 'other': {n_other}")
    if n_other:
        print(_norm("highway", e)[e[column] == "other"].value_counts().head(10).to_string())

    print(f"sidepath-flagged edges: {e['is_sidepath'].sum()} ({e['is_sidepath'].mean():.1%})")

    g = e.groupby(column)["length"]
    km = g.sum() / 1000
    summary = pd.DataFrame({"edges": g.size(), "km": km.round(1),
                            "share": (km / km.sum()).round(3)}
                           ).sort_values("km", ascending=False)
    print("\n=== network by riding regime ===")
    print(summary.to_string())
    summary.to_csv(out_dir / "task1_length_by_label.csv")
    print(f"[csv] saved -> {out_dir / 'task1_length_by_label.csv'}")


# ========= plotting =========


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {path}")


def fig_class_small_multiples(edges, column="edge_class", out_dir=INSPECT_DIR, ncols=4, min_edges=10):
    """One panel per regime, the rest of the network greyed behind it."""
    km = edges.groupby(column)["length"].sum() / 1000
    cats = [c for c in edges[column].value_counts().index
            if (edges[column] == c).sum() >= min_edges]
    nrows = -(-len(cats) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    for ax, cat in zip(axes.flat, cats):
        edges.plot(ax=ax, color="lightgrey", linewidth=0.3)  # the rest of the network
        edges[edges[column] == cat].plot(ax=ax, color="crimson", linewidth=0.7)
        ax.set_title(f"{cat} ({km[cat]:.0f} km)", fontsize=11)
        ax.set_axis_off()
    for ax in axes.flat[len(cats):]:
        ax.set_visible(False)
    _save(fig, out_dir / "task1_labels_small_multiples.png")


def fig_network_overview(edges, column="edge_class", out_dir=INSPECT_DIR):
    """The whole network, coloured by riding regime."""
    # sorted, so a regime keeps its colour between runs.
    colors = dict(zip(sorted(edges[column].unique()), plt.get_cmap("tab20").colors))

    fig, ax = plt.subplots(figsize=(12, 12))
    for cat, color in colors.items():
        subset = edges[edges[column] == cat]
        subset.plot(ax=ax, color=color, linewidth=0.6, label=f"{cat} ({len(subset)})")
    ax.set_title("Münster cyclable network by riding regime", fontsize=11)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=11)
    _save(fig, out_dir / "task1_network_overview.png")


# ========= static covariates =========


def _first(v):
    """First value of a multi-valued OSM tag, left alone if it is not a list."""
    if not isinstance(v, list):
        return v
    return v[0] if v else np.nan


def _parse_maxspeed(v):
    """OSM maxspeed (free text) -> km/h, NaN when it says nothing usable."""
    v = _first(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if s in ("walk", "schrittgeschwindigkeit"):
        return 7.0
    m = re.search(r"\d+", s)
    if not m:
        return np.nan # 'none', 'signals', 'variable', junk
    return float(m.group()) * (1.609 if "mph" in s else 1.0)


def edge_betweenness(G, k=BETWEENNESS_SAMPLES):
    """Approximate edge betweenness per directed (u, v), length-weighted."""

    # collapse the multigraph to a simple DiGraph
    D = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        if not D.has_edge(u, v) or length < D[u][v]["length"]:
            D.add_edge(u, v, length=length)
    print(f"Betweenness on {D.number_of_edges()} edges from {k} sources...", flush=True)

    t0 = time.time()
    bc = nx.edge_betweenness_centrality(D, k=k, weight="length", seed=SEED)
    print(f"Finished in {time.time() - t0:.0f} s")

    return pd.DataFrame([(int(u), int(v), b) for (u, v), b in bc.items()],
                        columns=["u", "v", "betweenness"])


def _read_accident_zip(path):
    """One Unfallatlas year out of its zip (member is .csv in some years, .txt in others)."""
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist()
                      if n.lower().endswith((".csv", ".txt")) and "schema" not in n.lower())
        with z.open(member) as f:
            return pd.read_csv(f, sep=";", dtype=str, encoding="utf-8-sig",
                               low_memory=False)


def accident_counts(edges_geom, acc_dir=ACCIDENT_DIR, max_dist=ACCIDENT_SNAP_M):
    """Bike accidents per undirected street (u_lo, v_hi), from the nearest edge within max_dist."""
    zips = sorted(acc_dir.glob("Unfallorte*_CSV.zip"))
    if not zips:
        print(f"No accident zips in {acc_dir} -> skipping accident counts")
        return None

    acc = pd.concat([_read_accident_zip(p) for p in zips], ignore_index=True)
    acc = acc[acc["IstRad"].astype(str).str.strip().eq("1")]  # IstRad: a bike was involved

    for c in ("LINREFX", "LINREFY"):
        acc[c] = pd.to_numeric(acc[c].str.replace(",", ".", regex=False), errors="coerce")

    minx, miny, maxx, maxy = edges_geom.total_bounds  # nationwide file, keep Muenster
    acc = acc[acc["LINREFX"].between(minx, maxx) & acc["LINREFY"].between(miny, maxy)]

    year = pd.to_numeric(acc["UJAHR"], errors="coerce")
    accidents = gpd.GeoDataFrame({"is_recent": year >= ACCIDENT_RECENT_FROM},
                                 geometry=gpd.points_from_xy(acc["LINREFX"], acc["LINREFY"]),
                                 crs=CRS_METRIC)

    matched = gpd.sjoin_nearest(accidents, edges_geom[["u_lo", "v_hi", "geometry"]],
                                max_distance=max_dist, how="inner")
    # an accident equidistant from two edges comes back once per edge; count it once
    matched = matched[~matched.index.duplicated(keep="first")]

    print(f"Bike accidents: {len(matched)}/{len(accidents)} snapped within {max_dist} m")
    return (matched.groupby(["u_lo", "v_hi"])
            .agg(n_acc_bike=("is_recent", "size"),
                 n_acc_bike_recent=("is_recent", "sum"))
            .reset_index())


def aadt_per_edge(edges_geom, path=TRAFFIC_ZIP, max_dist=AADT_SNAP_M, max_angle=AADT_MAX_ANGLE):
    """Motor AADT ["DTVKFZA"] and heavy-vehicle AADT ["DTVSVA"] per undirected street
    (u_lo, v_hi), from the nearest parallel counted segment within max_dist.
    Dataset covers only about 3% of edges.
    """
    if not path.exists():
        print(f"No traffic counts at {path} -> skipping AADT")
        return None

    counts = gpd.read_file(f"zip://{path}!VERKEHRSWERTE_line.shp")
    minx, miny, maxx, maxy = edges_geom.total_bounds  # NRW-wide file, keep Muenster
    counts = counts.cx[minx:maxx, miny:maxy].copy()
    counts = counts[counts["DTVKFZA"].notna() & (counts["DTVKFZA"] > 0)]
    counts["count_bearing"] = counts.geometry.map(_main_bearing)

    edge_points = edges_geom.copy()
    edge_points["edge_bearing"] = edge_points.geometry.map(_main_bearing)

    # join from a point guaranteed to sit on the edge
    edge_points["geometry"] = edge_points.geometry.representative_point()

    matched = gpd.sjoin_nearest(
        edge_points[["u_lo", "v_hi", "edge_bearing", "geometry"]],
        counts[["DTVKFZA", "DTVSVA", "count_bearing", "geometry"]],
        max_distance=max_dist, how="inner")
    matched = matched[~matched.index.duplicated(keep="first")]

    # a street merely crossing a counted road must not inherit its traffic
    matched = matched[_angle_between(matched["edge_bearing"], matched["count_bearing"]) <= max_angle]

    print(f"AADT: {len(matched)}/{len(edge_points)} edges matched within {max_dist} m")
    return (matched.groupby(["u_lo", "v_hi"])
            .agg(aadt_kfz=("DTVKFZA", "median"), aadt_hgv=("DTVSVA", "median"))
            .reset_index())


def build_covariates(edges, G):
    """One row per directed (u, v): regime, length, speed limit, lanes, betweenness,
    bike accidents and AADT, from the graph and classified edges already in memory.
    """
    e = edges.reset_index()
    e.attrs = {}

    keep = ["u", "v", "edge_class", "length", "highway", "maxspeed", "lanes",
            "is_sidepath", "is_cycling_street", "has_track_tag", "has_lane_tag"]
    edge_rows = e[[c for c in keep if c in e.columns]].copy()
    for c in ("highway", "maxspeed", "lanes"):
        if c in edge_rows.columns:
            edge_rows[c] = edge_rows[c].map(_first)

    # NaN where OSM is silent
    edge_rows["maxspeed_kmh"] = edge_rows["maxspeed"].map(_parse_maxspeed)
    edge_rows["lanes_n"] = pd.to_numeric(edge_rows["lanes"], errors="coerce")
    edge_rows = edge_rows.sort_values("length").drop(columns=["maxspeed", "lanes"], errors="ignore")
    agg = {c: "first" for c in edge_rows.columns if c not in ("u", "v")}
    cov = edge_rows.groupby(["u", "v"], as_index=False).agg(agg).rename(columns={"length": "length_m"})
    cov = cov.merge(edge_betweenness(G), on=["u", "v"], how="left")
    cov["betweenness"] = cov["betweenness"].fillna(0.0)

    for df in (cov, e):  # both directions inherit the street's value (!!!)
        df["u_lo"] = np.minimum(df["u"], df["v"])
        df["v_hi"] = np.maximum(df["u"], df["v"])
    edges_geom = e[["u_lo", "v_hi", "geometry"]]

    acc = accident_counts(edges_geom)
    if acc is not None:
        cov = cov.merge(acc, on=["u_lo", "v_hi"], how="left")
        for c in ("n_acc_bike", "n_acc_bike_recent"):
            cov[c] = cov[c].fillna(0).astype(int)

    aadt = aadt_per_edge(edges_geom)
    if aadt is not None:
        cov = cov.merge(aadt, on=["u_lo", "v_hi"], how="left")
    return cov


def write_edges(edges, path=EDGES_PATH):
    """Classified edges to GPKG, stringifying the list tags it cannot store."""
    out = edges.copy()
    for c in out.columns:
        if c != "geometry" and out[c].map(lambda v: isinstance(v, list)).any():
            out[c] = out[c].map(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    out.to_file(path, driver="GPKG")
    print(f"[gpkg] saved -> {path}")


def main():
    G = get_graph()
    _, edges = ox.graph_to_gdfs(G)
    edges = flag_sidepaths(edges)
    edges = classify_edge_type(edges)

    inspect_classification(edges)
    fig_class_small_multiples(edges)
    fig_network_overview(edges)
    write_edges(edges)

    # betweenness dominates the runtime, so this half is cached
    if COVARIATES_CSV.exists():
        print(f"{COVARIATES_CSV.name} exists -> skipped (delete it to recompute)")
    else:
        cov = build_covariates(edges, G)
        cov.to_csv(COVARIATES_CSV, index=False)
        print(f"[csv] saved -> {COVARIATES_CSV} ({len(cov)} directed edges)")
    print("\nDONE")


if __name__ == "__main__":
    main()