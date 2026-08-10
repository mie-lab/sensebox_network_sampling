"""Acquire and classify the cyclable street network of Münster, then attach the static
per-edge covariates the risk model screens.

Two stages, one pass over the network:
  1. the network — download (cached), flag sidepaths, assign a riding regime per edge
  2. the covariates — length, speed limit, lanes, betweenness, accident counts, AADT

Stage 2 runs on the graph and the classified edges already in memory, and is the SLOW
step (approximate betweenness over ~90k edges), so it is cached like the download:
delete input/muenster_edge_covariates.csv to recompute.

Betweenness is EXPLORATORY, not a clean AADT stand-in: it is estimated from
BETWEENNESS_SAMPLES sampled source nodes, and it is confounded — high-betweenness
arterials in Münster mostly carry SEPARATED cycle tracks, so it ends up negatively
associated with the overtake rate, the opposite of the causal expectation. It does not
enter the Task-5 risk model; it appears only in the descriptive forest plots.

Accidents come from the Unfallatlas (police-reported injury accidents, already in
EPSG:25832) in input/accidents/; AADT from Straßen.NRW in input/traffic/. Neither has a
direction, so both are joined to a street via the undirected key (u_lo, v_hi) and
replicated to both directions.
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
PLOT_PATH = Path("output/task1_network/task1_network_overview.png")
INSPECT_DIR = Path("output/task1_network/task1_inspection")

CRS_METRIC = "EPSG:25832"  # UTM 32N

# AADT snapping. Straßen.NRW only publishes counts for federal/state/county roads
# (B/L/K), so most municipal streets get no value -- that is a property of the
# source, not a bug. A cycleway running alongside a counted road SHOULD inherit
# its traffic, hence the generous distance, but only if roughly parallel, so
# that streets merely crossing a main road do not inherit its volume.
AADT_SNAP_M = 25
AADT_MAX_ANGLE = 30

BETWEENNESS_SAMPLES = 1000    # source nodes sampled; higher = smoother, slower.
                              # 300 was visibly noisy; 1000 is the new default. Note
                              # this only sharpens a confounded, exploratory proxy
                              # (see module docstring) — it does not change the sign.
SEED = 42
ACCIDENT_SNAP_M = 30          # furthest an accident may sit from its edge
# Accident counts come in two flavours: all downloaded years (more years = more
# stable counts, the usual choice for a risk covariate) and the "recent" subset
# contemporaneous with the senseBox data. 2026 is not published yet.
ACCIDENT_RECENT_FROM = 2024

# fallback speeds (km/h) when OSM has no maxspeed, by road type
DEFAULT_SPEED = {
    "motorway": 100, "trunk": 80, "primary": 50, "secondary": 50, "tertiary": 50,
    "unclassified": 50, "residential": 30, "living_street": 7, "service": 20,
    "pedestrian": 7, "footway": 7, "path": 7, "cycleway": 20, "track": 20,
}

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
            "tertiary", "tertiary_link", "trunk", "trunk_link"}  # unclassified not here. To think whether include.

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
    ct.to_csv(out_dir / "task1_highway_vs_label.csv")
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
        overlap.to_csv(out_dir / "task1_condition_overlap.csv")

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
    summary.to_csv(out_dir / "task1_length_by_label.csv")
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
    fig.savefig(out_dir / "task1_labels_small_multiples.png", dpi=200, bbox_inches="tight")
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


# ===================== static per-edge covariates =====================

def parse_maxspeed(v):
    """OSM maxspeed -> km/h. Handles '30', '30 mph', 'walk', and multi-values
    ('30;50', stringified lists) by taking the FIRST number -- concatenating the
    digits instead would turn '30;50' into 3050."""
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if s in ("walk", "schrittgeschwindigkeit"):
        return 7.0
    m = re.search(r"\d+", s)
    if not m:
        return np.nan                      # 'none', 'signals', 'variable', junk
    val = float(m.group()) * (1.609 if "mph" in s else 1.0)
    return val if 3 <= val <= 150 else np.nan   # implausible for a city street


def _first(v):
    return v[0] if isinstance(v, list) and v else (np.nan if isinstance(v, list) else v)


def edge_betweenness(G, k=BETWEENNESS_SAMPLES):
    """Approximate edge betweenness per directed (u, v), length-weighted."""
    # collapse the multigraph: betweenness needs a simple DiGraph; keep the
    # shortest parallel edge, which is the one a router would use anyway
    D = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        if not D.has_edge(u, v) or length < D[u][v]["length"]:
            D.add_edge(u, v, length=length)
    print(f"[cov] betweenness on {D.number_of_edges()} edges from {k} sources...", flush=True)
    t0 = time.time()
    bc = nx.edge_betweenness_centrality(D, k=k, weight="length", seed=SEED)
    print(f"[cov]   done in {time.time() - t0:.0f} s")
    return pd.DataFrame([{"u": int(u), "v": int(v), "betweenness": b}
                         for (u, v), b in bc.items()])


def _read_accident_zip(path):
    """One Unfallatlas year out of its zip (member is .csv in some years, .txt in others)."""
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist()
                      if n.lower().endswith((".csv", ".txt")) and "schema" not in n.lower())
        with z.open(member) as f:
            return pd.read_csv(f, sep=";", dtype=str, encoding="utf-8-sig",
                               low_memory=False)


def accident_counts(edges_geom, acc_dir=ACCIDENT_DIR, max_dist=ACCIDENT_SNAP_M):
    """Accidents snapped to the nearest street, counted per undirected edge.
    Returns total / bicycle-involved / severe (killed or seriously injured)."""
    zips = sorted(acc_dir.glob("Unfallorte*_CSV.zip"))
    if not zips:
        print(f"[cov] no accident zips in {acc_dir} -> skipping accident covariates")
        return None
    acc = pd.concat([_read_accident_zip(p) for p in zips], ignore_index=True)
    for c in ("LINREFX", "LINREFY"):
        acc[c] = pd.to_numeric(acc[c].str.replace(",", ".", regex=False), errors="coerce")
    acc = acc.dropna(subset=["LINREFX", "LINREFY"])

    minx, miny, maxx, maxy = edges_geom.total_bounds
    acc = acc[acc["LINREFX"].between(minx, maxx) & acc["LINREFY"].between(miny, maxy)]
    years = sorted(acc["UJAHR"].dropna().unique())

    g = gpd.GeoDataFrame(
        acc, geometry=gpd.points_from_xy(acc["LINREFX"], acc["LINREFY"]),
        crs=CRS_METRIC)
    g["is_bike"] = g["IstRad"].astype(str).str.strip().eq("1")
    g["is_severe"] = g["UKATEGORIE"].astype(str).str.strip().isin(["1", "2"])
    g["year"] = pd.to_numeric(g["UJAHR"], errors="coerce")

    j = gpd.sjoin_nearest(g[["geometry", "is_bike", "is_severe", "year"]],
                          edges_geom[["u_lo", "v_hi", "geometry"]],
                          max_distance=max_dist, how="inner")
    # an accident equidistant from parallel edges is returned once per edge;
    # keep one so a single crash is never counted twice
    j = j[~j.index.duplicated(keep="first")]

    recent = j["year"] >= ACCIDENT_RECENT_FROM
    j["bike_recent"] = j["is_bike"] & recent
    j["any_recent"] = recent
    print(f"[cov] accidents {years[0]}-{years[-1]}: {len(j)}/{len(g)} snapped within "
          f"{max_dist} m, {int(recent.sum())} from {ACCIDENT_RECENT_FROM}+")
    return (j.groupby(["u_lo", "v_hi"])
            .agg(n_accidents=("is_bike", "size"), n_acc_bike=("is_bike", "sum"),
                 n_acc_severe=("is_severe", "sum"),
                 n_accidents_recent=("any_recent", "sum"),
                 n_acc_bike_recent=("bike_recent", "sum"))
            .reset_index())


def aadt_per_edge(edges_geom, path=TRAFFIC_ZIP, max_dist=AADT_SNAP_M,
                  max_angle=AADT_MAX_ANGLE):
    """Motor-traffic AADT (DTVKFZA) and heavy-vehicle AADT (DTVSVA) per street,
    taken from the nearest roughly-parallel counted road segment."""
    if not path.exists():
        print(f"[cov] {path} not found -> skipping AADT")
        return None
    vw = gpd.read_file(f"zip://{path}!VERKEHRSWERTE_line.shp")
    minx, miny, maxx, maxy = edges_geom.total_bounds
    vw = vw.cx[minx:maxx, miny:maxy].copy()
    vw = vw[vw["DTVKFZA"].notna() & (vw["DTVKFZA"] > 0)]
    vw["vw_bearing"] = vw.geometry.map(_main_bearing)

    e = edges_geom.copy()
    e["edge_bearing"] = e.geometry.map(_main_bearing)
    rep = e.copy()
    rep["geometry"] = rep.geometry.representative_point()

    j = gpd.sjoin_nearest(
        rep[["u_lo", "v_hi", "edge_bearing", "geometry"]],
        vw[["DTVKFZA", "DTVSVA", "vw_bearing", "geometry"]],
        max_distance=max_dist, how="inner", distance_col="d")
    j = j[~j.index.duplicated(keep="first")]

    ang = (j["edge_bearing"] - j["vw_bearing"]).abs() % 180
    ang = np.minimum(ang, 180 - ang)
    j = j[ang <= max_angle]
    print(f"[cov] AADT: {len(vw)} counted segments nearby, {len(j)} edges within "
          f"{max_dist} m of a parallel one ({len(j) / len(e):.1%} of the network)")
    return (j.groupby(["u_lo", "v_hi"])
            .agg(aadt_kfz=("DTVKFZA", "median"), aadt_hgv=("DTVSVA", "median"))
            .reset_index())


def build_covariates(edges, G):
    """One row per directed (u, v): class, length, speed limit, lanes, betweenness,
    accident counts and AADT. Works on the graph and the classified edges already in
    memory — nothing is re-read from disk."""
    e = edges.reset_index()
    # classify_edge_type stashes its precedence masks in .attrs for the audit in
    # inspect_classification(). They are Series, and pandas compares .attrs elementwise
    # when it concatenates multi-column aggregation results -> "truth value is ambiguous".
    # The covariates do not need the audit trail, so drop it.
    e.attrs = {}
    keep = ["u", "v", "edge_class", "length", "highway", "maxspeed", "lanes",
            "is_sidepath", "is_cycling_street", "has_track_tag", "has_lane_tag"]
    a = e[[c for c in keep if c in e.columns]].copy()
    for c in ("highway", "maxspeed", "lanes"):
        if c in a.columns:
            a[c] = a[c].map(_first)

    a["maxspeed_kmh"] = a["maxspeed"].map(parse_maxspeed)
    hw = a["highway"].astype(str).str.replace("_link", "", regex=False)
    a["maxspeed_kmh"] = a["maxspeed_kmh"].fillna(hw.map(DEFAULT_SPEED))
    a["lanes_n"] = pd.to_numeric(a["lanes"], errors="coerce")

    # one row per directed (u, v); parallel edges collapse (matching cannot
    # distinguish them anyway) taking the shortest
    a = a.sort_values("length")
    agg = {c: "first" for c in
           ["edge_class", "highway", "maxspeed_kmh", "lanes_n", "is_sidepath",
            "is_cycling_street", "has_track_tag", "has_lane_tag"] if c in a.columns}
    agg["length"] = "first"
    cov = a.groupby(["u", "v"], as_index=False).agg(agg).rename(
        columns={"length": "length_m"})

    cov = cov.merge(edge_betweenness(G), on=["u", "v"], how="left")
    cov["betweenness"] = cov["betweenness"].fillna(0.0)

    # accidents and AADT belong to the STREET, not to a direction: join on the
    # undirected key and let both directions inherit the same value
    for df in (cov, e):
        df["u_lo"] = np.minimum(df["u"], df["v"])
        df["v_hi"] = np.maximum(df["u"], df["v"])
    geom = e[["u_lo", "v_hi", "geometry"]]

    acc = accident_counts(geom)
    if acc is not None:
        cov = cov.merge(acc, on=["u_lo", "v_hi"], how="left")
        for c in ("n_accidents", "n_acc_bike", "n_acc_severe",
                  "n_accidents_recent", "n_acc_bike_recent"):
            cov[c] = cov[c].fillna(0).astype(int)

    aadt = aadt_per_edge(geom)          # NaN where the source has no count
    if aadt is not None:
        cov = cov.merge(aadt, on=["u_lo", "v_hi"], how="left")
    return cov


def main(force_covariates=False):
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

    # the slow half — cached like the download, since betweenness dominates the runtime
    if COVARIATES_CSV.exists() and not force_covariates:
        print(f"[cov] {COVARIATES_CSV.name} exists -> skipped (delete it to recompute)")
        return
    cov = build_covariates(edges, G)
    cov.to_csv(COVARIATES_CSV, index=False)
    streets = cov.drop_duplicates(subset=["u_lo", "v_hi"])
    bits = [f"maxspeed {cov['maxspeed_kmh'].notna().mean():.0%}"]
    if "n_acc_bike" in cov.columns:
        bits.append(f"{streets['n_acc_bike'].sum()} bike accidents on "
                    f"{(streets['n_acc_bike'] > 0).sum()} streets")
    if "aadt_kfz" in cov.columns:
        bits.append(f"AADT on {streets['aadt_kfz'].notna().mean():.1%} of streets")
    print(f"[cov] {len(cov)} directed edges -> {COVARIATES_CSV}  |  " + "  |  ".join(bits))


if __name__ == "__main__":
    main()