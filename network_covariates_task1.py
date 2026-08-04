"""Static per-edge covariates for the risk model (cached — this is the slow step).

Produces one row per DIRECTED edge (u, v) with:
  edge_class, length_m, maxspeed_kmh, highway, lanes, the infrastructure
  booleans, edge betweenness centrality, and recorded accident counts.

Betweenness is approximate: exact edge betweenness on ~90k edges is far too slow,
so it is estimated from BETWEENNESS_SAMPLES randomly chosen source nodes with
shortest paths weighted by length. It is meant as a proxy for through-traffic
volume (the covariate AADT would otherwise provide) — but treat it as EXPLORATORY,
not a clean AADT stand-in: it is a confounded proxy. High-betweenness arterials in
Münster mostly carry SEPARATED cycle tracks, so betweenness ends up NEGATIVELY
associated with the recorded overtake rate (the opposite of the causal expectation),
because it also indexes infrastructure type. It does not enter the Task-5 risk
model; it appears only in the descriptive forest plots.

Accidents come from the Unfallatlas (police-reported injury accidents, already in
EPSG:25832) in input/accidents/. Each accident is snapped to the nearest edge
within ACCIDENT_SNAP_M. Accidents have no direction, so a count is a property of
the street and is joined to both directions via the undirected key.

Writes input/muenster_edge_covariates.csv (cached; delete to recompute).
"""
from pathlib import Path
import re
import time
import zipfile

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

GRAPH_PATH = Path("input/muenster_bike.graphml")
EDGES_PATH = Path("input/muenster_edges_classified.gpkg")
ACCIDENT_DIR = Path("input/accidents")
TRAFFIC_ZIP = Path("input/traffic/Verkehrswerte.zip")
OUT_CSV = Path("input/muenster_edge_covariates.csv")

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


def edge_betweenness(graph_path=GRAPH_PATH, k=BETWEENNESS_SAMPLES):
    """Approximate edge betweenness per directed (u, v), length-weighted."""
    G = ox.load_graphml(graph_path)
    # collapse the multigraph: betweenness needs a simple DiGraph; keep the
    # shortest parallel edge, which is the one a router would use anyway
    D = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 1.0))
        if not D.has_edge(u, v) or length < D[u][v]["length"]:
            D.add_edge(u, v, length=length)
    print(f"[cent] graph: {D.number_of_nodes()} nodes, {D.number_of_edges()} directed edges")
    print(f"[cent] approximating edge betweenness from {k} source nodes...")
    t0 = time.time()
    bc = nx.edge_betweenness_centrality(D, k=k, weight="length", seed=SEED)
    print(f"[cent] done in {time.time() - t0:.0f} s")
    return pd.DataFrame([{"u": int(u), "v": int(v), "betweenness": b}
                         for (u, v), b in bc.items()])


def _read_accident_zip(path):
    """One Unfallatlas year out of its zip (member is .csv in some years, .txt in others."""
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist()
                      if n.lower().endswith((".csv", ".txt")) and "schema" not in n.lower())
        with z.open(member) as f:
            return pd.read_csv(f, sep=";", dtype=str, encoding="utf-8-sig",
                               low_memory=False)


def accident_counts(edges_geom, acc_dir=ACCIDENT_DIR, max_dist=ACCIDENT_SNAP_M):
    """Accidents snapped to the nearest edge, counted per undirected edge.
    Returns total / bicycle-involved / severe (killed or seriously injured)."""
    zips = sorted(acc_dir.glob("Unfallorte*_CSV.zip"))
    if not zips:
        print(f"[acc] no accident zips in {acc_dir} -> skipping accident covariates")
        return None
    acc = pd.concat([_read_accident_zip(p) for p in zips], ignore_index=True)
    for c in ("LINREFX", "LINREFY"):
        acc[c] = pd.to_numeric(acc[c].str.replace(",", ".", regex=False), errors="coerce")
    acc = acc.dropna(subset=["LINREFX", "LINREFY"])

    minx, miny, maxx, maxy = edges_geom.total_bounds
    acc = acc[acc["LINREFX"].between(minx, maxx) & acc["LINREFY"].between(miny, maxy)]
    years = sorted(acc["UJAHR"].dropna().unique())
    print(f"[acc] {len(acc)} accidents in the study area, years {years[0]}-{years[-1]}")

    g = gpd.GeoDataFrame(
        acc, geometry=gpd.points_from_xy(acc["LINREFX"], acc["LINREFY"]),
        crs="EPSG:25832")
    g["is_bike"] = g["IstRad"].astype(str).str.strip().eq("1")
    g["is_severe"] = g["UKATEGORIE"].astype(str).str.strip().isin(["1", "2"])
    g["year"] = pd.to_numeric(g["UJAHR"], errors="coerce")

    j = gpd.sjoin_nearest(g[["geometry", "is_bike", "is_severe", "year"]],
                          edges_geom[["u_lo", "v_hi", "geometry"]],
                          max_distance=max_dist, how="inner")
    # an accident equidistant from parallel edges is returned once per edge;
    # keep one so a single crash is never counted twice
    j = j[~j.index.duplicated(keep="first")]
    print(f"[acc] {len(j)} snapped to an edge within {max_dist} m "
          f"({len(j) / len(g):.0%} of accidents in area)")

    recent = j["year"] >= ACCIDENT_RECENT_FROM
    j["bike_recent"] = j["is_bike"] & recent
    j["any_recent"] = recent
    print(f"[acc] of these, {int(recent.sum())} are from {ACCIDENT_RECENT_FROM}+ "
          f"(contemporaneous with the senseBox data)")
    return (j.groupby(["u_lo", "v_hi"])
            .agg(n_accidents=("is_bike", "size"), n_acc_bike=("is_bike", "sum"),
                 n_acc_severe=("is_severe", "sum"),
                 n_accidents_recent=("any_recent", "sum"),
                 n_acc_bike_recent=("bike_recent", "sum"))
            .reset_index())


def _main_bearing(geom):
    """Bearing (0-180 deg) of the longest segment of a LineString."""
    coords = list(geom.coords)
    best_len, best = -1.0, 0.0
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        seg = np.hypot(x1 - x0, y1 - y0)
        if seg > best_len:
            best_len, best = seg, np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180
    return best


def aadt_per_edge(edges_geom, path=TRAFFIC_ZIP, max_dist=AADT_SNAP_M,
                  max_angle=AADT_MAX_ANGLE):
    """Motor-traffic AADT (DTVKFZA) and heavy-vehicle AADT (DTVSVA) per street,
    taken from the nearest roughly-parallel counted road segment."""
    if not path.exists():
        print(f"[aadt] {path} not found -> skipping AADT")
        return None
    vw = gpd.read_file(f"zip://{path}!VERKEHRSWERTE_line.shp")
    minx, miny, maxx, maxy = edges_geom.total_bounds
    vw = vw.cx[minx:maxx, miny:maxy].copy()
    vw = vw[vw["DTVKFZA"].notna() & (vw["DTVKFZA"] > 0)]
    vw["vw_bearing"] = vw.geometry.map(_main_bearing)
    print(f"[aadt] {len(vw)} counted road segments in the study area "
          f"(median AADT {vw['DTVKFZA'].median():.0f})")

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
    print(f"[aadt] {len(j)} edges within {max_dist} m of a parallel counted road "
          f"({len(j) / len(e):.1%} of the network)")
    return (j.groupby(["u_lo", "v_hi"])
            .agg(aadt_kfz=("DTVKFZA", "median"), aadt_hgv=("DTVSVA", "median"))
            .reset_index())


def build_covariates():
    edges = gpd.read_file(EDGES_PATH, layer="muenster_edges_classified",
                          ignore_geometry=True)
    keep = ["u", "v", "edge_class", "length", "highway", "maxspeed", "lanes",
            "is_sidepath", "is_cycling_street", "has_track_tag", "has_lane_tag"]
    e = edges[[c for c in keep if c in edges.columns]].copy()
    for c in ("highway", "maxspeed", "lanes"):
        if c in e.columns:
            e[c] = e[c].map(_first)

    e["maxspeed_kmh"] = e["maxspeed"].map(parse_maxspeed)
    hw = e["highway"].astype(str).str.replace("_link", "", regex=False)
    e["maxspeed_kmh"] = e["maxspeed_kmh"].fillna(hw.map(DEFAULT_SPEED))
    e["lanes_n"] = pd.to_numeric(e["lanes"], errors="coerce")

    # one row per directed (u, v); parallel edges collapse (matching cannot
    # distinguish them anyway) taking the shortest
    e = e.sort_values("length")
    agg = {c: "first" for c in
           ["edge_class", "highway", "maxspeed_kmh", "lanes_n", "is_sidepath",
            "is_cycling_street", "has_track_tag", "has_lane_tag"] if c in e.columns}
    agg["length"] = "first"
    cov = e.groupby(["u", "v"], as_index=False).agg(agg).rename(
        columns={"length": "length_m"})

    cov = cov.merge(edge_betweenness(), on=["u", "v"], how="left")
    cov["betweenness"] = cov["betweenness"].fillna(0.0)

    # accidents: undirected (a crash belongs to the street, not to a direction)
    uv = [tuple(sorted((int(a), int(b)))) for a, b in zip(cov["u"], cov["v"])]
    cov["u_lo"] = [a for a, _ in uv]
    cov["v_hi"] = [b for _, b in uv]
    geom = gpd.read_file(EDGES_PATH, layer="muenster_edges_classified",
                         columns=["u", "v"])
    guv = [tuple(sorted((int(a), int(b)))) for a, b in zip(geom["u"], geom["v"])]
    geom["u_lo"] = [a for a, _ in guv]
    geom["v_hi"] = [b for _, b in guv]
    acc = accident_counts(geom)
    if acc is not None:
        cov = cov.merge(acc, on=["u_lo", "v_hi"], how="left")
        for c in ("n_accidents", "n_acc_bike", "n_acc_severe",
                  "n_accidents_recent", "n_acc_bike_recent"):
            cov[c] = cov[c].fillna(0).astype(int)

    # AADT: left NaN where the source has no count (most municipal streets)
    aadt = aadt_per_edge(geom)
    if aadt is not None:
        cov = cov.merge(aadt, on=["u_lo", "v_hi"], how="left")
    return cov


if __name__ == "__main__":
    cov = build_covariates()
    OUT_CSV.parent.mkdir(exist_ok=True)
    cov.to_csv(OUT_CSV, index=False)
    print(f"\n[cov] {len(cov)} directed edges -> {OUT_CSV}")
    print(cov[["length_m", "maxspeed_kmh", "betweenness"]].describe().round(3).to_string())
    print("\nmaxspeed coverage:", f"{cov['maxspeed_kmh'].notna().mean():.0%}")
    if "n_accidents" in cov.columns:
        # counts are a street property replicated to both directions, so total
        # them on unique streets, not on directed edges
        streets = cov.drop_duplicates(subset=["u_lo", "v_hi"])
        print(f"accidents on streets: {streets['n_accidents'].sum()} total, "
              f"{streets['n_acc_bike'].sum()} bicycle-involved, "
              f"{streets['n_acc_severe'].sum()} severe")
        print(f"  of which {ACCIDENT_RECENT_FROM}+: "
              f"{streets['n_accidents_recent'].sum()} total, "
              f"{streets['n_acc_bike_recent'].sum()} bicycle-involved")
        print(f"streets with >=1 bicycle accident: {(streets['n_acc_bike'] > 0).sum()}")
    streets = cov.drop_duplicates(subset=["u_lo", "v_hi"])
    if "aadt_kfz" in cov.columns:
        have = streets[streets["aadt_kfz"].notna()]
        print(f"\nAADT: {len(have)} of {len(streets)} streets have a count "
              f"({len(have) / len(streets):.1%}); median {have['aadt_kfz'].median():.0f} veh/day")
        # is betweenness a usable stand-in where AADT is missing? check where both exist
        if len(have) > 30:
            r = have["betweenness"].corr(have["aadt_kfz"], method="spearman")
            print(f"betweenness vs AADT (Spearman, on the overlap): {r:.2f}"
                  "  <- how well the proxy tracks measured traffic")

    print("\nper regime — mean betweenness, bicycle accidents, median AADT:")
    agg = {"betweenness": ("betweenness", "mean"), "acc_bike": ("n_acc_bike", "sum")}
    if "aadt_kfz" in streets.columns:
        agg["aadt_median"] = ("aadt_kfz", "median")
        agg["with_aadt"] = ("aadt_kfz", lambda s: int(s.notna().sum()))
    print(streets.groupby("edge_class").agg(**agg)
          .sort_values("betweenness", ascending=False).round(4).to_string())
