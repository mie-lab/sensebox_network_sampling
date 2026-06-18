# SenseBox Network Sampling

Predicting cycling overtake exposure on street network edges from openSenseMap bicycle sensor data.

## Idea

[senseBox:bike](https://opensensemap.org/) devices record GPS tracks with measured lateral distances of overtaking vehicles. We match these measurements to OSM street network edges, enrich the edges with static traffic and infrastructure attributes, and test how well sparse sensor data can predict overtake exposure across the network — what hurts prediction more, starving the temporal dimension (collection frequency) or the spatial coverage (which edges get sampled)?

## Setup

```bash
conda env create -f environment.yml
conda activate sensebox_network_sampling
```

## Pipeline

```bash
python network_preprocessing.py        # build + classify the street network
python senseboxbike_preprocessing.py   # points -> diagnose -> events
python mapmatch.py                      # match good trajectories to edges
```

### 1. Network

Downloads Münster's cyclable network via OSMnx, projects to EPSG:25832, caches as GraphML. Flags sidepaths (OSM `is_sidepath=yes` or a parallel-to-road geometric test) and assigns each edge one riding-regime class — `bicycle_street`, `roadside_track`, `independent_path`, `painted_lane`, `pedestrian_zone`, `residential_street`, `service_way`, `offroad_track`, `bus_lane`, `minor_road_shared`, `main_road_shared` — by precedence. Writes `input/muenster_edges_classified.gpkg`, an overview map, and audit diagnostics to `output/inspection/`.

Cached — delete `input/muenster_bike.graphml` or call `get_graph(force_download=True)` to refresh.

### 2. Trajectory preprocessing

One pass: load and clip points, segment into trajectories on temporal gaps, drop unusable trajectories (via the diagnostics functions below), then build overtake events from the survivors. Events collapse ~1 Hz nonzero bursts into one row per pass, each carrying its closest-approach point for later edge placement. Writes `trajectory_points.gpkg`, `overtake_events.gpkg`, `trajectory_summary.csv`.

### 3. Map-matching

Matches each trajectory to the network with Leuven (HMM/Viterbi); the ordered path resolves parallel edges (road vs sidepath). Each event inherits the edge of its closest-approach point, joined to `edge_class`. Writes `matched_points.gpkg`, `matched_events.gpkg` — the event↔edge link feeds the exposure modelling.

### Diagnostics

`trajectory_diagnostics.py` holds the trajectory quality logic used by stage 2 — flagging sensors that are stuck, near-stationary, or firing implausibly often (>250 detections/km) — and writes a quality table plus per-box overview and small-multiple plots to `output/diagnostics/`. Run standalone for the plots; stage 2 imports its functions for filtering.

## Data

OpenStreetMap for the street network and static attributes, openSenseMap for the bicycle sensor measurements. Raw data lives in `input/` and is not tracked in the repo.

## Related work

Network semantics informed by the IIP bikeability pipeline ([OSMBicycleInfrastructure](https://github.com/niebl/OSMBicycleInfrastructure), [IP-OSeM-Backend](https://github.com/Rajasirpi/IP-OSeM-Backend)), reimplemented in Python on a complete cyclable network.

*Work in progress.*