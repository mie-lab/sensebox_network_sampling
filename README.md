# SenseBox Network Sampling

Predicting cycling overtake exposure on street network edges from
openSenseMap bicycle sensor data.

## Idea

[senseBox:bike](https://opensensemap.org/) devices record GPS tracks with
measured lateral distances of overtaking vehicles. We match these
measurements to OSM street network edges, enrich the edges with static
traffic and infrastructure attributes, and test how well sparse sensor
data can predict overtake exposure across the network — what hurts
prediction more, starving the temporal dimension (collection frequency)
or the spatial coverage (which edges get sampled)?

## Setup

```bash
conda env create -f environment.yml
conda activate sensebox_network_sampling
```

## Network preprocessing

`senseboxbike_preprocessing.py` builds the analysis network for Münster:

1. **Acquisition** — downloads the cyclable street network
   (`network_type="bike"`: all ways cyclists may legally use, not just
   bike infrastructure) via OSMnx, topologically simplified, projected
   to EPSG:25832, cached as GraphML in `input/`.
2. **Sidepath detection** — Münster's cycle infrastructure is largely
   sidewalk-level tracks alongside roads. Car-free edges are flagged as
   sidepaths via the OSM tag `is_sidepath=yes` combined with a geometric
   check: within 12 m of a motor-traffic road unconditionally, within
   12–30 m only if roughly parallel (bearing of longest segment within
   25°).
3. **Classification** — each edge gets one riding-regime class
   (precedence-ordered, most specific first):

   | class | meaning |
   |---|---|
   | `bicycle_street` | Fahrradstraße, cars are guests |
   | `roadside_track` | separated track along a road (red tiles etc.) |
   | `independent_path` | cycleway/path away from motor traffic |
   | `painted_lane` | on-carriageway cycle lane |
   | `pedestrian_zone` | riding among pedestrians |
   | `residential_street` | low-speed residential streets |
   | `service_way` | access/parking/service ways |
   | `offroad_track` | unpaved field/forest tracks |
   | `bus_lane` | bus and bikes share the carriageway |
   | `minor_road_shared` | mixed traffic, minor connector roads |
   | `main_road_shared` | mixed traffic on an arterial road |

   OSM tag booleans (`has_lane_tag`, `is_cycling_street`, `is_sidepath`,
   …) are kept as separate columns for modelling.
4. **Audit & outputs** — diagnostics (highway-tag crosstab, precedence
   audit, length-weighted class shares, small-multiple maps) are written
   to `output/inspection/`; the classified edges to
   `input/muenster_edges_classified.gpkg`; an overview map to
   `output/network_overview.png`.

Run with:

```bash
python network_preprocessing.py
```

The OSM download is cached — delete `input/muenster_bike.graphml` or call
`get_graph(force_download=True)` to refresh.

## Data

OpenStreetMap for the street network and static attributes, openSenseMap
for the bicycle sensor measurements. Raw data lives in `input/` and is
not tracked in the repo.

## Related work

Network semantics informed by the IIP bikeability pipeline
([OSMBicycleInfrastructure](https://github.com/niebl/OSMBicycleInfrastructure),
[IP-OSeM-Backend](https://github.com/Rajasirpi/IP-OSeM-Backend)),
reimplemented in Python on a complete cyclable network.

*Work in progress.*