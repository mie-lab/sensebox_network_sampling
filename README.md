# SenseBox Network Sampling

[senseBox:bike](https://opensensemap.org/) devices record a cyclist's GPS track together
with the lateral distance of vehicles overtaking them. This repo turns those rides into an
overtake-risk estimate for every street in a city, and then asks how such rides should be
collected in the first place.

The unit of analysis is a directed street edge, described by **N**, the overtakes recorded
on it, and **E**, its exposure in rider-hours. The rate **N/E** is what we estimate.

Coverage is the difficulty: most edges have never been ridden, and most of the ridden ones
recorded no overtake at all. So the modelling question is **whose data an edge should
borrow when it has too little of its own**. We build a risk model on that basis, then test
sampling strategies by thinning the collected rides and comparing the estimates against
the full-data baseline.

## Setup

```bash
conda env create -f environment.yml
conda activate sensebox_network_sampling
```

## Pipeline

Each stage reads the previous one's output, and everything a stage writes is prefixed with its task number. If the step is expensive, it checks for the output file first and skips if it exists.

### Task 0: Data acquisition

```bash
python task0_download.py
```

Downloads both overtaking channels from the [openSenseMap](https://opensensemap.org/) API: the measured `overtake distance`, and the classifier's `overtake maneuver` probability.

*Saves in `input/`:* `muenster_overtaking_distance_*.csv`, `muenster_overtaking_manoeuvre_*.csv`

### Task 1: Network and street regimes

```bash
python task1_network.py
```

Downloads the cyclable network from [OpenStreetMap](https://www.openstreetmap.org/) and classifies every edge into 11 riding regimes (bicycle street, roadside track, main road shared, ...), then attaches the static covariates each edge carries. Both orientations of every street are present. A cyclist can ride either way, and contraflow on one-ways is common, so the two directions are counted separately.

*Saves in `input/`:* `muenster_edges_classified.gpkg` (every edge with its riding regime), `muenster_edge_covariates.csv` (speed limit, lanes, betweenness, accidents, AADT)

### Task 2: Overtake data extraction

```bash
python task2a_ride_quality.py
```

Segments the raw points into rides and filters out the unusable ones: stuck sensors repeating a value, implausible speeds, rides too short to measure.

*Saves in `output/task2_diagnostics/`:* `task2a_segmented_points.gpkg` (all points, with a ride id), `task2a_trajectory_quality.csv` (one row per ride, with the keep/drop verdict and why)

```bash
python task2b_overtake_events.py
```

Extracts overtake events from the rides that passed.

*Saves in `output/task2_trajectories/`:* `task2b_overtake_events.gpkg` (one row per overtake, with time, position and clearance), `task2b_trajectory_points.gpkg` (points of the kept rides, the input to map-matching), `task2b_trajectory_summary.csv` (one row per ride: length, duration, overtake count)

### Task 3: Map-matching

```bash
python task3_mapmatching.py
```

Matches each ride to an ordered path of network edges, preserving travel direction, and pins every overtake onto the edge it happened on.
  
*Saves in `output/task3_matching/`:* `task3_matched_points.gpkg` (every ride point with its matched edge), `task3_matched_events.gpkg` (every overtake with its edge and street type), `task3_match_summary.csv` (how much matched, how many events linked, direction checks)

### Task 4: Edge oracle

```bash
python task4_oracle.py
```

Aggregates everything to one row per directed edge. This is the interface between the pipeline so far and the risk exposure modelling, and it covers the ridden network. It also writes the descriptive figures.
  
*Saves in `output/task4_oracle/`:* `task4_edge_oracle.csv` is the unit of analysis: **one directed edge `(u, v)` per row** (exposure, overtake count and covariates per edge), `task4_edge_traversals.csv` (one row per edge × ride), `task4_edge_events.csv` (one row per overtake), and `task4_*.png` (descriptives like coverage, revisits, rate by predictor)

### Task 5: Overtake risk model

```bash
python task5a_diagnostics.py
```

Produces the evidence behind the model choices: which exposure unit fits, how overdispersed the counts are, how much a few riders dominate, whether rates drift over time, and whether spatial structure is left over after pooling by street type.
  
*Saves in `output/task5_diagnostics/`:* `task5a_*.csv` and `task5a_*.png`

```bash
python task5b_risk_gamma.py
```

Estimates the Poisson-Gamma risk model. It fits an overtake rate for every edge, shrinking each edge toward its street type.
  
*Saves in `output/task5_risk/`:* `task5b_edge_risk.csv` (per-edge rate, 95% credible interval, shrink weight), `task5b_cv_performance.csv` (held-out accuracy against the simpler baselines: regime average and single-edge rate)

```bash
python task5c_risk_gp.py
```

Estimates the Gaussian-Process risk model: the same model with a spatial kernel integrated, so an edge borrows from its geographic neighbours as well as from its street type.
  
*Saves in `output/task5_risk/`:* `task5c_gp_cv.csv` (held-out accuracy against Poisson-Gamma)

### Task 6: Sampling strategies (TODO)


## Related work

Network semantics informed by the IIP bikeability pipeline
([OSMBicycleInfrastructure](https://github.com/niebl/OSMBicycleInfrastructure),
[IP-OSeM-Backend](https://github.com/Rajasirpi/IP-OSeM-Backend)), reimplemented in Python
on a complete cyclable network.

*Work in progress.*
