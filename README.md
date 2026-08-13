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

## Project structure

One script per pipeline stage, all run from the repo root, each rerunnable on its own.
Expensive results are cached as files: delete the file to recompute it.

```
task0_download.py          
task1_network.py           
task2a_ride_quality.py      
task2b_overtake_events.py   
task3_mapmatching.py       
task4_oracle.py             
task5a_diagnostics.py       
task5b_risk_gamma.py       
task5c_risk_gp.py           

input/
  muenster_overtaking_*.csv         openSenseMap channels
  muenster_bike.graphml             OSM network
  muenster_edges_classified.gpkg    generated in task 1
  muenster_edge_covariates.csv      generated in task 1
  accidents/, traffic/              Unfallatlas and Straßen.NRW downloads

output/
  task1_network/            classification audit and its figures
  task2_rides/              the rides, their quality verdicts and figures
  task2_overtakes/          the overtake events and their figures
  task3_matching/           matched points, and where matching failed
  task4_oracle/             the per-edge tables the models read, and the descriptives
  task5_diagnostics/        model-choice evidence
  task5_risk/               fitted risk estimates and CV scores

environment.yml             conda environment
```

## Pipeline

Each stage reads the previous one's output, and everything a stage writes is prefixed with its task number. If the step is expensive, it checks for the output file first and skips if it exists.

### Task 0: Data acquisition
Downloads both overtaking channels from the [openSenseMap](https://opensensemap.org/) API: the measured `overtake distance`, and the classifier's `overtake maneuver` probability.

```bash
python task0_download.py
```

*→ `input/muenster_overtaking_{distance,manoeuvre}_*.csv`*

### Task 1: Network and street regimes
Downloads the cyclable network from [OpenStreetMap](https://www.openstreetmap.org/), classifies every edge into 13 riding regimes (bicycle street, roadside track, major road shared, ...), and attaches its static covariates. Both orientations of every street are kept, since contraflow riding is common.

```bash
python task1_network.py
```

*→ `input/muenster_edges_classified.gpkg`, `input/muenster_edge_covariates.csv`, and the classification audit in `output/task1_network/`*

### Task 2: Overtake data extraction
Task 2a cuts the raw points into rides and judges each against four rules: blocked distance sensor, missing classifier channel, non-cycling speed, too short to measure. Nothing is deleted; every ride and every point carries its verdict.

```bash
python task2a_ride_quality.py
```

*→ `output/task2_rides/`: the points, one row per ride, and what each threshold is worth*

Task 2b collapses bursts of high classifier probability into one event per car pass.

```bash
python task2b_overtake_events.py
```

*→ `output/task2_overtakes/`: one row per overtake, one per ride, and the event-count sweeps*

### Task 3: Map-matching
Matches each ride to an ordered path of network edges in travel order, on geometry alone. Attributes are joined later, in task 4.

```bash
python task3_mapmatching.py
```

*→ `output/task3_matching/`: every point with its matched edge, and where matching failed*

### Task 4: Edge oracle
Aggregates everything to one row per directed edge. This is the interface between the pipeline so far and the risk modelling.

```bash
python task4_oracle.py
```

*→ `output/task4_oracle/`: `task4_edge_oracle.csv` is the unit of analysis, **one directed edge `(u, v)` per row**, alongside one row per edge × ride, one per overtake, and the coverage descriptives*

### Task 5: Overtake risk model
Task 5a is the evidence behind the model choices: exposure unit, count distribution, rider dominance, temporal drift, which covariates survive street type, and what spatial structure is left.

```bash
python task5a_diagnostics.py
```

*→ `output/task5_diagnostics/`: one CSV and one figure per diagnostic*

Task 5b fits the Poisson-Gamma model, shrinking each edge's rate toward its street type. Task 5c adds a spatial kernel, so an edge borrows from its neighbours as well.

```bash
python task5b_risk_gamma.py
python task5c_risk_gp.py
```

*→ `output/task5_risk/`: per-edge rates with credible intervals, and held-out scores against the simpler baselines*

### Task 6: Sampling strategies (TODO)


## Related work

Network semantics informed by the IIP bikeability pipeline
([OSMBicycleInfrastructure](https://github.com/niebl/OSMBicycleInfrastructure),
[IP-OSeM-Backend](https://github.com/Rajasirpi/IP-OSeM-Backend)), reimplemented in Python
on a complete cyclable network.

*Work in progress.*
