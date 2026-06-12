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

## Data

OpenStreetMap for the street network and static attributes, openSenseMap
for the bicycle sensor measurements. 

## Structure

```
input/      raw data (gitignored)
output/     results, figures
```

*Work in progress.*