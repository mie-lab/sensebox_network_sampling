"""Inspect the overtake data before modelling — what does the risk model face?

Unit = undirected street (both travel directions summed). We look at the count
distribution, the exposure, how count scales with exposure, the spread per
regime, and the spatial spread. These decide the model: a heavily zero, over-
dispersed count with an exposure offset -> Poisson-Gamma; residual spatial
structure -> a spatial term.

Run:  python overtake_inspection_task5.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORACLE = Path("output/task4_oracle/edge_oracle_task4.csv")
FIG = Path("output/figures")
INK, MUTED, ACC, ACC2 = "#0b0b0b", "#52514e", "#2a78d6", "#e34948"
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def load_streets():
    o = pd.read_csv(ORACLE)
    g = (o.groupby(["u_lo", "v_hi"])
         .agg(n_events=("n_events", "sum"), rider_km=("rider_km", "sum"),
              n_traversals=("n_traversals", "sum"),
              edge_class=("edge_class", "first"))
         .reset_index())
    g["is_obs"] = g["rider_km"] > 0
    return g


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    g = load_streets()
    obs = g[g["is_obs"]].copy()
    obs["rate"] = obs["n_events"] / obs["rider_km"]

    print(f"streets: {len(g)} total | {len(obs)} observed ({len(obs)/len(g):.0%})")
    print(f"never ridden: {(~g['is_obs']).sum()}")

    # --- count distribution ---
    vc = obs["n_events"].value_counts().sort_index()
    print("\ncount distribution (observed streets):")
    for k in [0, 1, 2, 3, 4, 5]:
        n = int(vc.get(k, 0))
        print(f"  N = {k}: {n:5d}  ({n/len(obs):5.1%})")
    print(f"  N >= 6: {int((obs['n_events'] >= 6).sum())}  "
          f"({(obs['n_events'] >= 6).mean():.1%})")
    print(f"  mean count {obs['n_events'].mean():.3f} | var {obs['n_events'].var():.3f} "
          f"| var/mean {obs['n_events'].var()/obs['n_events'].mean():.2f}")

    # --- exposure ---
    print(f"\nexposure per observed street (rider-km): "
          f"median {obs['rider_km'].median():.2f}, "
          f"p10 {obs['rider_km'].quantile(.1):.2f}, p90 {obs['rider_km'].quantile(.9):.2f}")
    print(f"traversals per street: median {int(obs['n_traversals'].median())}, "
          f"max {int(obs['n_traversals'].max())}")
    for t in [1, 2, 3, 5, 10]:
        print(f"  streets with >= {t:2d} traversals: {(obs['n_traversals'] >= t).sum():5d}")

    # --- events vs exposure (should scale ~linearly for a rate model) ---
    obs["km_bin"] = pd.qcut(obs["rider_km"], 6, duplicates="drop")
    eb = obs.groupby("km_bin", observed=True).agg(
        n=("u_lo", "size"), events=("n_events", "sum"), km=("rider_km", "sum"))
    eb["rate"] = eb["events"] / eb["km"]
    print("\nrate by exposure sextile (flat rate => exposure is a clean offset):")
    print(eb[["n", "events", "km", "rate"]].round(2).to_string())

    # --- figure ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].bar(vc.index[:8], vc.values[:8], color=ACC, alpha=0.85)
    ax[0].set_xlabel("overtakes on a street  N")
    ax[0].set_ylabel("streets")
    ax[0].set_title(f"Counts are mostly zero\n{(obs['n_events']==0).mean():.0%} of ridden "
                    f"streets record none", fontsize=11, loc="left")

    ax[1].scatter(obs["rider_km"], obs["n_events"], s=6, alpha=0.25, color=ACC,
                  edgecolors="none")
    ax[1].set_xlim(0, obs["rider_km"].quantile(0.98))
    ax[1].set_ylim(0, obs["n_events"].quantile(0.995) + 1)
    ax[1].set_xlabel("exposure  (rider-km)")
    ax[1].set_ylabel("overtakes  N")
    ax[1].set_title("Count grows with exposure\n-> exposure is the offset", fontsize=11, loc="left")

    reg = (obs.groupby("edge_class").agg(events=("n_events", "sum"),
           km=("rider_km", "sum")).assign(rate=lambda d: d.events / d.km)
           .sort_values("rate"))
    ax[2].barh(range(len(reg)), reg["rate"], color=ACC2, alpha=0.8)
    ax[2].set_yticks(range(len(reg)))
    ax[2].set_yticklabels([r.replace("_", " ") for r in reg.index], fontsize=8)
    ax[2].set_xlabel("overtake rate  (per rider-km)")
    ax[2].set_title("Rate varies ~10x by street type\n-> regime is the main structure",
                    fontsize=11, loc="left")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_overtake_inspection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {FIG / 'fig_overtake_inspection.png'}")


if __name__ == "__main__":
    main()
