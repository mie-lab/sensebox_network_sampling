"""The evidence behind the task 5 model choices.

Every function answers "why is the estimator built this way?". Exposure is
rider-hours throughout, which is the choice exposure_unit_choice() makes.

Writes to output/task5_diagnostics/:
  task5a_exposure_unit_choice.csv     why rider-hours and not traversals or rider-km
  task5a_exposure_unit_by_regime.csv  speed and relative rate per regime
  task5a_poisson_vs_nb.csv            does the negative binomial beat Poisson
  task5a_regime_dispersion.csv        overdispersion per street type
  task5a_covariate_adjustment.csv     covariate effects, marginal against regime-adjusted
  task5a_covariate_correlation.csv    rate against the static covariates
  task5a_moran_threshold.csv          residual spatial autocorrelation as coverage improves
  task5a_rider_dominance.csv          how few boxes carry the data, and what that costs
  task5a_temporal_drift.csv           per-regime rate ratio between the two windows
  task5a_*.png                        one figure per diagnostic
"""
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import spearmanr

from task4_oracle import (EVENTS_CSV, MIN_EVENTS, ORACLE_CSV, TRAVERSALS_CSV,
                          poisson_ci)

OUT_DIR = Path("output/task5_diagnostics")
EXPOSURE_UNIT_CSV = OUT_DIR / "task5a_exposure_unit_choice.csv"
EXPOSURE_BY_REGIME_CSV = OUT_DIR / "task5a_exposure_unit_by_regime.csv"
RIDER_DOMINANCE_CSV = OUT_DIR / "task5a_rider_dominance.csv"
REGIME_DISPERSION_CSV = OUT_DIR / "task5a_regime_dispersion.csv"
TEMPORAL_DRIFT_CSV = OUT_DIR / "task5a_temporal_drift.csv"
POISSON_VS_NB_CSV = OUT_DIR / "task5a_poisson_vs_nb.csv"
COVARIATE_ADJUSTMENT_CSV = OUT_DIR / "task5a_covariate_adjustment.csv"
COVARIATE_CORRELATION_CSV = OUT_DIR / "task5a_covariate_correlation.csv"
MORAN_CSV = OUT_DIR / "task5a_moran_threshold.csv"

EXPOSURE_UNIT_FIG = OUT_DIR / "task5a_exposure_unit_choice.png"
RIDER_DOMINANCE_FIG = OUT_DIR / "task5a_rider_dominance.png"
REGIME_DISPERSION_FIG = OUT_DIR / "task5a_regime_dispersion.png"
TEMPORAL_DRIFT_FIG = OUT_DIR / "task5a_temporal_drift.png"
COVARIATE_ADJUSTMENT_FIG = OUT_DIR / "task5a_covariate_adjustment.png"
COVARIATE_CORRELATION_FIG = OUT_DIR / "task5a_covariate_correlation.png"

SEED = 42   # one generator per resampling diagnostic, so main() can be reordered safely


# ========= diagnostics =========


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[csv] saved -> {path}")


def _load_rides(oracle):
    """Events and traversals, both carrying edge_class and rider-hours."""
    ev = pd.read_csv(EVENTS_CSV)
    tr = pd.read_csv(TRAVERSALS_CSV)
    tr = tr.merge(oracle[["u", "v", "edge_class"]], on=["u", "v"], how="left")
    tr["rider_h"] = tr["on_edge_s"] / 3600.0
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    return ev, tr


def _nb_alpha(y, X, offset, grid=np.geomspace(0.05, 10, 60)):
    """NB dispersion by profile likelihood, since a GLM at fixed alpha always converges
    where the joint NB2 optimiser does not."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lls = [sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=a),
                      offset=offset).fit().llf for a in grid]
    return float(grid[int(np.argmax(lls))])


def _regime_design(d):
    """Intercept plus one column per regime, one dropped as the reference level."""
    return sm.add_constant(pd.get_dummies(d["edge_class"], drop_first=True,
                                          dtype=float))


def _dispersion(events, hours):
    """Count variance against the Poisson variance at that exposure. 1 is Poisson-like."""
    expected = (events.sum() / hours.sum() * hours).clip(lower=1e-9)
    return ((events - expected) ** 2 / expected).sum() / (len(events) - 1)


def exposure_unit_choice(oracle, n_bins=10):
    """Fits the same Poisson rate model under each of the three exposure offsets and scores
    them: AIC on a common response, the dispersion each leaves, and the slope of log(E)
    added on top of its own offset."""
    units = {"per traversal": "n_traversals", "per rider-km": "rider_km",
             "per rider-hour": "rider_h"}
    d = oracle[oracle["is_observed"] & (oracle["rider_km"] > 0) & (oracle["rider_h"] > 0)]
    total_events = d["n_events"].sum()

    by_regime = d.groupby("edge_class").agg(events=("n_events", "sum"),
                                            km=("rider_km", "sum"), hr=("rider_h", "sum"))
    by_regime = by_regime[by_regime["events"] >= MIN_EVENTS].copy()
    by_regime["speed_kmh"] = by_regime["km"] / by_regime["hr"]
    by_regime["rel_km"] = ((by_regime["events"] / by_regime["km"]) / (total_events / d["rider_km"].sum()))
    by_regime["rel_hr"] = ((by_regime["events"] / by_regime["hr"]) / (total_events / d["rider_h"].sum()))
    by_regime = by_regime.sort_values("speed_kmh")
    rho_km = float(spearmanr(by_regime["speed_kmh"], by_regime["rel_km"]).statistic)
    rho_hr = float(spearmanr(by_regime["speed_kmh"], by_regime["rel_hr"]).statistic)

    design = _regime_design(d)
    regimes = design.drop(columns="const")
    rows, curves = [], {}
    for unit, exposure in units.items():
        log_exposure = np.log(d[exposure])
        fit = sm.GLM(d["n_events"], design, family=sm.families.Poisson(),
                     offset=log_exposure).fit()
        with_scale = sm.add_constant(pd.concat([regimes, log_exposure.rename("logE")], axis=1))
        scaled = sm.GLM(d["n_events"], with_scale, family=sm.families.Poisson(),
                        offset=log_exposure).fit()
        deciles = (d.groupby(pd.qcut(d[exposure], n_bins, duplicates="drop"), observed=True)
                   .agg(events=("n_events", "sum"), exposure=(exposure, "sum")))
        curves[unit] = ((deciles["events"] / deciles["exposure"]).to_numpy()
                        / (total_events / d[exposure].sum()))
        rows.append(dict(unit=unit, AIC=round(fit.aic, 1),
                         dispersion=round(fit.pearson_chi2 / fit.df_resid, 2),
                         scale_slope=round(float(scaled.params["logE"]), 3),
                         slope_p=round(float(scaled.pvalues["logE"]), 4)))

    summary = pd.DataFrame(rows)
    summary["dAIC"] = (summary["AIC"] - summary["AIC"].min()).round(1)
    _write(summary, EXPOSURE_UNIT_CSV)
    _write(by_regime.round(3).reset_index(), EXPOSURE_BY_REGIME_CSV)
    fig_exposure_unit_choice(curves, by_regime, rho_km)

    print(summary.to_string(index=False))
    print(f"speed {by_regime['speed_kmh'].min():.1f}-{by_regime['speed_kmh'].max():.1f} km/h, "
          f"rank corr with the relative rate: per-km {rho_km:+.2f}, per-hour {rho_hr:+.2f}")
    return summary


def poisson_vs_nb(oracle):
    """Poisson against negative binomial on the regime-only rate model. NB winning
    licenses the Gamma prior in task5b, which is a negative binomial once integrated."""
    d = oracle[oracle["is_observed"] & (oracle["rider_h"] > 0)]
    y, X, offset = d["n_events"], _regime_design(d), np.log(d["rider_h"])

    poisson = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    alpha = _nb_alpha(y, X, offset)
    nb = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()
    k = int(poisson.df_model + 1)
    aic_poisson, aic_nb = -2 * poisson.llf + 2 * k, -2 * nb.llf + 2 * (k + 1)

    df = pd.DataFrame({
        "metric": ["Poisson loglik", "Poisson parameters", "Poisson AIC",
                   "NB loglik", "NB parameters", "NB AIC", "dAIC (Poisson - NB)",
                   "NB dispersion alpha", "Poisson Pearson dispersion"],
        "value": [round(poisson.llf, 1), k, round(aic_poisson, 1),
                  round(nb.llf, 1), k + 1, round(aic_nb, 1),
                  round(aic_poisson - aic_nb, 1), round(alpha, 3),
                  round(poisson.pearson_chi2 / poisson.df_resid, 2)],
    })
    _write(df, POISSON_VS_NB_CSV)
    print(df.to_string(index=False))
    return df


def regime_dispersion(oracle):
    """Overdispersion per street type, to be read against the pooled value."""
    d = oracle[oracle["is_observed"] & (oracle["rider_h"] > 0)]

    df = pd.DataFrame(
        [dict(regime=regime, edges=len(g), events=int(g["n_events"].sum()),
              rate_per_hour=round(g["n_events"].sum() / g["rider_h"].sum(), 2),
              dispersion=round(_dispersion(g["n_events"], g["rider_h"]), 2))
         for regime, g in d.groupby("edge_class")
         if g["n_events"].sum() >= MIN_EVENTS]).sort_values("dispersion", ascending=False)
    pooled = _dispersion(d["n_events"], d["rider_h"])

    _write(df, REGIME_DISPERSION_CSV)
    fig_regime_dispersion(df, pooled)
    print(df.to_string(index=False))
    print(f"  pooled dispersion = {pooled:.2f}")
    return df


def covariate_adjustment(oracle):
    """Each covariate's rate ratio, marginal and adjusted for regime, from a negative binomial."""
    d = oracle[oracle["is_observed"] & (oracle["rider_h"] > 0)].dropna(subset=["maxspeed_kmh"]).copy()
    d["logE"] = np.log(d["rider_h"])
    d["acc_any"] = (d["n_acc_bike"] > 0).astype(int)
    d["sp10"] = d["maxspeed_kmh"] / 10.0
    lb = np.log(d["betweenness"].clip(lower=1e-7))
    d["lbet_z"] = (lb - lb.mean()) / lb.std()
    terms = {"acc_any": ">=1 bike accident", "sp10": "speed +10 km/h",
             "lbet_z": "betweenness +1 SD"}

    design = _regime_design(d)
    alpha = _nb_alpha(d["n_events"], design, d["logE"])
    regimes = design.drop(columns="const")

    def rr(X, name):
        fit = sm.GLM(d["n_events"], sm.add_constant(X.astype(float)),
                     family=sm.families.NegativeBinomial(alpha=alpha),
                     offset=d["logE"]).fit()
        lo, hi = np.exp(fit.conf_int().loc[name])
        return float(np.exp(fit.params[name])), float(lo), float(hi), float(fit.pvalues[name])

    rows = []
    for term, label in terms.items():
        marginal = rr(d[[term]], term)
        adjusted = rr(pd.concat([regimes, d[[term]]], axis=1), term)
        rows.append(dict(covariate=label, rr_marginal=marginal[0], lo_m=marginal[1],
                         hi_m=marginal[2], rr_adjusted=adjusted[0], lo_a=adjusted[1],
                         hi_a=adjusted[2], p_adjusted=adjusted[3]))

    df = pd.DataFrame(rows)
    _write(df.round(3), COVARIATE_ADJUSTMENT_CSV)
    fig_covariate_adjustment(df)
    print(df.round(3).to_string(index=False))
    return df


def covariate_correlation(oracle, min_trav=3):
    """Spearman correlations between the per-edge rate and the static covariates, on edges
    with at least min_trav traversals, below which the rate is mostly noise."""
    d = oracle[oracle["is_observed"] & (oracle["n_traversals"] >= min_trav) & (oracle["rider_h"] > 0)].copy()
    d["overtake_rate"] = d["n_events"] / d["rider_h"]
    d["speed_kmh"] = d["rider_km"] / d["rider_h"]

    display = {
        "overtake_rate": "overtake rate", "n_traversals": "traversals",
        "rider_km": "rider-km", "rider_h": "rider-hours", "speed_kmh": "rider speed",
        "length_m": "edge length", "n_acc_bike": "bike accidents",
        "maxspeed_kmh": "speed limit", "betweenness": "betweenness",
        "lanes_n": "car lanes", "aadt_kfz": "AADT",
    }
    names = {k: v for k, v in display.items() if k in d.columns}
    sub = d[list(names)].rename(columns=names)
    corr = sub.corr(method="spearman")
    _write(corr.round(3).reset_index(names="var"), COVARIATE_CORRELATION_CSV)

    coverage = sub.notna().mean()
    fig_covariate_correlation(corr, coverage, len(d), min_trav)
    return corr


def _moran(streets, col, n_perm=999, rng=None):
    """Moran's I on binary adjacency: two streets neighbour if they share an endpoint node.
    Streets left without a neighbour stay in, which pulls I toward 0 by at most 0.004 here.
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    streets = streets.reset_index(drop=True)

    at_node = {}
    for i, (u, v) in enumerate(zip(streets["u_lo"], streets["v_hi"])):
        at_node.setdefault(u, []).append(i)
        at_node.setdefault(v, []).append(i)

    neighbours = {(min(i, j), max(i, j)) for shared in at_node.values() if len(shared) > 1
                  for i, j in combinations(shared, 2)}
    if not neighbours:
        return None
    left = np.fromiter((i for i, _ in neighbours), dtype=int)
    right = np.fromiter((j for _, j in neighbours), dtype=int)

    deviation = streets[col].to_numpy(float)
    deviation = deviation - deviation.mean()
    total_variance = deviation @ deviation
    if total_variance == 0:
        return None

    n_streets, n_pairs = len(deviation), len(neighbours)
    scale = n_streets / (n_pairs * total_variance)
    observed = scale * np.sum(deviation[left] * deviation[right])

    null = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = rng.permutation(deviation)
        null[k] = scale * np.sum(shuffled[left] * shuffled[right])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)

    return {"streets": n_streets, "pairs": n_pairs,
            "mean_neighbors": 2 * n_pairs / n_streets,
            "morans_I": observed, "perm_p": p}


def moran_by_threshold(oracle, thresholds=(1, 2, 4, 6, 8, 10)):
    """Moran's I of the street-type Pearson residuals, as coverage improves."""
    # pooled to undirected streets, or a street's two directions neighbour each other
    streets = (oracle[oracle["rider_h"] > 0]
               .groupby(["u_lo", "v_hi"], as_index=False)
               .agg(n_events=("n_events", "sum"), rider_h=("rider_h", "sum"),
                    n_trav=("n_traversals", "sum"), edge_class=("edge_class", "first")))

    rate = streets.groupby("edge_class")["n_events"].sum() / \
        streets.groupby("edge_class")["rider_h"].sum()
    expected = (streets["edge_class"].map(rate) * streets["rider_h"]).clip(lower=1e-9)

    # the residual, not the raw rate
    streets["pearson"] = (streets["n_events"] - expected) / np.sqrt(expected)
    rng = np.random.default_rng(SEED)
    rows = [{"min_traversals": t, **_moran(streets[streets["n_trav"] >= t], "pearson", rng=rng)}
            for t in thresholds]
    df = pd.DataFrame(rows).round(3)
    _write(df, MORAN_CSV)
    print(df.to_string(index=False))
    return df


# ========= diagnosed but unmodelled =========


def rider_dominance(ev, tr, n_boot=3000):
    """Concentration of the data across boxes, and the overall rate under two intervals:
    a naive Poisson one and a bootstrap that resamples whole boxes."""
    tr = tr.assign(rider_km=tr["length_m"] / 1000.0)
    per = tr.groupby("boxId").agg(hr=("rider_h", "sum"), km=("rider_km", "sum"))
    per["ev"] = ev.groupby("boxId").size().reindex(per.index).fillna(0).astype(int)
    per = per[per["hr"] > 0]

    events, hours = per["ev"].to_numpy(), per["hr"].to_numpy()
    n_events, n_hours = events.sum(), hours.sum()
    _, naive_lo, naive_hi = poisson_ci(n_events, n_hours)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(per), size=(n_boot, len(per)))   # whole boxes, with replacement
    boot = events[draws].sum(axis=1) / hours[draws].sum(axis=1)
    clustered_lo, clustered_hi = np.percentile(boot, [2.5, 97.5])

    stats = {
        "boxes with exposure": len(per),
        "events": int(n_events),
        "rider-hours": round(n_hours, 1),
        "rate per hour": round(n_events / n_hours, 2),
        "rate per km": round(n_events / per["km"].sum(), 3),
        "top-3 share of events": round(np.sort(events)[-3:].sum() / n_events, 3),
        "top-3 share of hours": round(np.sort(hours)[-3:].sum() / n_hours, 3),
        "effective sample size (Kish)": round(n_events ** 2 / (events ** 2).sum(), 1),
        "naive 95% lo": round(naive_lo, 2),
        "naive 95% hi": round(naive_hi, 2),
        "clustered 95% lo": round(clustered_lo, 2),
        "clustered 95% hi": round(clustered_hi, 2),
        "interval inflation": round((clustered_hi - clustered_lo) / (naive_hi - naive_lo), 2),
    }
    summary = pd.DataFrame(stats.items(), columns=["metric", "value"])
    _write(summary, RIDER_DOMINANCE_CSV)
    fig_rider_dominance(stats)
    print(summary.to_string(index=False))
    return summary


def _regime_temporal_drift(ev, tr, min_events=10):
    """Per-regime rate ratio between the windows, from a Poisson GLM on regime x window
    x box cells. SEs are clustered by box, since one box's rides are alike."""

    keys = ["edge_class", "win", "boxId"]
    cells = pd.concat([tr.groupby(keys, observed=True)["rider_h"].sum().rename("hr"),
                       ev.groupby(keys, observed=True).size().rename("n")],
                      axis=1).reset_index()
    cells = cells[cells["hr"] > 0].dropna(subset=["win"])   # NaN hours fail this too
    cells = cells.assign(n=cells["n"].fillna(0), win=cells["win"].astype(float),
                         loghr=np.log(cells["hr"]))

    def ratio(g):
        if g["n"].sum() < min_events or g["win"].nunique() < 2:
            return dict(rr=np.nan, lo=np.nan, hi=np.nan, p=np.nan, drift=False)
        model = sm.GLM(g["n"], sm.add_constant(g[["win"]]),
                       family=sm.families.Poisson(), offset=g["loghr"])
        try:
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": g["boxId"]})
        except Exception:
            fit = model.fit()                       # too few boxes to cluster
        lo, hi = np.exp(fit.conf_int().loc["win"])
        return dict(rr=float(np.exp(fit.params["win"])), lo=float(lo), hi=float(hi),
                    p=float(fit.pvalues["win"]), drift=bool(lo > 1 or hi < 1))

    return pd.DataFrame([dict(regime=r, **ratio(g))
                         for r, g in cells.groupby("edge_class")])


def temporal_drift(ev, tr, n_windows=2, min_events=10):
    """Per-regime rate ratio between the two halves of the study period."""
    edges = list(pd.date_range(tr["enter_time"].min(), tr["enter_time"].max(),
                               periods=n_windows + 1))
    edges[0] -= pd.Timedelta(seconds=1)
    edges[-1] += pd.Timedelta(seconds=1)
    tr = tr.assign(win=pd.cut(tr["enter_time"], edges, labels=range(n_windows)))
    ev = ev.assign(win=pd.cut(ev["time"], edges, labels=range(n_windows)))

    drift = _regime_temporal_drift(ev, tr, min_events=min_events)
    drift["events"] = (drift["regime"].map(ev.groupby("edge_class").size())
                       .fillna(0).astype(int))
    drift = drift.sort_values("events", ascending=False)
    _write(drift.round(3), TEMPORAL_DRIFT_CSV)
    fig_temporal_drift(drift)
    print(drift.round(3).to_string(index=False))
    return drift


# ========= plotting =========


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved -> {path}")


def _despine(ax, sides=("top", "right")):
    for side in sides:
        ax.spines[side].set_visible(False)


def fig_exposure_unit_choice(curves, by_regime, rho_km, path=EXPOSURE_UNIT_FIG):
    """Rate across exposure deciles, flat under a clean offset and each
    regime's relative rate per km against per hour, ordered by how slow it is."""
    colors = {"per traversal": "dimgrey", "per rider-km": "crimson",
              "per rider-hour": "blue"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                      gridspec_kw={"wspace": 0.45})
    for unit, curve in curves.items():
        left.plot(np.linspace(1, 10, len(curve)), curve, "-o", color=colors[unit],
                  ms=5, lw=2, label=unit)
    left.axhline(1, color="black", ls="--", lw=1)
    left.set_xlabel("edge exposure decile, low to high")
    left.set_ylabel("rate / overall rate")
    left.set_title("A clean offset leaves the rate flat", fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=9, loc="upper right")

    y = np.arange(len(by_regime))
    right.hlines(y, by_regime["rel_km"], by_regime["rel_hr"], color="lightgrey", lw=2.5)
    right.scatter(by_regime["rel_km"], y, color="crimson", s=45, zorder=3, label="per rider-km")
    right.scatter(by_regime["rel_hr"], y, color="blue", s=45, zorder=3, label="per rider-hour")
    right.axvline(1, color="black", ls="--", lw=1)
    right.set_yticks(y)
    right.set_yticklabels([f"{r.replace('_', ' ')}  ({s:.0f} km/h)"
                           for r, s in zip(by_regime.index, by_regime["speed_kmh"])],
                          fontsize=9)
    right.invert_yaxis()
    right.set_xlabel("relative overtake rate (1 = network average)")
    right.set_title(f"Per-km inflates the slow streets, speed rho {rho_km:+.2f}",
                    fontsize=11, loc="left")
    right.legend(frameon=False, fontsize=9, loc="lower right")

    for ax in (left, right):
        _despine(ax)
    _save(fig, path)


def fig_rider_dominance(stats, path=RIDER_DOMINANCE_FIG):
    """The overall rate under both interval methods."""
    rate = stats["rate per hour"]
    bands = (("rider-clustered bootstrap", "clustered", "crimson"),
             ("naive Poisson, events independent", "naive", "dimgrey"))

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    for y, (label, key, color) in enumerate(bands):
        lo, hi = stats[f"{key} 95% lo"], stats[f"{key} 95% hi"]
        ax.errorbar(rate, y, xerr=[[rate - lo], [hi - rate]], fmt="o", color=color,
                    capsize=4, ms=8, lw=2, label=label)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([key for _, key, _ in bands])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("overall overtake rate (per rider-hour, 95% interval)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(f"Clustering by rider widens the interval "
                 f"{stats['interval inflation']}x, effective n "
                 f"{stats['effective sample size (Kish)']:.0f} not "
                 f"{stats['boxes with exposure']}", fontsize=11, loc="left")
    _despine(ax)
    _save(fig, path)


def fig_regime_dispersion(df, pooled, path=REGIME_DISPERSION_FIG):
    """One bar per regime against the Poisson line at 1 and the pooled value."""
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(df) + 1.4))
    ax.barh(y, df["dispersion"], color="crimson", alpha=0.85)
    ax.axvline(1, color="dimgrey", ls="--", lw=1.2)
    ax.text(1, len(df) - 0.3, " Poisson (=1)", color="dimgrey", fontsize=9, va="center")
    ax.axvline(pooled, color="black", ls=":", lw=1.5)
    ax.text(pooled, -0.7, f" pooled {pooled:.1f}", color="black", fontsize=9, ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.replace('_', ' ')}  (n={n})"
                        for r, n in zip(df["regime"], df["events"])])
    ax.invert_yaxis()
    ax.set_xlabel("overdispersion (variance / Poisson variance, given exposure)")
    ax.set_title(f"Overdispersion by street type, {df['dispersion'].min():.1f}x to "
                 f"{df['dispersion'].max():.1f}x", fontsize=11, loc="left", pad=14)
    _despine(ax)
    _save(fig, path)


def fig_temporal_drift(drift, path=TEMPORAL_DRIFT_FIG):
    """One interval per regime against the no-change line at 1x."""
    d = drift.dropna(subset=["rr"])
    colors = np.where(d["drift"], "crimson", "dimgrey")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(d) + 1.6))
    ax.hlines(y, d["lo"], d["hi"], color=colors, lw=2.2, alpha=0.6)
    ax.scatter(d["rr"], y, color=colors, s=49, zorder=3)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.replace('_', ' ')}  (n={n})"
                        for r, n in zip(d["regime"], d["events"])], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["1/4x", "1/2x", "1x (no drift)", "2x", "4x"])
    ax.set_xlabel("window-2 / window-1 rate ratio (95% rider-clustered CI)")
    for label, color in (("drift detected", "crimson"), ("no drift detected", "dimgrey")):
        ax.plot([], [], "o", color=color, label=label)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, frameon=False)
    ax.set_title(f"{int(d['drift'].sum())} of {len(d)} regimes drift between the windows",
                 fontsize=11, loc="left", pad=26)
    _despine(ax)
    _save(fig, path)


def fig_covariate_adjustment(df, path=COVARIATE_ADJUSTMENT_FIG):
    """Each covariate twice: on its own, and compared within street type."""
    y = np.arange(len(df))
    series = (("marginal, confounded by regime", "dimgrey", 0.16, "rr_marginal", "lo_m", "hi_m"),
              ("adjusted for regime", "crimson", -0.16, "rr_adjusted", "lo_a", "hi_a"))

    fig, ax = plt.subplots(figsize=(8.8, 0.9 * len(df) + 1.8))
    for label, color, offset, rate, lo, hi in series:
        ax.hlines(y + offset, df[lo], df[hi], color=color, lw=2, alpha=0.6)
        ax.scatter(df[rate], y + offset, color=color, s=64, zorder=3, label=label)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["covariate"], fontsize=11)
    ax.invert_yaxis()

    ax.set_xscale("log")
    lo_x = df[["lo_m", "lo_a"]].min().min() / 1.1
    hi_x = df[["hi_m", "hi_a"]].max().max() * 1.1
    ticks = [t for t in (0.5, 0.8, 1.0, 1.25, 1.5, 2.0) if lo_x <= t <= hi_x]
    ax.set_xlim(lo_x, hi_x)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}x" for t in ticks], fontsize=11)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("overtake rate ratio (log scale)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Covariate rate-ratios, marginal against regime-adjusted",
                 fontsize=11, loc="left")
    _despine(ax)
    _save(fig, path)


def fig_covariate_correlation(corr, coverage, n_edges, min_trav,
                              path=COVARIATE_CORRELATION_FIG):
    """Lower triangle only, dropping the row and column that would be entirely masked."""
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    sns.heatmap(corr.iloc[1:, :-1], mask=mask[1:, :-1], cmap="RdBu_r", vmin=-1, vmax=1,
                center=0, annot=True, fmt=".2f", annot_kws={"size": 8}, linewidths=0.6,
                linecolor="white", square=True,
                cbar_kws={"shrink": 0.55, "label": "Spearman rho"}, ax=ax)
    ax.set_title(f"What the overtake rate ties to, on {n_edges:,} edges with "
                 f">={min_trav} traversals", fontsize=11, loc="left")
    thin = [f"{c} on {v:.0%} of edges" for c, v in coverage.items() if v < 0.5]
    if thin:
        ax.text(0, -0.9, "sparse: " + ", ".join(thin), fontsize=8, color="dimgrey")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    _save(fig, path)


def main():
    oracle = pd.read_csv(ORACLE_CSV)
    ev, tr = _load_rides(oracle)

    print("\nexposure unit")
    exposure_unit_choice(oracle)
    print("\nPoisson against negative binomial")
    poisson_vs_nb(oracle)
    print("\nregime dispersion")
    regime_dispersion(oracle)
    print("\ncovariate adjustment")
    covariate_adjustment(oracle)
    print("\ncovariate correlation")
    covariate_correlation(oracle)
    print("\nMoran's I by threshold")
    moran_by_threshold(oracle)

    # what it leaves out
    print("\nrider dominance")
    rider_dominance(ev, tr)
    print("\ntemporal drift")
    temporal_drift(ev, tr)
    print("\nDONE")


if __name__ == "__main__":
    main()
