"""Task 5 diagnostics — the evidence behind the model choices.

Every function here answers "why is the estimator built this way?", not "what is in the
data" — the descriptive side lives in task4_oracle.py, next to the tables it describes.

  exposure_unit_choice()     why rider-hours and not traversals or rider-km
  rider_dominance()          a few boxes drive the estimate -> clustered CI (~6x wider)
  regime_dispersion()        overdispersion per street type (the sampling-budget knob)
  temporal_drift()           does each regime's rate drift between years? (clustered RR)
  poisson_vs_nb()            does the negative binomial beat Poisson? (AIC)
  covariate_adjustment()     covariate effects, marginal vs regime-adjusted
  covariate_correlation()    do static covariates predict the rate? (heatmap)
  moran_by_threshold()       residual spatial autocorrelation vs. exposure threshold

Tables -> output/task5_diagnostics/ ,  figures -> output/figures/.
Run from an activated env:   python task5a_diagnostics.py
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):   # keep unicode tables from crashing a cp1252 console
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(".")
ORACLE = BASE / "output/task4_oracle/task4_edge_oracle.csv"
TRAV = BASE / "output/task4_oracle/task4_edge_traversals.csv"
EVENTS = BASE / "output/task4_oracle/task4_edge_events.csv"
OUT = BASE / "output/task5_diagnostics"
FIG = BASE / "output/figures"

INK, MUTED, ACC, ACC2 = "#0b0b0b", "#52514e", "#2a78d6", "#e34948"
RNG = np.random.default_rng(42)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def _setup():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig]  {p}")


def _write(df, name):
    p = OUT / name
    df.to_csv(p, index=False)
    print(f"  [csv]  {p}")


def _poisson_ci(n, e):
    """95% interval on a rate n/e from the Poisson count n."""
    if e <= 0:
        return np.nan, np.nan, np.nan
    return n / e, max(n - 1.96 * np.sqrt(n), 0) / e, (n + 1.96 * np.sqrt(n)) / e


# ---------------------------------------------------------------------------
def _regime_drift(ev, tr, min_events=10):
    """Per-regime window rate-ratio (window 1 vs 0) with rider-CLUSTERED SE.

    Poisson GLM  n ~ win  with a log rider-km offset, on cells = regime x window x
    box, SEs clustered by box (repeated rides from one box are correlated). Returns
    one row per regime: rate-ratio, 95% CI, p, and a drift flag (CI excludes 1).
    ev, tr must already carry a 'win' column and edge_class/boxId/rider_km."""
    import statsmodels.api as sm

    km = tr.groupby(["edge_class", "win", "boxId"], observed=True)["rider_h"].sum()
    n = ev.groupby(["edge_class", "win", "boxId"], observed=True).size()
    cell = pd.concat([km.rename("km"), n.rename("n")], axis=1).reset_index()
    cell["km"] = cell["km"].fillna(0.0)
    cell["n"] = cell["n"].fillna(0).astype(int)
    cell = cell[cell["km"] > 0].dropna(subset=["win"]).copy()
    cell["win"] = cell["win"].astype(float)
    cell["logkm"] = np.log(cell["km"])

    rows = []
    for r, g in cell.groupby("edge_class"):
        if g["n"].sum() < min_events or g["win"].nunique() < 2:
            rows.append(dict(regime=r, rr=np.nan, lo=np.nan, hi=np.nan,
                             p=np.nan, drift=False))
            continue
        X = sm.add_constant(g[["win"]])             # intercept + window
        base = sm.GLM(g["n"], X, family=sm.families.Poisson(), offset=g["logkm"])
        try:
            m = base.fit(cov_type="cluster", cov_kwds={"groups": g["boxId"]})
        except Exception:
            m = base.fit()                          # too few boxes to cluster
        ci = np.exp(m.conf_int().loc["win"])
        rr, lo, hi, p = (float(np.exp(m.params["win"])), float(ci[0]),
                         float(ci[1]), float(m.pvalues["win"]))
        rows.append(dict(regime=r, rr=rr, lo=lo, hi=hi, p=p,
                         drift=bool(lo > 1 or hi < 1)))
    return pd.DataFrame(rows)


def temporal_drift(n_windows=2, min_events=10):
    """The authoritative temporal test: does each regime's rate drift between the
    two ~year windows? A per-regime window rate-ratio with rider-clustered SE
    (see _regime_drift) — this replaces the old CI-overlap 'stable' flag."""
    ev = pd.read_csv(EVENTS)
    tr = pd.read_csv(TRAV)
    cls = pd.read_csv(ORACLE, usecols=["u", "v", "edge_class"])
    tr = tr.merge(cls, on=["u", "v"], how="left")
    if "edge_class" not in ev.columns:
        ev = ev.merge(cls, on=["u", "v"], how="left")
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    tr["rider_h"] = tr["on_edge_s"] / 3600.0
    t0, t1 = tr["enter_time"].min(), tr["enter_time"].max()
    bins = [t0 + (t1 - t0) * f for f in np.linspace(0, 1, n_windows + 1)]
    bins[0] -= pd.Timedelta(seconds=1)
    bins[-1] += pd.Timedelta(seconds=1)
    tr["win"] = pd.cut(tr["enter_time"], bins=bins, labels=range(n_windows))
    ev["win"] = pd.cut(ev["time"], bins=bins, labels=range(n_windows))

    drift = _regime_drift(ev, tr, min_events=min_events)
    ev_tot = ev.groupby("edge_class").size()
    drift["events"] = drift["regime"].map(ev_tot).fillna(0).astype(int)
    drift = drift.sort_values("events", ascending=False)
    _write(drift.round(3), "task5a_temporal_drift.csv")

    d = drift.dropna(subset=["rr"])
    fig, ax = plt.subplots(figsize=(8.4, 0.5 * len(d) + 1.6))
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        col = ACC2 if r.drift else MUTED
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=2.2, alpha=0.6,
                solid_capstyle="round")
        ax.plot(r.rr, i, "o", color=col, ms=7)
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.replace('_', ' ')}  (n={n})"
                        for r, n in zip(d["regime"], d["events"])], fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["¼×", "½×", "1× (no drift)", "2×", "4×"])
    ax.set_xlabel("window-2 ÷ window-1 overtake-rate ratio  (95% rider-clustered CI)")
    ax.plot([], [], "o", color=ACC2, label="drift detected (CI excludes 1)")
    ax.plot([], [], "o", color=MUTED, label="no drift detected")
    # above the axes: the widest CI (thin regimes) runs to the far right of the bottom rows
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, frameon=False)
    n_drift = int(d["drift"].sum())
    ax.set_title(f"Rates are not stable across the two years ({n_drift}/{len(d)} regimes drift)\n"
                 "the two best-measured road regimes both rose ~70% — but which riders were "
                 "active also changed between windows",
                 fontsize=11.5, loc="left", pad=26)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "task5a_temporal_drift.png")
    print(drift.to_string(index=False))
    return drift


# ---------------------------------------------------------------------------
def rider_dominance(n_boot=3000):
    """Quantify how concentrated the data is in a few boxes, and compare a
    naive Poisson interval on the overall rate against a rider-clustered
    bootstrap interval (resampling whole boxes)."""
    ev = pd.read_csv(EVENTS)
    tr = pd.read_csv(TRAV)
    tr["rider_km"] = tr["length_m"] / 1000.0
    ev_box = ev.groupby("boxId").size()
    km_box = tr.groupby("boxId")["rider_km"].sum()
    boxes = sorted(set(ev_box.index) | set(km_box.index))
    per = pd.DataFrame(index=boxes)
    per["ev"] = ev_box.reindex(boxes).fillna(0).astype(int)
    per["km"] = km_box.reindex(boxes).fillna(0.0)
    per = per[per["km"] > 0]

    tot_ev, tot_km = per["ev"].sum(), per["km"].sum()
    rate = tot_ev / tot_km
    top3_ev = per["ev"].sort_values(ascending=False).head(3).sum() / tot_ev
    top3_km = per["km"].sort_values(ascending=False).head(3).sum() / tot_km
    evv, kmv = per["ev"].values, per["km"].values
    n_eff = evv.sum() ** 2 / (evv ** 2).sum()         # Kish effective sample size

    _, nlo, nhi = _poisson_ci(tot_ev, tot_km)         # naive Poisson interval
    boot = np.array([
        evv[s].sum() / kmv[s].sum()
        for s in (RNG.integers(0, len(per), len(per)) for _ in range(n_boot))])
    clo, chi = np.percentile(boot, [2.5, 97.5])

    summary = pd.DataFrame([
        dict(metric="boxes (with exposure)", value=round(len(per), 1)),
        dict(metric="events", value=int(tot_ev)),
        dict(metric="rider-km", value=round(tot_km, 1)),
        dict(metric="overall rate /km", value=round(rate, 3)),
        dict(metric="top-3 boxes share of events", value=round(top3_ev, 3)),
        dict(metric="top-3 boxes share of km", value=round(top3_km, 3)),
        dict(metric="effective sample size (Kish)", value=round(n_eff, 1)),
        dict(metric="naive 95% lo", value=round(nlo, 3)),
        dict(metric="naive 95% hi", value=round(nhi, 3)),
        dict(metric="clustered 95% lo", value=round(clo, 3)),
        dict(metric="clustered 95% hi", value=round(chi, 3)),
        dict(metric="CI width inflation (clustered/naive)",
             value=round((chi - clo) / (nhi - nlo), 2)),
    ])
    _write(summary, "task5a_rider_dominance.csv")

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    ax.errorbar(rate, 1, xerr=[[rate - nlo], [nhi - rate]], fmt="o", color=MUTED,
                capsize=4, ms=8, lw=2, label="naive Poisson (events independent)")
    ax.errorbar(rate, 0, xerr=[[rate - clo], [chi - rate]], fmt="o", color=ACC2,
                capsize=4, ms=8, lw=2, label="rider-clustered bootstrap")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["clustered", "naive"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("overall overtake rate  (per rider-km, 95% interval)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(f"A few boxes carry the data — top 3 hold {top3_ev:.0%} of events, "
                 f"{top3_km:.0%} of km\n"
                 f"clustering by rider widens the interval "
                 f"{(chi - clo) / (nhi - nlo):.1f}× (effective n ≈ {n_eff:.0f}, not {len(per)})",
                 fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "task5a_rider_dominance.png")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
def poisson_vs_nb():
    """Fit a regime-only rate model as Poisson and as Negative Binomial (both
    with a log rider-km offset); compare AIC. Lower AIC for NB confirms the
    overdispersion needs the extra parameter."""
    import warnings

    import statsmodels.api as sm
    o = pd.read_csv(ORACLE)
    d = o[(o["is_observed"]) & (o["rider_km"] > 0)].copy()
    d["logE"] = np.log(d["rider_km"])
    y = d["n_events"]
    X = sm.add_constant(pd.get_dummies(d["edge_class"], drop_first=True, dtype=float))
    off = d["logE"].values
    pois = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit()
    # profile the NB dispersion alpha — a GLM at fixed alpha always converges,
    # so this is far more stable than the joint NB2 optimiser
    best_ll, best_a = -np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in np.geomspace(0.05, 10, 60):
            m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=a),
                       offset=off).fit()
            if m.llf > best_ll:
                best_ll, best_a = m.llf, a
    k_p = int(pois.df_model + 1)
    k_nb = k_p + 1                                   # + the dispersion parameter
    aic_p = -2 * pois.llf + 2 * k_p
    aic_nb = -2 * best_ll + 2 * k_nb
    pearson_disp = pois.pearson_chi2 / pois.df_resid
    df = pd.DataFrame([
        dict(model="Poisson", loglik=round(pois.llf, 1), k=k_p, AIC=round(aic_p, 1)),
        dict(model="Negative Binomial", loglik=round(best_ll, 1), k=k_nb,
             AIC=round(aic_nb, 1)),
        dict(model="ΔAIC (Poisson − NB)", loglik="", k="", AIC=round(aic_p - aic_nb, 1)),
        dict(model="NB dispersion α", loglik="", k="", AIC=round(best_a, 3)),
        dict(model="Poisson Pearson dispersion", loglik="", k="", AIC=round(pearson_disp, 2)),
    ])
    _write(df, "task5a_poisson_vs_nb.csv")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
def covariate_adjustment():
    """Corrects the 'covariates are flat' reading of the forest plot, which is
    UNIVARIATE and confounded by regime. Each covariate's marginal rate-ratio is
    shown next to its regime-ADJUSTED one (NegBin, log rider-km offset). Adjusting
    for regime shrinks every covariate to a small effect, and betweenness flips to
    the 'wrong' (confounded) sign — so the honest reading is 'second-order and
    regime-collinear', not 'flat'."""
    import statsmodels.api as sm
    o = pd.read_csv(ORACLE)
    d = o[(o["is_observed"]) & (o["rider_km"] > 0)].dropna(subset=["maxspeed_kmh"]).copy()
    d["logE"] = np.log(d["rider_km"])
    d["acc_any"] = (d["n_acc_bike"] > 0).astype(int)
    d["sp10"] = d["maxspeed_kmh"] / 10.0                      # per +10 km/h
    lb = np.log(d["betweenness"].clip(lower=1e-7))
    d["lbet_z"] = (lb - lb.mean()) / lb.std()                # per +1 SD
    terms = [("acc_any", "≥1 bike accident"),
             ("sp10", "speed  +10 km/h"),
             ("lbet_z", "betweenness  +1 SD")]
    ALPHA = 2.8
    dummies = pd.get_dummies(d["edge_class"], drop_first=True, dtype=float)

    def rr(X, name):
        X = sm.add_constant(X.astype(float))
        m = sm.GLM(d["n_events"], X, family=sm.families.NegativeBinomial(alpha=ALPHA),
                   offset=d["logE"].values).fit()
        ci = m.conf_int()
        return (float(np.exp(m.params[name])), float(np.exp(ci.loc[name, 0])),
                float(np.exp(ci.loc[name, 1])), float(m.pvalues[name]))

    rows = []
    for t, lab in terms:
        rm, lm, hm, _ = rr(d[[t]], t)                                # marginal
        ra, la, ha, pa = rr(pd.concat([dummies, d[[t]]], axis=1), t)  # regime-adjusted
        rows.append(dict(covariate=lab, rr_marginal=rm, lo_m=lm, hi_m=hm,
                         rr_adjusted=ra, lo_a=la, hi_a=ha, p_adjusted=pa))
    df = pd.DataFrame(rows)
    _write(df.round(3), "task5a_covariate_adjustment.csv")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.8, 0.9 * len(df) + 1.8))
    y = np.arange(len(df))
    for i, r in enumerate(df.itertuples()):
        ax.plot([r.lo_m, r.hi_m], [i + 0.16] * 2, color=MUTED, lw=2, alpha=0.5,
                solid_capstyle="round")
        ax.plot(r.rr_marginal, i + 0.16, "o", color=MUTED, ms=8)
        ax.plot([r.lo_a, r.hi_a], [i - 0.16] * 2, color=ACC, lw=2, alpha=0.6,
                solid_capstyle="round")
        ax.plot(r.rr_adjusted, i - 0.16, "o", color=ACC, ms=8)
    ax.axvline(1, color=INK, ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r.covariate for r in df.itertuples()], fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.72, 1.55)
    ax.set_xticks([0.8, 1.0, 1.25, 1.5])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("overtake rate-ratio  (log scale)")
    ax.plot([], [], "o", color=MUTED, label="marginal (univariate, confounded by regime)")
    ax.plot([], [], "o", color=ACC, label="adjusted for regime (partial effect)")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title("Covariates are second-order and regime-collinear, not 'flat'\n"
                 "adjusting for regime: accidents & speed wash out; betweenness stays "
                 "significant but negative (a confound)",
                 fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "task5a_covariate_adjusted.png")
    return df


# ---------------------------------------------------------------------------
def _eta2_ranks(x, groups):
    """Share of x's RANK variance explained by a categorical grouping (eta-squared on
    ranks) — the categorical analogue of Spearman, on the same 0..1 scale."""
    from scipy.stats import rankdata
    m = x.notna()
    if m.sum() < 2:
        return np.nan
    r = pd.Series(rankdata(x[m]), index=x[m].index)
    gm = r.mean()
    ssb = sum(len(v) * (v.mean() - gm) ** 2 for _, v in r.groupby(groups[m]))
    return float(ssb / ((r - gm) ** 2).sum())


def covariate_correlation(min_trav=3):
    """Spearman correlation heatmap: the per-edge overtake rate against the static
    covariates, on well-observed edges (>= min_trav traversals, to tame the rate
    noise). Shows the honest picture — the rate ties only weakly to any covariate,
    and the covariates are collinear with each other, so none is a clean predictor.
    Accidents are the external (injury-record) signal to eyeball for convergent
    validity; AADT is included but only 3% of edges carry it."""
    import seaborn as sns

    o = pd.read_csv(ORACLE)
    d = o[(o["is_observed"]) & (o["n_traversals"] >= min_trav) & (o["rider_h"] > 0)].copy()
    d["overtake_rate"] = d["n_events"] / d["rider_h"]          # the modelled estimand
    d["speed_kmh"] = d["rider_km"] / d["rider_h"]
    names = {"overtake_rate": "overtake rate", "n_traversals": "traversals",
             "rider_km": "rider-km", "rider_h": "rider-hours", "speed_kmh": "rider speed",
             "length_m": "edge length", "n_acc_bike": "bike accidents",
             "maxspeed_kmh": "speed limit", "betweenness": "betweenness",
             "lanes_n": "car lanes", "aadt_kfz": "AADT"}
    names = {k: v for k, v in names.items() if k in d.columns}
    sub = d[list(names)].rename(columns=names)
    corr = sub.corr(method="spearman")        # pairwise-complete: each cell drops its own NaNs

    # street regime is categorical, so its association is eta^2 (share of RANK variance it
    # explains) rather than a correlation — appended as the last row/col, same 0..1 direction
    eta = {c: _eta2_ranks(sub[c], d["edge_class"]) for c in sub.columns}
    corr.loc["street regime η²"] = pd.Series(eta)
    corr["street regime η²"] = pd.Series({**eta, "street regime η²": 1.0})
    _write(corr.round(3).reset_index(names="var"), "task5a_covariate_correlation.csv")

    # lower triangle only, minus the all-masked first row and last column
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_t, mask_t = corr.iloc[1:, :-1], mask[1:, :-1]
    thin = [c for c in sub.columns if sub[c].notna().mean() < 0.5]   # sparse covariates
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    sns.heatmap(corr_t, mask=mask_t, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", annot_kws={"size": 8}, linewidths=0.6, linecolor="white",
                square=True, cbar_kws={"shrink": 0.55, "label": "Spearman ρ"}, ax=ax)
    ax.set_title(f"What the overtake rate ties to   (n={len(d):,} edges, ≥{min_trav} traversals)",
                 fontsize=12, loc="left")
    note = "cells are Spearman ρ; bottom row is η² (rank variance explained by regime)"
    if thin:
        note += "\nsparse: " + ", ".join(f"{c} on {sub[c].notna().mean():.0%} of edges" for c in thin)
    ax.text(0, -0.9, note, fontsize=8, color=MUTED)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    _save(fig, "task5a_covariate_correlation.png")
    print(corr.round(2).to_string())
    print("\n  pairwise coverage (share of the subset with a value):")
    print("   " + " | ".join(f"{c} {sub[c].notna().mean():.0%}" for c in sub.columns))
    return corr


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def _moran(sub, col, n_perm=999):
    """Moran's I using binary edge adjacency (two streets share an endpoint node)."""
    sub = sub.reset_index(drop=True)

    node2edges = {}
    for i, (u, v) in enumerate(zip(sub["u_lo"], sub["v_hi"])):
        node2edges.setdefault(u, []).append(i)
        node2edges.setdefault(v, []).append(i)

    pairs = set()
    for edges in node2edges.values():
        if len(edges) > 1:
            for i, j in combinations(edges, 2):
                pairs.add((min(i, j), max(i, j)))
    if not pairs:
        return None

    I_idx = np.fromiter((i for i, _ in pairs), dtype=int)
    J_idx = np.fromiter((j for _, j in pairs), dtype=int)

    z = sub[col].to_numpy(float)
    z = z - z.mean()
    denom = np.dot(z, z)
    if denom == 0:                      # all residuals identical -> I undefined
        return None

    n, w = len(z), len(pairs)
    I = (n / w) * np.sum(z[I_idx] * z[J_idx]) / denom

    perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = RNG.permutation(z)
        perm[k] = (n / w) * np.sum(zp[I_idx] * zp[J_idx]) / denom
    p = (np.sum(np.abs(perm) >= abs(I)) + 1) / (n_perm + 1)

    return {"streets": n, "pairs": w, "mean_neighbors": 2 * w / n,
            "morans_I": I, "perm_p": p}


def moran_by_threshold(thresholds=(1, 2, 4, 6, 8, 10)):
    """Moran's I of street-type Pearson residuals as observation coverage increases.

    Streets are aggregated to UNDIRECTED edges (pooling directions — otherwise the two
    directions of one street, sharing both endpoints, count as each other's neighbour).
    Expected counts mu = regime_rate(edge_class) x rider_h; the Pearson residual
    (N - mu)/sqrt(mu) is what is tested. The raw rate difference is NOT used: it is
    dominated by low-exposure streets (one overtake in a few seconds = thousands/hour).

    Read the headline value at the well-observed end. The rise across thresholds is a
    consistency check, not a finding: sparse streets are noise-dominated and noise is
    spatially independent, so I is attenuated toward 0 there by construction. Rows are
    also nested, so they are not independent samples."""
    o = pd.read_csv(ORACLE)
    g = (o.groupby(["u_lo", "v_hi"])
         .agg(n_events=("n_events", "sum"), rider_h=("rider_h", "sum"),
              n_trav=("n_traversals", "sum"), edge_class=("edge_class", "first"))
         .reset_index())
    g = g[g["rider_h"] > 0].copy()
    reg = g.groupby("edge_class").apply(
        lambda d: d["n_events"].sum() / d["rider_h"].sum(), include_groups=False).to_dict()
    mu = (g["edge_class"].map(reg) * g["rider_h"]).clip(lower=1e-9)
    g["pearson"] = (g["n_events"] - mu) / np.sqrt(mu)

    rows = []
    for t in thresholds:
        res = _moran(g[g["n_trav"] >= t], "pearson")
        if res is None:
            continue
        rows.append({"min_traversals": t,
                     **{k: round(v, 3) if isinstance(v, float) else v
                        for k, v in res.items()}})

    df = pd.DataFrame(rows)
    _write(df, "task5a_moran_threshold.csv")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
def regime_dispersion():
    """Overdispersion per street type: variance of counts vs. what a Poisson
    (given each edge's exposure) would allow. 1 = Poisson-like, >1 = overdispersed."""
    o = pd.read_csv(ORACLE)
    d = o[(o["is_observed"]) & (o["rider_km"] > 0)].copy()
    rows = []
    for r, g in d.groupby("edge_class"):
        if g["n_events"].sum() < 20:
            continue
        rate = g["n_events"].sum() / g["rider_km"].sum()
        mu = (rate * g["rider_km"]).clip(lower=1e-9)
        disp = ((g["n_events"] - mu) ** 2 / mu).sum() / (len(g) - 1)
        rows.append(dict(regime=r, edges=len(g), events=int(g["n_events"].sum()),
                         rate_per_km=round(rate, 3), dispersion=round(disp, 2)))
    df = pd.DataFrame(rows).sort_values("dispersion", ascending=False)
    # pooled
    rate = d["n_events"].sum() / d["rider_km"].sum()
    mu = (rate * d["rider_km"]).clip(lower=1e-9)
    pooled = ((d["n_events"] - mu) ** 2 / mu).sum() / (len(d) - 1)
    _write(df, "task5a_regime_dispersion.csv")

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(df) + 1.4))
    ax.barh(range(len(df)), df["dispersion"], color=ACC, alpha=0.85)
    ax.axvline(1, color=MUTED, ls="--", lw=1.2)
    ax.text(1, len(df) - 0.3, " Poisson (=1)", color=MUTED, fontsize=9, va="center")
    ax.axvline(pooled, color=ACC2, ls=":", lw=1.5)
    ax.text(pooled, -0.7, f"pooled {pooled:.1f}", color=ACC2, fontsize=9, ha="center")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.regime.replace('_', ' ')}  (n={r.events})"
                        for r in df.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("overdispersion  (variance ÷ Poisson variance, given exposure)")
    ax.set_title("Overdispersion differs by street type (1–13×)\n"
                 "— the argument for fitting the Gamma prior per regime",
                 fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "task5a_regime_dispersion.png")
    print(df.to_string(index=False), f"\n  pooled dispersion = {pooled:.2f}")
    return df




def exposure_unit_choice(n_bins=10):
    """Why the risk model measures exposure in rider-HOURS, not traversals or rider-km.

    Overtaking is a temporal arrival process — cars pass a rider at some rate per unit of
    TIME — so the natural Poisson offset is time-at-risk. The candidates are linked by an
    identity,
        rate_per_km = rate_per_hour / speed,
    so a per-km rate carries the rider's speed inside it: wherever cyclists are slow
    (junctions, congestion, pedestrian zones) it is inflated by the slowness itself rather
    than by extra danger.

    Three model-free checks:
      (1) speed varies systematically by street type, and the per-km relative rate is
          strongly rank-correlated with slowness — per-km partly measures speed;
      (2) a Poisson GLM (N ~ regime, log-exposure offset) keeps the SAME response N under
          all three offsets, so their AIC is directly comparable;
      (3) a clean offset makes counts proportional to exposure: adding log(E) as a
          covariate on top of the log(E) offset should give a slope of 0.
    """
    import statsmodels.api as sm
    from scipy.stats import spearmanr

    units = {"per traversal": "n_traversals", "per rider-km": "rider_km",
             "per rider-hour": "rider_h"}
    cols = {"per traversal": MUTED, "per rider-km": ACC2, "per rider-hour": ACC}

    o = pd.read_csv(ORACLE)
    d = o[o["is_observed"] & (o["rider_km"] > 0) & (o["rider_h"] > 0)].copy()

    # (1) what a per-km rate silently contains -------------------------------
    reg = d.groupby("edge_class").agg(events=("n_events", "sum"),
                                      km=("rider_km", "sum"), hr=("rider_h", "sum"))
    reg = reg[reg["events"] >= 20].copy()
    reg["speed_kmh"] = reg["km"] / reg["hr"]
    reg["rel_km"] = (reg["events"] / reg["km"]) / (d["n_events"].sum() / d["rider_km"].sum())
    reg["rel_hr"] = (reg["events"] / reg["hr"]) / (d["n_events"].sum() / d["rider_h"].sum())
    reg = reg.sort_values("speed_kmh")
    rho_km = float(spearmanr(reg["speed_kmh"], reg["rel_km"]).statistic)
    rho_hr = float(spearmanr(reg["speed_kmh"], reg["rel_hr"]).statistic)

    # (2) GLM fit + (3) proportionality of counts to exposure -----------------
    dummies = pd.get_dummies(d["edge_class"], drop_first=True, dtype=float)
    X = sm.add_constant(dummies)
    rows, curves = [], {}
    for lab, col in units.items():
        logE = np.log(d[col])
        m = sm.GLM(d["n_events"], X, family=sm.families.Poisson(), offset=logE).fit()
        # slope of log(E) ON TOP of the log(E) offset: 0 => counts scale with exposure.
        # (bin-free, unlike a decile ratio — n_traversals is discrete and bins coarsely)
        Xs = sm.add_constant(pd.concat([dummies, logE.rename("logE")], axis=1))
        ms = sm.GLM(d["n_events"], Xs, family=sm.families.Poisson(), offset=logE).fit()
        b = (d.groupby(pd.qcut(d[col], n_bins, duplicates="drop"), observed=True)
             .agg(N=("n_events", "sum"), E=(col, "sum")))
        curves[lab] = (b["N"] / b["E"]).to_numpy() / (d["n_events"].sum() / d[col].sum())
        rows.append(dict(unit=lab, AIC=round(m.aic, 1),
                         dispersion=round(m.pearson_chi2 / m.df_resid, 2),
                         scale_slope=round(float(ms.params["logE"]), 3),
                         slope_p=round(float(ms.pvalues["logE"]), 4)))
    summary = pd.DataFrame(rows)
    summary["dAIC"] = (summary["AIC"] - summary["AIC"].min()).round(1)
    _write(summary, "task5a_exposure_unit_choice.csv")
    _write(reg.round(3).reset_index(), "task5a_exposure_unit_by_regime.csv")

    best = summary.loc[summary["AIC"].idxmin(), "unit"]
    print(summary.to_string(index=False))
    print(f"\n  speed across street types: {reg['speed_kmh'].min():.1f}-{reg['speed_kmh'].max():.1f} km/h")
    print(f"  rank corr(speed, relative rate): per-km {rho_km:+.2f} | per-hour {rho_hr:+.2f}"
          "   <- per-km tracks slowness more closely")
    print(f"  best offset by AIC: {best}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8),
                                   gridspec_kw={"wspace": 0.45})
    for lab, c in curves.items():
        ax1.plot(np.linspace(1, 10, len(c)), c, "-o", color=cols[lab], ms=5, lw=2, label=lab)
    ax1.axhline(1, color=INK, ls="--", lw=1)
    ax1.set_xlabel("edge exposure decile  (low → high)")
    ax1.set_ylabel("rate ÷ overall rate")
    ax1.set_title("A clean offset leaves the rate flat", fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    y = np.arange(len(reg))
    ax2.hlines(y, reg["rel_km"], reg["rel_hr"], color="0.82", lw=2.5, zorder=1)
    ax2.scatter(reg["rel_km"], y, color=ACC2, s=45, zorder=3, label="per rider-km")
    ax2.scatter(reg["rel_hr"], y, color=ACC, s=45, zorder=3, label="per rider-hour")
    ax2.axvline(1, color=INK, ls="--", lw=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{r.replace('_', ' ')}  ({s:.0f} km/h)"
                         for r, s in zip(reg.index, reg["speed_kmh"])], fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("relative overtake rate  (1 = network average)")
    ax2.set_title(f"Per-km inflates the SLOW streets  (speed ρ = {rho_km:+.2f})",
                  fontsize=11, loc="left")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    for a in (ax1, ax2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    _save(fig, "task5a_exposure_unit_choice.png")
    return summary


def main():
    _setup()
    for name, fn in [("EXPOSURE UNIT CHOICE (why rider-hours)", exposure_unit_choice),
                     ("RIDER DOMINANCE", rider_dominance),
                     ("REGIME DISPERSION", regime_dispersion),
                     ("TEMPORAL DRIFT (clustered rate-ratio test)", temporal_drift),
                     ("POISSON vs NEGATIVE BINOMIAL", poisson_vs_nb),
                     ("COVARIATE ADJUSTMENT (marginal vs regime-adjusted)", covariate_adjustment),
                     ("COVARIATE CORRELATION HEATMAP", covariate_correlation),
                     ("MORAN'S I BY THRESHOLD", moran_by_threshold)]:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        fn()


if __name__ == "__main__":
    main()
