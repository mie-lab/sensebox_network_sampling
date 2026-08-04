"""Task 5 diagnostics — the evidence behind the model choices.

  rider_dominance()          a few boxes drive the estimate -> clustered CI (~6x wider)
  regime_dispersion()        overdispersion per street type (the sampling-budget knob)
  temporal_drift()           does each regime's rate drift between years? (clustered RR)
  poisson_vs_nb()            does the negative binomial beat Poisson? (AIC)
  covariate_adjustment()     covariate effects, marginal vs regime-adjusted
  covariate_correlation()    do static covariates predict the rate? (heatmap)
  intersection_robustness()  is the regime ranking a junction artifact?
  moran_by_threshold()       residual spatial autocorrelation vs. exposure threshold
  forest_support_table()     the numbers behind the Task-4 forest plot + a support flag

Tables -> output/task5_diagnostics/ ,  figures -> output/figures/.
Run from an activated env:   python diagnostics_task5.py
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
ORACLE = BASE / "output/task4_oracle/edge_oracle_task4.csv"
TRAV = BASE / "output/task4_oracle/edge_traversals_task4.csv"
EVENTS = BASE / "output/task4_oracle/edge_events_task4.csv"
MATCHED_EVENTS = BASE / "output/task3_matching/matched_events_task3.gpkg"
GRAPHML = BASE / "input/muenster_bike.graphml"
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

    km = tr.groupby(["edge_class", "win", "boxId"], observed=True)["rider_km"].sum()
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
    tr["rider_km"] = tr["length_m"] / 1000.0
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
    _write(drift.round(3), "temporal_drift.csv")

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
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    n_drift = int(d["drift"].sum())
    ax.set_title("Drift is only resolvable where riders are many "
                 f"({n_drift}/{len(d)} regimes)\n"
                 "point estimates mostly rise, but the thin late window can't confirm it "
                 "— temporal coverage is starved",
                 fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "fig_temporal_drift.png")
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
    _write(summary, "rider_dominance.csv")

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
    _save(fig, "fig_rider_dominance.png")
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
    _write(df, "poisson_vs_nb.csv")
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
    _write(df.round(3), "covariate_adjustment.csv")
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
    _save(fig, "fig_covariate_adjusted.png")
    return df


# ---------------------------------------------------------------------------
def covariate_correlation(min_trav=3):
    """Spearman correlation heatmap: the per-edge overtake rate against the static
    covariates, on well-observed edges (>= min_trav traversals, to tame the rate
    noise). Shows the honest picture — the rate ties only weakly to any covariate,
    and the covariates are collinear with each other, so none is a clean predictor.
    Accidents are the external (injury-record) signal to eyeball for convergent
    validity; AADT is included but only 3% of edges carry it."""
    o = pd.read_csv(ORACLE)
    d = o[(o["is_observed"]) & (o["n_traversals"] >= min_trav) & (o["rider_km"] > 0)].copy()
    d["overtake_rate"] = d["n_events"] / d["rider_km"]
    names = {"overtake_rate": "overtake rate", "n_acc_bike": "bike accidents",
             "maxspeed_kmh": "speed limit", "betweenness": "betweenness",
             "aadt_kfz": "AADT (3% cov)", "lanes_n": "car lanes",
             "length_m": "edge length", "n_traversals": "traversals"}
    names = {k: v for k, v in names.items() if k in d.columns}
    corr = d[list(names)].rename(columns=names).corr(method="spearman")
    _write(corr.round(2).reset_index(names="var"), "covariate_correlation.csv")

    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else INK)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ")
    ax.set_title(f"Static covariates vs the overtake rate  (edges with ≥ {min_trav} traversals, "
                 f"n={len(d)})\nrate ties weakly to everything; covariates are collinear "
                 "→ none is a clean predictor",
                 fontsize=10.5, loc="left")
    _save(fig, "fig_covariate_correlation.png")
    print(corr.to_string())
    return corr


# ---------------------------------------------------------------------------
def intersection_robustness(buffer_m=15):
    """Is the regime ranking an intersection artifact? Overtakes are modelled as an
    edge property, yet many sit near junctions (turning/queuing cars, and matching
    is least sure there). Refit each regime's rate with events near a REAL road
    junction removed, and check the ranking holds.

    A 'real junction' is a node of degree >= 3 in the MOTOR-road subgraph — this
    drops the many degree-3 cycleway/path connectors that inflate the naive count."""
    import geopandas as gpd
    import osmnx as ox
    from collections import Counter
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    MOTOR = {"primary", "primary_link", "secondary", "secondary_link", "tertiary",
             "tertiary_link", "trunk", "trunk_link", "unclassified", "residential",
             "living_street"}                       # service excluded: low cross-traffic

    def hw1(d):
        h = d.get("highway")
        return h[0] if isinstance(h, list) else h

    G = ox.load_graphml(GRAPHML)
    deg = Counter()
    for u, v, d in ox.convert.to_undirected(G).edges(data=True):
        if hw1(d) in MOTOR:
            deg[u] += 1
            deg[v] += 1
    nodes = ox.graph_to_gdfs(G, edges=False).to_crs(25832)
    real = nodes[[deg.get(n, 0) >= 3 for n in nodes.index]]

    ev = gpd.read_file(MATCHED_EVENTS).to_crs(25832)
    ev = ev[ev["edge_class"].notna()].copy()
    tree = cKDTree(np.c_[real.geometry.x, real.geometry.y])
    dist, _ = tree.query(np.c_[ev.geometry.x, ev.geometry.y])
    ev["near"] = dist <= buffer_m

    obs = pd.read_csv(ORACLE)
    obs = obs[obs["is_observed"]]
    km = obs.groupby("edge_class")["rider_km"].sum()
    reg = pd.DataFrame({"km": km,
                        "n_all": ev.groupby("edge_class").size(),
                        "n_mid": ev[~ev["near"]].groupby("edge_class").size()})
    reg = reg.dropna(subset=["km"]).fillna({"n_all": 0, "n_mid": 0})
    reg = reg[reg["n_all"] >= 20].copy()
    reg["share_near"] = 1 - reg["n_mid"] / reg["n_all"]
    reg["rate_all"] = reg["n_all"] / reg["km"]
    reg["rate_mid"] = reg["n_mid"] / reg["km"]
    ov_all = reg["n_all"].sum() / reg["km"].sum()
    ov_mid = reg["n_mid"].sum() / reg["km"].sum()
    reg["rel_all"] = reg["rate_all"] / ov_all
    reg["rel_mid"] = reg["rate_mid"] / ov_mid
    rho = float(spearmanr(reg["rel_all"], reg["rel_mid"]).statistic)
    reg = reg.sort_values("rate_all", ascending=False).reset_index()
    _write(reg.round(3), "intersection_robustness.csv")

    overall = float(ev["near"].mean())
    print(f"  real motor-junction nodes: {len(real)}  |  events near (<= {buffer_m} m): {overall:.0%}")
    print(f"  regime relative-rate rank stability (Spearman, full vs mid-edge): rho = {rho:.2f}")
    print(reg[["edge_class", "n_all", "share_near", "rel_all", "rel_mid"]].to_string(index=False))

    verdict = "barely moves" if rho >= 0.9 else ("mostly holds" if rho >= 0.7 else "shifts")
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    lim = max(reg["rel_all"].max(), reg["rel_mid"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=1, label="unchanged")
    ax.scatter(reg["rel_all"], reg["rel_mid"], s=45, color=ACC, zorder=3)
    for r in reg.itertuples():
        ax.annotate(r.edge_class.replace("_", " "), (r.rel_all, r.rel_mid), (5, 3),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("relative overtake rate — all events")
    ax.set_ylabel(f"relative rate — events within {buffer_m} m of a junction removed")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.set_title(f"Regime ranking {verdict} when near-junction overtakes are dropped\n"
                 f"{overall:.0%} of events sit near a real road junction · rank ρ = {rho:.2f}",
                 fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "fig_intersection_robustness.png")
    return reg


# ---------------------------------------------------------------------------
def _moran(sub, col):
    sub = sub.reset_index(drop=True)
    node2e = {}
    for i, (a, b) in enumerate(zip(sub["u_lo"], sub["v_hi"])):
        node2e.setdefault(a, []).append(i)
        node2e.setdefault(b, []).append(i)
    pairs = set()
    for edges in node2e.values():
        if len(edges) > 1:
            for i, j in combinations(edges, 2):
                pairs.add((i, j) if i < j else (j, i))
    if not pairs:
        return None
    I_idx = np.fromiter((p[0] for p in pairs), int)
    J_idx = np.fromiter((p[1] for p in pairs), int)
    z = sub[col].to_numpy(float) - sub[col].mean()
    N, P = len(z), len(pairs)
    denom = (z ** 2).sum()
    I = (N / P) * (z[I_idx] * z[J_idx]).sum() / denom
    perm = np.array([(N / P) * (zp[I_idx] * zp[J_idx]).sum() / denom
                     for zp in (RNG.permutation(z) for _ in range(999))])
    p = (np.sum(np.abs(perm) >= abs(I)) + 1) / 1000
    return N, P, I, p


def moran_by_threshold(thresholds=(1, 2, 4, 6, 8, 10)):
    """Moran's I of the residual rate (rate - regime mean) on undirected
    streets, at rising exposure thresholds; the signal emerges as the
    per-edge rate gets less noisy."""
    o = pd.read_csv(ORACLE)
    g = (o.groupby(["u_lo", "v_hi"])
         .agg(n_events=("n_events", "sum"), rider_km=("rider_km", "sum"),
              n_trav=("n_traversals", "sum"), edge_class=("edge_class", "first"))
         .reset_index())
    g = g[g["rider_km"] > 0].copy()
    g["rate"] = g["n_events"] / g["rider_km"]
    reg = g.groupby("edge_class").apply(
        lambda d: d["n_events"].sum() / d["rider_km"].sum(), include_groups=False).to_dict()
    g["resid"] = g["rate"] - g["edge_class"].map(reg)

    rows = []
    for t in thresholds:
        sub = g[g["n_trav"] >= t]
        r = _moran(sub, "resid")
        if r:
            rows.append(dict(min_traversals=t, streets=r[0], pairs=r[1],
                             morans_I=round(r[2], 4), perm_p=round(r[3], 3)))
    df = pd.DataFrame(rows)
    _write(df, "moran_threshold.csv")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.plot(df["min_traversals"], df["morans_I"], "-o", color=ACC2, ms=9, lw=2)
    for _, r in df.iterrows():
        star = " *" if r["perm_p"] < 0.05 else ""
        ax.annotate(f"I={r['morans_I']:.2f}{star}\nn={int(r['streets'])}",
                    (r["min_traversals"], r["morans_I"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=9, color=INK)
    ax.set_xlabel("streets kept  (minimum traversals)")
    ax.set_ylabel("Moran's I  (residual rate)")
    ax.set_xticks(list(thresholds))
    ax.set_ylim(-0.02, max(df["morans_I"]) * 1.25 + 0.02)
    ax.set_title("Spatial autocorrelation rises as noise falls\n"
                 "residual (rate − regime mean) stays clustered → regime doesn't absorb it "
                 "(* p<0.05)", fontsize=11.5, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, "fig_moran_threshold.png")
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
    _write(df, "regime_dispersion.csv")

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
    _save(fig, "fig_regime_dispersion.png")
    print(df.to_string(index=False), f"\n  pooled dispersion = {pooled:.2f}")
    return df


def forest_support_table():
    """Reproduce the exact binning of the Task-4 forest plot and write the
    numbers behind every row: events, exposure, rate, ratio-to-average, 95%
    interval, and a support flag (solid >=100 events, ok 30-100, thin <30)."""
    o = pd.read_csv(ORACLE)
    obs = o[o["is_observed"]].copy()
    tr = pd.read_csv(TRAV)
    ev = pd.read_csv(EVENTS)
    tr["enter_time"] = pd.to_datetime(tr["enter_time"], utc=True, format="mixed")
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    tr["km"] = tr["length_m"] / 1000.0
    overall = obs["n_events"].sum() / obs["rider_km"].sum()

    rows = []

    def add(group, label, n, e):
        n, e = float(n), float(e)
        if n < 5 or e <= 0:
            return
        rate, lo, hi = _poisson_ci(n, e)
        flag = "solid" if n >= 100 else ("ok" if n >= 30 else "THIN")
        rows.append(dict(group=group, label=str(label), events=int(n),
                         rider_km=round(e, 1), rate=round(rate, 3),
                         x_average=round(rate / overall, 2), lo=round(lo, 3),
                         hi=round(hi, 3), support=flag))

    def edge_bins(group, col, bins, labels):
        d = obs.dropna(subset=[col]).copy()
        d["_b"] = pd.cut(d[col], bins, labels=labels)
        g = d.groupby("_b", observed=True).agg(e=("n_events", "sum"), k=("rider_km", "sum"))
        for lab, r in g.iterrows():
            add(group, lab, r["e"], r["k"])

    g = (obs.groupby("edge_class").agg(e=("n_events", "sum"), k=("rider_km", "sum"))
         .sort_values("e", ascending=False))
    for lab, r in g[g["e"] >= 20].iterrows():
        add("riding regime", lab.replace("_", " "), r["e"], r["k"])
    edge_bins("bicycle accidents", "n_acc_bike", [-1, 0, 1, 3, 1000], ["none", "1", "2–3", "4+"])
    edge_bins("speed limit", "maxspeed_kmh", [0, 20, 30, 50, 200], ["<=20", "21-30", "31-50", ">50"])
    edge_bins("betweenness", "betweenness", [-1, 1e-5, 1e-4, 1e-3, 1], ["lowest", "low", "mid", "high"])
    edge_bins("measured AADT", "aadt_kfz", [0, 5000, 12000, 1e6], ["<5k", "5-12k", ">12k"])
    hb, hl = [-1, 6, 9, 15, 19, 24], ["night", "morning peak", "midday", "evening peak", "late"]
    ev_h = pd.cut(ev["time"].dt.hour, hb, labels=hl).value_counts().reindex(hl, fill_value=0)
    km_h = tr.groupby(pd.cut(tr["enter_time"].dt.hour, hb, labels=hl), observed=False)["km"].sum().reindex(hl)
    for lab in hl:
        add("time of day", lab, ev_h[lab], km_h[lab])
    wmap = {False: "weekday", True: "weekend"}
    ev_w = (ev["time"].dt.dayofweek >= 5).map(wmap).value_counts()
    km_w = tr.groupby((tr["enter_time"].dt.dayofweek >= 5).map(wmap))["km"].sum()
    for lab in ["weekday", "weekend"]:
        add("day type", lab, ev_w.get(lab, 0), km_w.get(lab, 0))

    df = pd.DataFrame(rows)
    _write(df, "forest_support_table.csv")
    thin = df[df["support"] == "THIN"]
    print(df.to_string(index=False))
    print(f"\n  overall rate = {overall:.3f} /rider-km")
    print(f"  THIN rows (<30 overtakes, treat with caution): "
          f"{', '.join(thin['group'] + ':' + thin['label']) if len(thin) else 'none'}")
    return df


def main():
    _setup()
    for name, fn in [("RIDER DOMINANCE", rider_dominance),
                     ("REGIME DISPERSION", regime_dispersion),
                     ("TEMPORAL DRIFT (clustered rate-ratio test)", temporal_drift),
                     ("POISSON vs NEGATIVE BINOMIAL", poisson_vs_nb),
                     ("COVARIATE ADJUSTMENT (marginal vs regime-adjusted)", covariate_adjustment),
                     ("COVARIATE CORRELATION HEATMAP", covariate_correlation),
                     ("INTERSECTION ROBUSTNESS (ranking check)", intersection_robustness),
                     ("MORAN'S I BY THRESHOLD", moran_by_threshold),
                     ("FOREST-PLOT SUPPORT TABLE", forest_support_table)]:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        fn()


if __name__ == "__main__":
    main()
