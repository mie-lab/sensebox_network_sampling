"""Task 5 — Empirical-Bayes Poisson-Gamma overtake risk model.
Estimate the overtake rate for each directed street segment (u, v).

Notation
--------
N : observed overtakes
E : rider-km (exposure)
λ : true overtake rate (per rider-km)
m : mean rate for a street type (regime)
a_r, b_r : Gamma prior parameters for regime r

Model
-----
    N | λ ~ Poisson(λE)
    λ ~ Gamma(a_r, b_r)

    => λ | N, E ~ Gamma(a_r + N, b_r + E)

Posterior mean:
    λ̂ = (N + a_r) / (E + b_r)

The regime prior is estimated from the training data by method of moments.
Prediction quality is evaluated out of sample by train/test ride splits using held-out Poisson deviance.
"""

from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import mean_poisson_deviance

BASE = Path(".")
ORACLE = BASE / "output/task4_oracle/edge_oracle_task4.csv"
TRAV = BASE / "output/task4_oracle/edge_traversals_task4.csv"
EVENTS = BASE / "output/task4_oracle/edge_events_task4.csv"
OUT = BASE / "output/task5_risk"
FIG = BASE / "output/figures"

# Prior-estimation safeguards.
VAR_FLOOR = 1e-4  # keep between-edge variance positive
MIN_EVENTS = 20   # thinner regimes borrow the global pooled dispersion  # TODO revisit this assumption
MIN_BETA = 1.0    # min prior strength (rider-km); avoids over-weak priors
MAX_BETA = 10.0   # max prior strength (rider-km); avoids near-delta priors

# ======= street data ========
DIRECTED_EDGE_KEY = ["u", "v"]


def load():
    """Read the three tables from oracle:
    traversals (one ride crossing one edge, with its rider-km),
    events (one overtake each),
    and the street type of each edge."""

    o = pd.read_csv(ORACLE, usecols=DIRECTED_EDGE_KEY + ["edge_class"])
    edge_t = o.drop_duplicates(DIRECTED_EDGE_KEY)
    tr = pd.read_csv(TRAV, usecols=DIRECTED_EDGE_KEY + ["traj_id", "length_m"])
    tr["km"] = tr["length_m"] / 1000.0
    ev = pd.read_csv(EVENTS, usecols=DIRECTED_EDGE_KEY + ["traj_id"])
    return tr, ev, edge_t


def make_streets(tr, ev, cls, mask_tr=None):
    """One row per directed edge: exposure E (rider-km) and count N (overtakes).
    With mask_tr (a per-ride train/test flag)."""

    def agg(t, e, suff):
        km = t.groupby(DIRECTED_EDGE_KEY)["km"].sum().rename("km" + suff)   # E
        n = e.groupby(DIRECTED_EDGE_KEY).size().rename("n" + suff)          # N
        return km, n

    if mask_tr is None:
        km, n = agg(tr, ev, "")
        s = pd.concat([km, n], axis=1).fillna({"n": 0})
    else:
        tr = tr.assign(_t=tr["traj_id"].map(mask_tr))
        ev = ev.assign(_t=ev["traj_id"].map(mask_tr))

        # used in cross_val()
        parts = (*agg(tr[tr._t], ev[ev._t], "_tr"),           # train rides
                 *agg(tr[~tr._t], ev[~ev._t], "_te"))         # test rides
        s = pd.concat(parts, axis=1).fillna(0)
    return s.reset_index().merge(cls, on=DIRECTED_EDGE_KEY, how="left")


# ======== prior/posterior ==========
def _mom_var(N, E, m):
    """Method-of-moments between-edge variance of the rate."""
    return ((N - m * E) ** 2 - m * E).sum() / (E ** 2).sum()


def fit_prior(streets, kcol="km", ncol="n"):
    """Per-regime Gamma(a_r, b_r) prior by method of moments. Thin regimes fall back
    on the pooled dispersion; b_r is clipped to [MIN_BETA, MAX_BETA]."""
    obs = streets[(streets[kcol] > 0) & streets["edge_class"].notna()]
    global_rate = obs[ncol].sum() / obs[kcol].sum()

    stats = []
    for regime, g in obs.groupby("edge_class"):
        N, E = g[ncol].to_numpy(float), g[kcol].to_numpy(float)
        mean = N.sum() / E.sum()
        stats.append((regime, len(g), int(N.sum()), mean, _mom_var(N, E, mean)))

    # dispersion the thin regimes borrow: median cv² over the ones with enough events
    cv2 = [var / mean**2 for _, _, n, mean, var in stats if n >= MIN_EVENTS and var > VAR_FLOOR]
    pooled_cv2 = np.median(cv2) if cv2 else 1.0

    rows = []
    for regime, n_edges, n_events, mean, var in stats:
        if mean <= 0:                       # ridden, but never overtaken
            mean, var = global_rate, np.nan
        thin = n_events < MIN_EVENTS or not np.isfinite(var) or var <= VAR_FLOOR
        if thin:
            var = max(pooled_cv2 * mean**2, VAR_FLOOR)
        b_r = np.clip(mean / var, MIN_BETA, MAX_BETA)
        a_r = mean * b_r
        rows.append(dict(edge_class=regime, n_edges=n_edges, n_events=n_events,
                         rate=mean, var_between=var, alpha_r=a_r, beta_r=b_r,
                         borrowed_dispersion=thin))

    prior = pd.DataFrame(rows)
    prior.attrs["global_rate"] = global_rate
    prior.attrs["pooled_cv2"] = pooled_cv2
    return prior


def posterior_rate(streets, prior, kcol="km", ncol="n"):
    """Shrink each edge's raw rate toward its street-type prior, weighted by exposure"""

    by_regime = prior.set_index("edge_class")
    global_rate = prior.attrs.get("global_rate", by_regime["rate"].median())

    # edges of a never-ridden street type have no prior: pool to the global rate
    # at a typical strength (median b_r) instead of a hard delta
    b_unseen = by_regime["beta_r"].median()
    a = streets["edge_class"].map(by_regime["alpha_r"]).fillna(global_rate * b_unseen).to_numpy()
    b = streets["edge_class"].map(by_regime["beta_r"]).fillna(b_unseen).to_numpy()
    N = streets[ncol].to_numpy(float)
    E = streets[kcol].to_numpy(float)

    # posterior params
    a_post, b_post = a + N, b + E

    return pd.DataFrame({
        "posterior_mean_rate": a_post / b_post,
        "lo": gamma_dist.ppf(0.025, a=a_post, scale=1 / b_post),
        "hi": gamma_dist.ppf(0.975, a=a_post, scale=1 / b_post),
        "prior_weight": b / (E + b)}, index=streets.index) # share coming from the prior


# ========= performance ===========

def poisson_pred_error(y_true, y_pred):
    """Total held-out Poisson deviance; divide by the edge count for the per-edge mean."""
    y_pred = np.maximum(np.asarray(y_pred, float), 1e-12)
    return mean_poisson_deviance(np.asarray(y_true, float), y_pred) * len(y_true)


def cross_val(tr, ev, cls, frac=0.7, reps=20):
    """Evaluate rate estimators on held-out rides.
    Lower mean Poisson deviance indicates better prediction."""

    rides = tr["traj_id"].unique()
    rng = np.random.default_rng(0)
    scores = {name: [] for name in ["raw", "regime", "eb_posterior_mean_rate"]}

    for _ in range(reps):
        train_mask = pd.Series(rng.random(len(rides)) < frac, index=rides)
        streets = make_streets(tr, ev, cls, mask_tr=train_mask)
        test = streets[streets["km_te"] > 0].copy()
        prior = fit_prior(streets, "km_tr", "n_tr")
        global_rate = prior.attrs["global_rate"]
        regime_rate = (prior.set_index("edge_class")["rate"])
        train_km = test["km_tr"].to_numpy(float)

        # different methods of estimating the rate for each edge in the test set
        rates = {
            "raw": np.where(train_km > 0, test["n_tr"] / train_km, global_rate),
            "regime": (test["edge_class"].map(regime_rate).fillna(global_rate).to_numpy()),
            "eb_posterior_mean_rate": posterior_rate(test, prior, "km_tr", "n_tr")["posterior_mean_rate"].to_numpy()
        }
        # ground truth
        observed = test["n_te"].to_numpy(float)  # N_test
        test_km = test["km_te"].to_numpy(float)  # E_test

        for name, rate in rates.items():
            scores[name].append(poisson_pred_error(observed, rate * test_km) / len(observed))

    result = {name: {"mean_dev": np.mean(v), "sd": np.std(v)} for name, v in scores.items()}
    perf = pd.DataFrame(result).T
    perf.attrs["scores"] = {k: np.asarray(v) for k, v in scores.items()}  # per-split, for paired tests
    return perf


# ========= figures ==========

def fig_uncertainty(streets, eb):
    """Histogram of each ridden edge's 95% posterior interval width, against the
    median rate — the intervals are several times wider than the rate itself."""
    obs = streets[streets["km"] > 0]
    width = (eb.loc[obs.index, "hi"] - eb.loc[obs.index, "lo"]).to_numpy()
    rate = eb.loc[obs.index, "posterior_mean_rate"].to_numpy()
    med, rate_med = np.median(width), np.median(rate)
    hi = np.percentile(width, 99)   # a few edges run much wider -> pile them in the last bin

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.clip(width, 0, hi), bins=50, alpha=0.85)
    ax.axvline(med, color="red", lw=1.5, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f"  median width {med:.2f}", color="red", fontsize=9)
    ax.axvline(rate_med, color="0.4", lw=1.2, ls=":")
    ax.text(rate_med, ax.get_ylim()[1] * 0.75, f"  median posterior width {rate_med:.2f}", color="0.4", fontsize=9)
    ax.set_xlim(0, hi)
    ax.set_xlabel("95% posterior interval width (per rider-km)")
    ax.set_ylabel("number of edges")
    ax.set_title("Per-edge rate uncertainty")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eb_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'fig_eb_uncertainty.png'}")


def fig_shrinkage(streets, eb, prior):
    """Raw per-edge rate (N/E) vs the EB posterior rate, dot size ∝ exposure:
    low-exposure edges collapse to the type mean, well-ridden ones stay near y=x."""
    obs = streets[(streets["km"] > 0) & streets["edge_class"].notna()]
    raw = (obs["n"] / obs["km"]).to_numpy()
    post = eb.loc[obs.index, "posterior_mean_rate"].to_numpy()
    E = obs["km"].to_numpy()

    regimes = list(prior["edge_class"])
    palette = plt.colormaps["tab20"].resampled(len(regimes))
    colors = {r: palette(i) for i, r in enumerate(regimes)}
    size = np.clip(2 + 7 * np.sqrt(E / np.median(E)), 2, 60)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(raw, post, s=size, alpha=0.4, edgecolors="none",
               c=obs["edge_class"].map(colors).tolist())
    lim = float(np.percentile(raw, 98)) or 1.0
    ax.plot([0, lim], [0, lim], "--", lw=1, color="0.3")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, float(np.percentile(post, 99)) * 1.05)
    ax.set_xlabel("raw edge rate  N / E   (per rider-km)")
    ax.set_ylabel("Empirical-Bayes posterior rate")
    ax.set_title("Shrinkage toward the street-type rate")
    ax.text(0.98, 0.03, "dot size ∝ exposure", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.5")

    handles = [Line2D([], [], marker="o", ls="", color=colors[r], label=r.replace("_", " ")) for r in regimes]
    handles.append(Line2D([], [], ls="--", color="0.3", label="no shrinkage (y=x)"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_eb_shrinkage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'fig_eb_shrinkage.png'}")


def fig_regime_caterpillar(prior):
    """Overtake rate per street type with its 95% credible interval, sorted by rate.
    Which types are riskier, and how well each one is pinned down."""
    reg = prior.copy()
    reg["km"] = reg["n_events"] / reg["rate"]        # recover exposure
    reg["n"] = reg["n_events"]
    post = posterior_rate(reg, prior)                # regime-level posterior via the model
    reg["rate_eb"] = post["posterior_mean_rate"]
    reg["lo"], reg["hi"] = post["lo"], post["hi"]
    reg = reg.sort_values("rate_eb")
    gr = prior.attrs["global_rate"]
    y = np.arange(len(reg))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hlines(y, reg["lo"], reg["hi"], color="#2a78d6", lw=2, alpha=0.6)
    ax.plot(reg["rate_eb"], y, "o", color="#2a78d6", ms=6)
    ax.axvline(gr, color="0.5", ls=":", lw=1)
    ax.text(gr, len(reg) - 0.4, f" overall {gr:.2f}", color="0.5", fontsize=8, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.replace('_', ' ')}  (n={n})"
                        for r, n in zip(reg["edge_class"], reg["n_events"])], fontsize=9)
    ax.set_xlabel("overtake rate  (per rider-km, 95% credible interval)")
    ax.set_title("Overtake rate by street type")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_regime_caterpillar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'fig_regime_caterpillar.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tr, ev, cls = load()

    # out-of-sample: raw / regime / EB, scored by held-out Poisson deviance
    perf = cross_val(tr, ev, cls)
    perf.to_csv(OUT / "cv_performance_task5.csv")
    print("=== held-out Poisson deviance (per edge, lower=better) ===")
    print(perf.round(4).to_string())

    sc = perf.attrs["scores"]
    eb_arr = sc["eb_posterior_mean_rate"]
    for base in ("raw", "regime"):
        diff = sc[base] - eb_arr
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        z = diff.mean() / se if se else np.inf
        print(f"  EB vs {base:<6}: {diff.mean() / perf.loc[base, 'mean_dev']:+.1%} deviance "
              f"| wins {(diff > 0).mean():.0%} of {len(diff)} | z={z:.1f}{'*' if abs(z) >= 2 else ''}")

    # full fit: per-edge risk surface + fitted per-type prior
    streets = make_streets(tr, ev, cls)
    prior = fit_prior(streets)
    eb = posterior_rate(streets, prior)
    out = pd.concat([streets, eb], axis=1)
    out.to_csv(OUT / "edge_risk_task5.csv", index=False)
    prior.to_csv(OUT / "regime_prior_task5.csv", index=False)

    print("\n=== fitted priors by street type ===")
    show = prior[["edge_class", "n_edges", "n_events", "rate", "beta_r", "borrowed_dispersion"]]
    print(show.rename(
        columns={"rate": "rate/km",
                 "beta_r": "type_strength",
                 "borrowed_dispersion": "global_pooled_disp"})
          .sort_values("n_events", ascending=False).round(3).to_string(index=False))

    obs = out[out["km"] > 0]
    print(f"\n{len(obs)} edges | global {prior.attrs['global_rate']:.3f}/km | pooled type cv² {prior.attrs['pooled_cv2']:.2f} "
          f"| prior-dominated {(obs['prior_weight'] > 0.9).mean():.0%} "
          f"| median exposure {obs['km'].median():.2f} km | median CI {(obs['hi'] - obs['lo']).median():.2f}/km")

    fig_shrinkage(streets, eb, prior)
    fig_uncertainty(streets, eb)
    fig_regime_caterpillar(prior)


if __name__ == "__main__":
    main()
