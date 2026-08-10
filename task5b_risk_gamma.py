"""Task 5 — Empirical-Bayes Poisson-Gamma overtake risk model.
Estimate the overtake rate for each directed street segment (u, v).

Notation
--------
N : observed overtakes
E : rider-hours (exposure)
λ : true overtake rate (per rider-hour)
m : mean rate for a street type (regime)
a_r, b_r : Gamma prior parameters for regime r

Model
-----
    N | λ ~ Poisson(λE)
    λ ~ Gamma(a_r, b_r)

    => λ | N, E ~ Gamma(a_r + N, b_r + E)

Posterior mean:
    λ̂ = (N + a_r) / (E + b_r)

Exposure is rider-hours: overtaking is a temporal arrival process, and time-at-risk was
the best offset of the three tested (traversals / rider-km / rider-hours) on held-out
Poisson deviance — see cross_val_exposure() below. task5a_diagnostics.exposure_unit_choice()
reaches the same answer from in-sample AIC, dispersion and a proportionality test, and
carries the mechanism: rate_per_km = rate_per_hour / speed, so a per-km rate is inflated
wherever riders are slow rather than where they are in danger.

The regime prior is estimated from the training data by method of moments.
Prediction quality is evaluated out of sample by train/test ride splits using held-out Poisson deviance.
"""

from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

BASE = Path(".")
ORACLE = BASE / "output/task4_oracle/task4_edge_oracle.csv"
TRAV = BASE / "output/task4_oracle/task4_edge_traversals.csv"
EVENTS = BASE / "output/task4_oracle/task4_edge_events.csv"
OUT = BASE / "output/task5_risk"
FIG = BASE / "output/figures"

# Prior-estimation safeguards, all on the Gamma SHAPE a_r = 1/cv², which is dimensionless.
# (b_r carries the exposure unit, so clipping it would silently re-tune the model whenever
# the unit changes: 1.0 is a weak prior in rider-km but swamps every edge in rider-hours.)
MIN_CV2 = 1e-4    # below this the spread is degenerate -> borrow the pooled dispersion
MIN_EVENTS = 20   # thinner regimes borrow the global pooled dispersion  # TODO revisit this assumption
MIN_SHAPE = 0.1   # cv² <= 10; guards against a uselessly diffuse prior
MAX_SHAPE = 20.0  # cv² >= 0.05; guards against a near-delta prior with fake-narrow intervals

# ======= street data ========
DIRECTED_EDGE_KEY = ["u", "v"]


def load():
    """Read the three tables from oracle:
    traversals (one ride crossing one edge, with its rider-hours),
    events (one overtake each),
    and the street type of each edge."""

    o = pd.read_csv(ORACLE, usecols=DIRECTED_EDGE_KEY + ["edge_class"])
    edge_t = o.drop_duplicates(DIRECTED_EDGE_KEY)
    tr = pd.read_csv(TRAV, usecols=DIRECTED_EDGE_KEY + ["traj_id", "on_edge_s", "length_m"])
    tr["hr"] = tr["on_edge_s"] / 3600.0
    tr["km"] = tr["length_m"] / 1000.0      # the runner-up offset, kept for cross_val_exposure
    ev = pd.read_csv(EVENTS, usecols=DIRECTED_EDGE_KEY + ["traj_id"])
    return tr, ev, edge_t


def make_streets(tr, ev, cls, mask_tr=None):
    """One row per directed edge: exposure E (rider-hours) and count N (overtakes).
    With mask_tr (a per-ride train/test flag)."""

    def agg(t, e, suff):
        hr = t.groupby(DIRECTED_EDGE_KEY)["hr"].sum().rename("hr" + suff)   # E
        n = e.groupby(DIRECTED_EDGE_KEY).size().rename("n" + suff)          # N
        return hr, n

    if mask_tr is None:
        hr, n = agg(tr, ev, "")
        s = pd.concat([hr, n], axis=1).fillna({"n": 0})
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


def fit_prior(streets, ecol="hr", ncol="n"):
    """Per-regime Gamma(a_r, b_r) prior by method of moments. Thin regimes fall back
    on the pooled dispersion; the shape a_r is clipped to [MIN_SHAPE, MAX_SHAPE]."""
    obs = streets[(streets[ecol] > 0) & streets["edge_class"].notna()]
    global_rate = obs[ncol].sum() / obs[ecol].sum()

    stats = []
    for regime, g in obs.groupby("edge_class"):
        N, E = g[ncol].to_numpy(float), g[ecol].to_numpy(float)
        mean = N.sum() / E.sum()
        stats.append((regime, len(g), int(N.sum()), mean, _mom_var(N, E, mean)))

    # dispersion the thin regimes borrow: median cv² over the ones with enough events
    cv2 = [var / mean**2 for _, _, n, mean, var in stats
           if n >= MIN_EVENTS and mean > 0 and var > MIN_CV2 * mean**2]
    pooled_cv2 = np.median(cv2) if cv2 else 1.0

    rows = []
    for regime, n_edges, n_events, mean, var in stats:
        if mean <= 0:                       # ridden, but never overtaken
            mean, var = global_rate, np.nan
        thin = n_events < MIN_EVENTS or not np.isfinite(var) or var <= MIN_CV2 * mean**2
        if thin:
            var = pooled_cv2 * mean**2
        a_r = np.clip(mean**2 / var, MIN_SHAPE, MAX_SHAPE)   # shape = 1/cv², unit-free
        b_r = a_r / mean                                     # strength, in exposure units
        rows.append(dict(edge_class=regime, n_edges=n_edges, n_events=n_events,
                         rate=mean, var_between=var, alpha_r=a_r, beta_r=b_r,
                         borrowed_dispersion=thin))

    prior = pd.DataFrame(rows)
    prior.attrs["global_rate"] = global_rate
    prior.attrs["pooled_cv2"] = pooled_cv2
    return prior


def posterior_rate(streets, prior, ecol="hr", ncol="n"):
    """Shrink each edge's raw rate toward its street-type prior, weighted by exposure"""

    by_regime = prior.set_index("edge_class")
    global_rate = prior.attrs.get("global_rate", by_regime["rate"].median())

    # edges of a never-ridden street type have no prior: pool to the global rate
    # at a typical strength (median b_r) instead of a hard delta
    b_unseen = by_regime["beta_r"].median()
    a = streets["edge_class"].map(by_regime["alpha_r"]).fillna(global_rate * b_unseen).to_numpy()
    b = streets["edge_class"].map(by_regime["beta_r"]).fillna(b_unseen).to_numpy()
    N = streets[ncol].to_numpy(float)
    E = streets[ecol].to_numpy(float)

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
    Lower mean Poisson deviance indicates better prediction. """

    rides = tr["traj_id"].unique()
    rng = np.random.default_rng(0)

    # Track both Deviance and MAE scores separately
    dev_scores = {name: [] for name in ["raw", "regime", "eb_posterior_mean_rate"]}
    mae_scores = {name: [] for name in ["raw", "regime", "eb_posterior_mean_rate"]}

    for _ in range(reps):
        train_mask = pd.Series(rng.random(len(rides)) < frac, index=rides)
        streets = make_streets(tr, ev, cls, mask_tr=train_mask)
        test = streets[streets["hr_te"] > 0].copy()
        prior = fit_prior(streets, "hr_tr", "n_tr")
        global_rate = prior.attrs["global_rate"]
        regime_rate = (prior.set_index("edge_class")["rate"])
        train_hr = test["hr_tr"].to_numpy(float)

        # different methods of estimating the rate for each edge in the test set
        rates = {
            "raw": np.where(train_hr > 0, test["n_tr"] / train_hr, global_rate),
            "regime": (test["edge_class"].map(regime_rate).fillna(global_rate).to_numpy()),
            "eb_posterior_mean_rate": posterior_rate(test, prior, "hr_tr", "n_tr")["posterior_mean_rate"].to_numpy()
        }

        # ground truth
        observed = test["n_te"].to_numpy(float)  # N_test
        test_hr = test["hr_te"].to_numpy(float)  # E_test

        for name, rate in rates.items():
            expected_overtakes = rate * test_hr
            dev_scores[name].append(poisson_pred_error(observed, expected_overtakes) / len(observed))
            mae_scores[name].append(mean_absolute_error(observed, expected_overtakes))

    result = {
        name: {
            "mean_dev": np.mean(dev_scores[name]),
            "sd_dev": np.std(dev_scores[name]),
            "mean_mae": np.mean(mae_scores[name]),
            "sd_mae": np.std(mae_scores[name])
        }
        for name in dev_scores
    }

    perf = pd.DataFrame(result).T
    # Keep dev_scores in attributes for the paired z-tests later in main()
    perf.attrs["scores"] = {k: np.asarray(v) for k, v in dev_scores.items()}
    return perf


EXPOSURE_UNITS = {"per traversal": "trav", "per rider-km": "km", "per rider-hour": "hr"}


def _streets_all_units(tr, ev, cls, mask_tr):
    """make_streets, but carrying ALL THREE candidate exposures at once — so every
    unit is scored on the same edges and the same held-out counts."""

    def agg(t, suff):
        g = t.groupby(DIRECTED_EDGE_KEY)
        return pd.DataFrame({f"trav{suff}": g.size().astype(float),
                             f"km{suff}": g["km"].sum(), f"hr{suff}": g["hr"].sum()})

    t = tr.assign(_t=tr["traj_id"].map(mask_tr))
    e = ev.assign(_t=ev["traj_id"].map(mask_tr))
    expo = pd.concat([agg(t[t._t], "_tr"), agg(t[~t._t], "_te")], axis=1)
    n = pd.concat([e[e._t].groupby(DIRECTED_EDGE_KEY).size().rename("n_tr"),
                   e[~e._t].groupby(DIRECTED_EDGE_KEY).size().rename("n_te")], axis=1)
    s = expo.join(n, how="outer").fillna(0.0).reset_index()
    return s.merge(cls, on=DIRECTED_EDGE_KEY, how="left")


def cross_val_exposure(tr, ev, cls, frac=0.7, reps=20):
    """Which exposure unit belongs in the Poisson offset?

    Hold the estimator fixed (EB) and vary only the denominator. The observed counts are
    identical under all three units and the test edges are the same set, so the held-out
    deviances are directly comparable — the lowest one predicts overtake counts best.
    This is the out-of-sample half of the argument; task5a_diagnostics.exposure_unit_choice()
    is the in-sample half (AIC, dispersion, proportionality) and agrees."""
    rides = tr["traj_id"].unique()
    rng = np.random.default_rng(0)
    scores = {lab: [] for lab in EXPOSURE_UNITS}
    for _ in range(reps):
        mask = pd.Series(rng.random(len(rides)) < frac, index=rides)
        s = _streets_all_units(tr, ev, cls, mask)
        test = s[s["trav_te"] > 0].copy()          # same test edges for every unit
        n_te = test["n_te"].to_numpy(float)
        for lab, u in EXPOSURE_UNITS.items():
            prior = fit_prior(s, f"{u}_tr", "n_tr")
            eb = posterior_rate(test, prior, f"{u}_tr", "n_tr")["posterior_mean_rate"].to_numpy()
            scores[lab].append(
                poisson_pred_error(n_te, eb * test[f"{u}_te"].to_numpy(float)) / len(test))
    perf = pd.DataFrame({lab: [np.mean(v), np.std(v)] for lab, v in scores.items()},
                        index=["mean_dev", "sd"]).T
    perf.attrs["scores"] = {k: np.asarray(v) for k, v in scores.items()}
    return perf


# ========= figures ==========

def fig_uncertainty(streets, eb):
    """Histogram of each ridden edge's 95% posterior interval width, against the
    median rate — the intervals are several times wider than the rate itself."""
    obs = streets[streets["hr"] > 0]
    width = (eb.loc[obs.index, "hi"] - eb.loc[obs.index, "lo"]).to_numpy()
    rate = eb.loc[obs.index, "posterior_mean_rate"].to_numpy()
    med, rate_med = np.median(width), np.median(rate)
    hi = np.percentile(width, 99)   # a few edges run much wider -> pile them in the last bin

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.clip(width, 0, hi), bins=50, alpha=0.85)
    ax.axvline(med, color="red", lw=1.5, ls="--")
    ax.text(med, ax.get_ylim()[1] * 0.92, f"  median width {med:.1f}", color="red", fontsize=9)
    ax.axvline(rate_med, color="0.4", lw=1.2, ls=":")
    ax.text(rate_med, ax.get_ylim()[1] * 0.75, f"  median posterior rate {rate_med:.1f}", color="0.4", fontsize=9)
    ax.set_xlim(0, hi)
    ax.set_xlabel("95% posterior interval width (per rider-hour)")
    ax.set_ylabel("number of edges")
    ax.set_title("Per-edge rate uncertainty")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "task5b_eb_uncertainty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'task5b_eb_uncertainty.png'}")


def fig_shrinkage(streets, eb, prior):
    """Raw per-edge rate (N/E) vs the EB posterior rate, dot size ∝ exposure:
    low-exposure edges collapse to the type mean, well-ridden ones stay near y=x."""
    obs = streets[(streets["hr"] > 0) & streets["edge_class"].notna()]
    raw = (obs["n"] / obs["hr"]).to_numpy()
    post = eb.loc[obs.index, "posterior_mean_rate"].to_numpy()
    E = obs["hr"].to_numpy()

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
    ax.set_xlabel("raw edge rate  N / E   (per rider-hour)")
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
    fig.savefig(FIG / "task5b_eb_shrinkage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'task5b_eb_shrinkage.png'}")


def fig_regime_caterpillar(prior):
    """Overtake rate per street type with its 95% credible interval, sorted by rate.
    Which types are riskier, and how well each one is pinned down.
    Types under MIN_EVENTS are dropped: their interval spans the whole axis and squashes
    every real regime into a corner (bus_lane, n=3, runs to ~88/h)."""
    reg = prior[prior["n_events"] >= MIN_EVENTS].copy()
    dropped = prior[prior["n_events"] < MIN_EVENTS]
    reg["hr"] = reg["n_events"] / reg["rate"]        # recover exposure
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
    ax.text(gr, len(reg) - 0.4, f" overall {gr:.1f}", color="0.5", fontsize=8, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.replace('_', ' ')}  (n={n})"
                        for r, n in zip(reg["edge_class"], reg["n_events"])], fontsize=9)
    ax.set_xlabel("overtake rate  (per rider-hour, 95% credible interval)")
    ax.set_title("Overtake rate by street type")
    if len(dropped):
        ax.text(0.98, 0.03, "excluded (<%d overtakes): %s" % (
            MIN_EVENTS, ", ".join(f"{r.edge_class.replace('_', ' ')} n={r.n_events}"
                                  for r in dropped.itertuples())),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.5")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "task5b_regime_caterpillar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'task5b_regime_caterpillar.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tr, ev, cls = load()

    # out-of-sample: raw / regime / EB, scored by held-out Poisson deviance
    perf = cross_val(tr, ev, cls)
    perf.to_csv(OUT / "task5b_cv_performance.csv")
    print("=== Cross-Validation Performance (per edge, lower=better) ===")
    print(perf.round(4).to_string())

    sc = perf.attrs["scores"]
    eb_arr = sc["eb_posterior_mean_rate"]
    for base in ("raw", "regime"):
        diff = sc[base] - eb_arr
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        z = diff.mean() / se if se else np.inf
        print(f"  EB vs {base:<6}: {diff.mean() / perf.loc[base, 'mean_dev']:+.1%} deviance "
              f"| wins {(diff > 0).mean():.0%} of {len(diff)} | z={z:.1f}{'*' if abs(z) >= 2 else ''}")

    # the offset itself: same estimator, same test edges, three denominators
    expo = cross_val_exposure(tr, ev, cls)
    expo.to_csv(OUT / "task5b_exposure_units.csv")
    print("\n=== exposure unit (EB, held-out Poisson deviance, lower=better) ===")
    print(expo.round(4).to_string())
    best = expo["mean_dev"].idxmin()
    esc = expo.attrs["scores"]
    for lab in EXPOSURE_UNITS:
        if lab == best:
            continue
        diff = esc[lab] - esc[best]                      # >0 => best is lower
        z = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        print(f"  {best} vs {lab:<14}: {1 - expo.loc[best, 'mean_dev'] / expo.loc[lab, 'mean_dev']:+.1%} "
              f"deviance | paired z={z:.1f}")

    # full fit: per-edge risk surface + fitted per-type prior
    streets = make_streets(tr, ev, cls)
    prior = fit_prior(streets)
    eb = posterior_rate(streets, prior)
    out = pd.concat([streets, eb], axis=1)
    out.to_csv(OUT / "task5b_edge_risk.csv", index=False)
    prior.to_csv(OUT / "task5b_regime_prior.csv", index=False)

    print("\n=== fitted priors by street type ===")
    show = prior[["edge_class", "n_edges", "n_events", "rate", "alpha_r", "borrowed_dispersion"]]
    print(show.rename(
        columns={"rate": "rate/hr",
                 "alpha_r": "type_shape",
                 "borrowed_dispersion": "global_pooled_disp"})
          .sort_values("n_events", ascending=False).round(3).to_string(index=False))

    obs = out[out["hr"] > 0]
    print(f"\n{len(obs)} edges | global {prior.attrs['global_rate']:.2f}/hr | pooled type cv² {prior.attrs['pooled_cv2']:.2f} "
          f"| prior-dominated {(obs['prior_weight'] > 0.9).mean():.0%} "
          f"| median exposure {obs['hr'].median() * 60:.1f} min | median CI {(obs['hi'] - obs['lo']).median():.1f}/hr")

    fig_shrinkage(streets, eb, prior)
    fig_uncertainty(streets, eb)
    fig_regime_caterpillar(prior)


if __name__ == "__main__":
    main()
