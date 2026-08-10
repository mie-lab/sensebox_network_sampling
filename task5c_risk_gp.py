"""Does spatial structure add anything over street type?

A ladder of estimators, each adding one idea, all scored the same way: held-out Poisson
deviance on a train/test split of rides.

  raw            N / E per edge, no pooling
  regime         one rate per street type, full pooling
  Poisson-Gamma  regime plus independent per-edge shrinkage
  spatial L km   regime plus a Matern-3/2 GP on edge centroids, with a per-edge nugget
  spatial-only   the same GP without the regime, a pure location field

The kernel is k(e,e') = amp * Matern32(distance) + nug * 1[same edge]. The nugget is the
Gaussian twin of what Poisson-Gamma does: without it the field could only borrow from
neighbours, so the comparison would be space INSTEAD of per-edge shrinkage rather than
space ON TOP of it. nug = 0 stays in the grid, so the model may still choose to be purely
spatial. Length-scales run from street scale (0.05 km) to district scale (1 km); at the
short end the kernel is nearly diagonal, so that row should land near Poisson-Gamma and
doubles as a sanity check.

Train and test edge sets overlap, so a shared edge's nugget is estimated from the training
rides and must carry into its prediction. That is why the nugget enters the train/test
cross-covariance too, not only the training diagonal (see _shared_edges).

Fitting is by Laplace approximation, which is O(n^3), so the field trains on a subsample of
N_TRAIN edges. ntrain_sweep() checks the conclusion is not an artefact of that cap.

Writes to output/task5_risk/:
  task5c_gp_cv.csv       the ladder, held-out deviance per estimator
  task5c_gp_ntrain.csv   deviance against training subsample size
plus two figures to output/figures/.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial.distance import cdist

from task5b_risk_gamma import (DIRECTED_EDGE_KEY, load, make_streets, fit_prior,
                               posterior_rate, poisson_pred_error)

EDGES = Path("input/muenster_edges_classified.gpkg")
OUT = Path("output/task5_risk")
FIG = Path("output/figures")
BLUE, RED = "blue", "red"           # primary / accent
INK, MUTED_BAR = "black", "lightgrey"   # text / de-emphasised bars
N_TRAIN = 2000                            # GP training edges (exact Laplace is O(n^3); see ntrain_sweep)
NTRAIN_GRID = [1000, 2000, 3000, 4000]    # subsample sizes for the handicap check
# Hyper-parameter grid, jointly picked by marginal likelihood. Re-centred on the last
# run's answer: amp sat on the old floor (0.5) in 37 of 40 fits and nug often on the old
# ceiling (2.0), so both boundaries are extended outward and the never-chosen high amps
# dropped. A pinned optimum is not an estimate.
AMPS = [0.1, 0.25, 0.5, 1.0]              # spatial signal variance
NUGGETS = [0.0, 0.5, 1.0, 2.0, 4.0]       # per-edge variance
# Matern neighbourhood scan (km). Street-scale first: a typical edge is ~100 m, so 0.05
# is "this street only" (K is then near-diagonal — effectively a pure per-edge effect,
# which is why this row doubles as the sanity reference) and 1.0 is "this district".
SPATIAL_LENGTHS = [0.05, 0.1, 0.3, 0.5, 1.0]
SWEEP_LENGTHS = (0.1, 0.3)             # the two the ntrain sweep re-fits at
JITTER = 1e-6                          # keeps the Cholesky stable when nug = 0
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.size": 11, "text.color": INK,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def centroids():
    """Edge-centroid (x, y) per directed edge — the input space for the kernel."""
    e = gpd.read_file(EDGES)[DIRECTED_EDGE_KEY + ["geometry"]].copy()
    c = e.geometry.centroid
    e["x"], e["y"] = c.x.to_numpy(), c.y.to_numpy()
    return e.drop_duplicates(DIRECTED_EDGE_KEY)[DIRECTED_EDGE_KEY + ["x", "y"]]


def coords(df, delta=0.005):
    """GP input per edge: (x, y, direction) in km. Direction = +/-delta from node order
    (u<v is forward), so a street's two directions separate explicitly — no random jitter."""
    d = delta * np.where(df["u"].to_numpy() < df["v"].to_numpy(), 1.0, -1.0)
    return np.c_[df["x"].to_numpy() / 1000, df["y"].to_numpy() / 1000, d]


def matern32(Xa, Xb, length):
    """Matern-3/2 CORRELATION on Euclidean distance — the standard spatial kernel
    (rougher than the RBF). Unit amplitude: the caller scales it, so the same matrix
    is reused across the amp grid. Correlation is 0.48 at exactly one length-scale
    apart. Network distance was tried and gave no gain (see header)."""
    r = np.sqrt(3) * cdist(Xa, Xb) / length
    return (1 + r) * np.exp(-r)


def _shared_edges(te, trs):
    """Row positions (i in te, j in trs) of directed edges that appear in BOTH sets.
    The nugget is a per-EDGE effect, so for a shared edge it is estimated from the
    training rides and must carry into the prediction — it therefore enters the
    cross-covariance at these positions, not only the training diagonal. Miss this and
    the nugget is a mere regulariser: the field would predict its smooth part only."""
    pos = {k: j for j, k in enumerate(zip(trs["u"], trs["v"]))}
    hit = [(i, pos[k]) for i, k in enumerate(zip(te["u"], te["v"])) if k in pos]
    if not hit:
        return np.empty(0, int), np.empty(0, int)
    i, j = map(np.asarray, zip(*hit))
    return i, j


def laplace_poisson(K, N, logoff, maxit=40, tol=1e-6):
    """Laplace approximation to the Poisson-GP posterior (R&W Alg. 3.1): Newton
    iteration to the mode of f for mu = exp(logoff + f). Returns the pieces needed
    to predict + the approximate log marginal likelihood (to pick hyper-parameters)."""
    f = np.zeros(len(N))
    di = np.diag_indices(len(N))
    mll_old = -np.inf
    for _ in range(maxit):
        mu = np.exp(np.clip(logoff + f, -20, 20))
        sW = np.sqrt(mu)                         # W = -∇∇ log p(N|f) = mu for a log-link Poisson
        B = sW[:, None] * K * sW[None, :]        # B = I + sqrt(W) K sqrt(W), diagonal added
        B[di] += 1.0                             # in place — K is 128 MB at n=4000
        try:
            L = cholesky(B, lower=True)
        except np.linalg.LinAlgError:
            return None
        b = mu * f + (N - mu)                     # Newton rhs: W f + ∇log p(N|f)
        z = solve_triangular(L, sW * (K @ b), lower=True)
        z = solve_triangular(L.T, z, lower=False)
        a = b - sW * z
        f = K @ a
        mll = (-0.5 * (a @ f) + (N * (logoff + f) - np.exp(logoff + f)).sum()
               - np.log(np.diag(L)).sum())
        if not np.isfinite(mll):
            return None
        if abs(mll - mll_old) < tol:
            break
        mll_old = mll
    return dict(a=a, sW=sW, L=L, f=f, mll=mll)


def gp_rate(Xtr, N_tr, logoff_tr, Xte, m_te, length, share=None):
    """Fit the Poisson GP at a fixed length-scale and predict the rate at the test edges.

    Kernel:  k(e,e') = amp * Matern32(d) + nug * 1[same edge]
    (amp, nug) are chosen jointly by the Laplace marginal likelihood — the same
    criterion, so 'no nugget' (nug=0) stays available and the choice is made on the
    TRAINING fold only. `share` is the (i, j) row match from _shared_edges: where a
    test edge is also a training edge, its own nugget carries into the prediction.

    Centering: with f ~ N(0, s2) the rate exp(m + f) has MEAN m*exp(s2/2), not m — but
    m_r is the regime's mean rate, so an uncentered offset puts the prior above the
    regime rate (e.g. 1.6x at s2=1) and every data-poor edge reverts to that inflated
    value. Both the offset and the prediction therefore carry -s2/2, so an edge with no
    information lands exactly on m_r. s2 varies per candidate, so it sits in the grid.

    Returns (rate at the test edges, chosen (amp, nug)) or None if no fit was stable."""
    base = matern32(Xtr, Xtr, length)                 # unit amplitude, reused across the grid
    di = np.diag_indices(len(Xtr))
    best = None
    for amp in AMPS:
        K = amp * base
        for nug in NUGGETS:
            s2 = amp + nug
            K[di] = amp + nug + JITTER                # rewrite the diagonal, no new matrix
            fit = laplace_poisson(K, N_tr, logoff_tr - s2 / 2)
            if fit and (best is None or fit["mll"] > best[0]):
                best = (fit["mll"], amp, nug, fit)
    if best is None:
        return None
    _, amp, nug, fit = best
    s2 = amp + nug

    Kts = amp * matern32(Xte, Xtr, length)
    kss = np.full(len(Xte), s2 + JITTER)              # prior var of (spatial + own nugget)
    if share is not None and len(share[0]):
        Kts[share] += nug                            # shared edge -> its nugget is observed
    fmean = Kts @ fit["a"]
    V = solve_triangular(fit["L"], fit["sW"][:, None] * Kts.T, lower=True)
    fvar = np.clip(kss - (V ** 2).sum(0), 0, None)
    # lognormal mean rate; an edge with no information lands exactly on m_te
    return np.exp(m_te - s2 / 2 + fmean + 0.5 * fvar), (amp, nug)


def _subsample(n):
    return RNG.choice(n, N_TRAIN, replace=False) if n > N_TRAIN else np.arange(n)


def cross_val(tr, ev, cls, cents, frac=0.7, reps=12):
    """Held-out Poisson deviance for the model ladder. The spatial rows put a Matern GP
    on the regime residual; 'spatial-only' drops the regime, so a pure location field has
    to reconstruct the type signal on its own."""
    rides = tr["traj_id"].unique()
    names = (["raw", "regime", "Poisson-Gamma"]
             + [f"spatial {L}km" for L in SPATIAL_LENGTHS] + ["spatial-only"])
    scores = {n: [] for n in names}
    picked = {L: [] for L in SPATIAL_LENGTHS}          # (amp, nug) the mll chose, per fold
    for _ in range(reps):
        mask = pd.Series(RNG.random(len(rides)) < frac, index=rides)
        s = make_streets(tr, ev, cls, mask).merge(cents, on=DIRECTED_EDGE_KEY, how="left")
        s = s[s["x"].notna()]
        prior = fit_prior(s, "hr_tr", "n_tr")
        reg = prior.set_index("edge_class")["rate"]
        gr = prior.attrs["global_rate"]
        trn, te = s[s["hr_tr"] > 0].copy(), s[s["hr_te"] > 0].copy()
        n_te, hr_te = te["n_te"].to_numpy(float), te["hr_te"].to_numpy(float)
        dev = lambda rate: poisson_pred_error(n_te, rate * hr_te) / len(te)

        # baselines — the same estimators the Poisson-Gamma script compares
        train_hr = te["hr_tr"].to_numpy(float)
        scores["raw"].append(dev(np.where(train_hr > 0, te["n_tr"] / train_hr, gr)))
        scores["regime"].append(dev(te["edge_class"].map(reg).fillna(gr).to_numpy()))
        scores["Poisson-Gamma"].append(dev(posterior_rate(te, prior, "hr_tr", "n_tr")["posterior_mean_rate"].to_numpy()))

        # spatial GP on subsampled train edges (Laplace is O(n^3))
        sub = _subsample(len(trn))
        trs = trn.iloc[sub]
        Xtr, Xte = coords(trs), coords(te)
        share = _shared_edges(te, trs)                     # where the nugget carries over
        N_tr = trs["n_tr"].to_numpy(float)
        loghr = np.log(trs["hr_tr"].to_numpy())
        m_tr = np.log(trs["edge_class"].map(reg).fillna(gr).to_numpy())    # log regime rate
        m_te = np.log(te["edge_class"].map(reg).fillna(gr).to_numpy())

        for L in SPATIAL_LENGTHS:              # regime mean + spatial field + per-edge nugget
            out = gp_rate(Xtr, N_tr, loghr + m_tr, Xte, m_te, L, share)
            scores[f"spatial {L}km"].append(np.nan if out is None else dev(out[0]))
            if out is not None:
                picked[L].append(out[1])
        # no regime: a pure location field (widest length-scale) must recover the type map
        out0 = gp_rate(Xtr, N_tr, loghr + np.log(gr), Xte, np.full(len(te), np.log(gr)),
                       SPATIAL_LENGTHS[-1], share)
        scores["spatial-only"].append(np.nan if out0 is None else dev(out0[0]))
    perf = pd.DataFrame({n: [np.nanmean(v), np.nanstd(v)] for n, v in scores.items()},
                        index=["mean_dev", "sd"]).T
    # which (amp, nug) the marginal likelihood settled on — nug=0 everywhere would mean
    # the nugget is not earning its place and the plain Matern kernel is the whole model
    perf.attrs["picked"] = {L: pd.Series(v).value_counts().to_dict() for L, v in picked.items()}
    return perf


def _plain(name):
    labels = {"regime": "street-type mean",
              "Poisson-Gamma": "Poisson-Gamma  (Gamma per-edge shrinkage)",
              "spatial-only": "spatial only  (pure geography, no street type)"}
    return labels.get(name) or f"street type + spatial GP  (reach {name.split()[1]})"


def fig_cv(perf):
    """Ranked held-out deviance across the ladder (raw is off-scale, shown in the table)."""
    o = perf.drop("raw").sort_values("mean_dev")
    best = o["mean_dev"].idxmin()
    re = {"Poisson-Gamma"}
    cols = [BLUE if n == best else (RED if n in re else MUTED_BAR) for n in o.index]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.barh(range(len(o)), o["mean_dev"], xerr=o["sd"], color=cols, alpha=0.95,
            error_kw=dict(lw=1.2, capsize=3))
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([_plain(i) for i in o.index])
    ax.invert_yaxis()
    ax.set_xlim((o["mean_dev"] - o["sd"]).min() - 0.004, (o["mean_dev"] + o["sd"]).max() + 0.004)
    ax.set_xlabel("held-out Poisson deviance   (lower is better)")
    ax.set_title("What each modelling layer adds — best: " + _plain(best), fontsize=11.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "task5c_gp_cv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'task5c_gp_cv.png'}")


def ntrain_sweep(tr, ev, cls, cents, frac=0.7, reps=6):
    """Held-out deviance of the spatial GP as the training subsample grows — the check
    that N_TRAIN is not what handicaps it. If the curve levels off well above EB, more
    data (e.g. a sparse GP) would not close the gap; the signal simply isn't there."""
    rides = tr["traj_id"].unique()
    spatial = {N: [] for N in NTRAIN_GRID}
    regime, eb = [], []
    for _ in range(reps):
        mask = pd.Series(RNG.random(len(rides)) < frac, index=rides)
        s = make_streets(tr, ev, cls, mask).merge(cents, on=DIRECTED_EDGE_KEY, how="left")
        s = s[s["x"].notna()]
        prior = fit_prior(s, "hr_tr", "n_tr")
        reg = prior.set_index("edge_class")["rate"]
        gr = prior.attrs["global_rate"]
        trn, te = s[s["hr_tr"] > 0].copy(), s[s["hr_te"] > 0].copy()
        n_te, hr_te = te["n_te"].to_numpy(float), te["hr_te"].to_numpy(float)
        dev = lambda rate: poisson_pred_error(n_te, rate * hr_te) / len(te)
        regime.append(dev(te["edge_class"].map(reg).fillna(gr).to_numpy()))
        eb.append(dev(posterior_rate(te, prior, "hr_tr", "n_tr")["posterior_mean_rate"].to_numpy()))

        Xte = coords(te)
        m_te = np.log(te["edge_class"].map(reg).fillna(gr).to_numpy())
        for N in NTRAIN_GRID:
            sub = RNG.choice(len(trn), min(N, len(trn)), replace=False)
            trs = trn.iloc[sub]
            share = _shared_edges(te, trs)
            N_tr = trs["n_tr"].to_numpy(float)
            logoff = np.log(trs["hr_tr"].to_numpy()) + np.log(trs["edge_class"].map(reg).fillna(gr).to_numpy())
            best = [dev(out[0]) for L in SWEEP_LENGTHS
                    if (out := gp_rate(coords(trs), N_tr, logoff, Xte, m_te, L, share)) is not None]
            spatial[N].append(min(best) if best else np.nan)
    out = pd.DataFrame({"N_train": NTRAIN_GRID,
                        "spatial": [np.nanmean(spatial[N]) for N in NTRAIN_GRID],
                        # carried in the table too, not only in attrs/the figure
                        "poisson_gamma": np.mean(eb), "regime": np.mean(regime)})
    out.attrs["regime"], out.attrs["eb"] = np.mean(regime), np.mean(eb)
    return out


def fig_ntrain(sweep):
    """Spatial-GP deviance vs training size, against the (size-independent) baselines."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(sweep["N_train"], sweep["spatial"], "-o", color=BLUE, lw=2, ms=7, label="spatial GP")
    ax.axhline(sweep.attrs["eb"], color=RED, ls="--", lw=1.5, label="Poisson-Gamma (EB)")
    ax.axhline(sweep.attrs["regime"], color="0.5", ls=":", lw=1.5, label="street-type mean")
    ax.set_xlabel("GP training edges  (N_train)")
    ax.set_ylabel("held-out Poisson deviance")
    ax.set_title("Does more data rescue the spatial GP?", loc="left")
    ax.legend(frameon=False, fontsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "task5c_gp_ntrain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'task5c_gp_ntrain.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tr, ev, cls = load()
    cents = centroids()

    print("=== held-out Poisson deviance: model ladder (lower=better) ===")
    perf = cross_val(tr, ev, cls, cents)
    perf.to_csv(OUT / "task5c_gp_cv.csv")
    print(perf.round(4).to_string())

    d = perf["mean_dev"]
    spatial_rows = [i for i in d.index if i.startswith("spatial ")]
    best_name = d[spatial_rows].idxmin()
    best_spatial, eb = d[best_name], d["Poisson-Gamma"]
    print(f"\npool by type:       raw {d['raw']:.3f} -> regime {d['regime']:.3f}")
    print(f"per-edge shrinkage: regime {d['regime']:.3f} -> Poisson-Gamma {eb:.3f}")
    print(f"add space:          best {best_name} {best_spatial:.3f} vs Poisson-Gamma {eb:.3f} "
          f"({'helps' if best_spatial < eb else 'adds nothing'})")
    print(f"location vs type:   spatial-only {d['spatial-only']:.3f} vs regime {d['regime']:.3f}")

    print("\n  hyper-parameters the marginal likelihood chose  (amp, nug) -> folds:")
    for L, counts in perf.attrs["picked"].items():
        print(f"    {L:>5} km : " + "  ".join(f"{k}x{v}" for k, v in counts.items()))
    print("    nug=0 throughout would mean the nugget earns nothing and the kernel is "
          "plain Matern.")
    # the shortest length-scale makes K near-diagonal, i.e. a nearly pure per-edge effect,
    # so it should land close to Poisson-Gamma. If it does not, suspect the fit.
    if d[f"spatial {SPATIAL_LENGTHS[0]}km"] > eb + 0.01:
        print(f"  [check] at {SPATIAL_LENGTHS[0]} km the kernel is nearly diagonal — a per-edge "
              f"effect much like Poisson-Gamma — so it should not score materially worse.")
    fig_cv(perf)

    print("\n=== handicap check: spatial-GP deviance vs training size ===")
    sweep = ntrain_sweep(tr, ev, cls, cents)
    sweep.to_csv(OUT / "task5c_gp_ntrain.csv", index=False)
    print(sweep.round(4).to_string(index=False))
    steps = ", ".join(f"{s:+.3f}" for s in np.diff(sweep["spatial"].to_numpy()))
    print(f"EB {sweep.attrs['eb']:.3f} | regime {sweep.attrs['regime']:.3f} | spatial "
          f"{sweep['spatial'].iloc[0]:.3f} -> {sweep['spatial'].iloc[-1]:.3f} over "
          f"N=800..{NTRAIN_GRID[-1]}, step sizes {steps}.")
    print("  Read the STEP SIZES, not the total drop: shrinking steps mean the curve is "
          "levelling off, i.e. the subsample is not what limits the field. A still-steep "
          "last step would mean the cap is the binding constraint and a sparse GP is needed.")
    fig_ntrain(sweep)


if __name__ == "__main__":
    main()
