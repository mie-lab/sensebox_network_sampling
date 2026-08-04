"""Task 5 — three overtake-rate models compared by held-out Poisson deviance.

Each adds one idea over the last, all sharing a log(regime rate) offset:

  1. Poisson-Gamma (EB)   N ~ Poisson(Eλ), λ ~ Gamma          — independent Gamma RE
  2. Poisson-lognormal    N ~ Poisson(E e^(m+γ)), γ ~ N(0,σ²) — independent Gaussian RE
  3. Spatial Poisson GP   γ = f(x), f ~ GP(0, K)              — spatially correlated RE

#2 and #3 are Laplace-approximated Gaussian-latent Poisson models (Rasmussen &
Williams, Alg. 3.1); #2 is the independent (diagonal-kernel) case, #3 adds a Matérn
kernel on edge centroids. So #1→#2 isolates the RE distribution (Gamma vs Gaussian)
and #2→#3 isolates independence vs spatial borrowing. Scored on a train/test ride
split, scanning the GP length-scale.

Run:  python risk_model_gp_task5.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

from risk_model_gamma_task5 import (DIRECTED_EDGE_KEY, load, make_streets, fit_prior,
                                    posterior_rate, poisson_pred_error)

EDGES = Path("input/muenster_edges_classified.gpkg")
OUT = Path("output/task5_risk")
FIG = Path("output/figures")
ACC, ACC2 = "#2a78d6", "#e34948"
N_TRAIN = 800                        # GP training edges (Laplace is O(n^3) per step)
AMPS = [0.25, 0.5, 1.0, 2.0, 4.0]    # signal-variance grid, picked by marginal likelihood
LENGTHS_KM = [0.2, 0.5, 1.0, 2.0]    # GP length-scale scan
RNG = np.random.default_rng(0)
plt.rcParams.update({"font.size": 11, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def centroids():
    """Edge-centroid (x, y) per directed edge — the input space for the kernel."""
    e = gpd.read_file(EDGES)[DIRECTED_EDGE_KEY + ["geometry"]].copy()
    c = e.geometry.centroid
    e["x"], e["y"] = c.x.to_numpy(), c.y.to_numpy()
    return e.drop_duplicates(DIRECTED_EDGE_KEY)[DIRECTED_EDGE_KEY + ["x", "y"]]


def rbf(Xa, Xb, length, amp):
    """Squared-exponential covariance between point sets Xa and Xb."""
    d2 = ((Xa[:, None, :] - Xb[None, :, :]) ** 2).sum(-1)
    return amp * np.exp(-0.5 * d2 / length ** 2)


def matern32(Xa, Xb, length, amp):
    """Matérn-3/2 covariance — rougher than the RBF, a common spatial default."""
    d = np.sqrt(((Xa[:, None, :] - Xb[None, :, :]) ** 2).sum(-1))
    r = np.sqrt(3) * d / length
    return amp * (1 + r) * np.exp(-r)


KERNEL = matern32   # spatial covariance for the GP; swap to rbf for squared-exponential


def laplace_poisson(K, N, logoff, maxit=40, tol=1e-6):
    """Laplace approximation to the Poisson-GP posterior (R&W Alg. 3.1): Newton
    iteration to the mode of f for mu = exp(logoff + f). Returns the pieces needed
    to predict + the approximate log marginal likelihood (to pick hyper-parameters)."""
    f = np.zeros(len(N))
    eye = np.eye(len(N))
    mll_old = -np.inf
    for _ in range(maxit):
        mu = np.exp(np.clip(logoff + f, -20, 20))
        sW = np.sqrt(mu)                         # W = -∇∇ log p(N|f) = mu for a log-link Poisson
        B = eye + sW[:, None] * K * sW[None, :]  # B = I + sqrt(W) K sqrt(W)
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


def gp_rate(Xtr, N_tr, logoff_tr, Xte, m_te, length):
    """Fit the Poisson GP at this length-scale (best amp by marginal likelihood) and
    predict the rate at the test edges."""
    best = None
    for amp in AMPS:
        K = KERNEL(Xtr, Xtr, length, amp) + 1e-6 * np.eye(len(Xtr))   # jitter for a stable Cholesky
        fit = laplace_poisson(K, N_tr, logoff_tr)
        if fit and (best is None or fit["mll"] > best[0]):
            best = (fit["mll"], amp, fit)
    if best is None:
        return None, None

    _, amp, fit = best
    Kts = KERNEL(Xte, Xtr, length, amp)
    fmean = Kts @ fit["a"]
    V = solve_triangular(fit["L"], fit["sW"][:, None] * Kts.T, lower=True)
    fvar = np.clip(amp - (V ** 2).sum(0), 0, None)
    return np.exp(m_te + fmean + 0.5 * fvar), amp     # lognormal mean rate: exp(mean + var/2)


def _laplace_lognormal(N, E, m, s2, iters=50):
    """Per-edge mode γ̂ and posterior variance v for N ~ Poisson(E exp(m - s2/2 + γ)),
    γ ~ N(0, s2). The -s2/2 centering keeps E[rate] at the regime rate."""
    data = E > 0
    o = np.where(data, np.log(np.where(data, E, 1.0)) + m - s2 / 2, 0.0)   # log offset
    g = np.zeros_like(m)
    for _ in range(iters):
        mu = np.where(data, np.exp(np.clip(o + g, -20, 20)), 0.0)
        step = (N - mu - g / s2) / (mu + 1 / s2)                           # Newton direction
        g = g + np.clip(step, -2.0, 2.0)                                   # damped — avoid GLM overshoot
    mu = np.where(data, np.exp(np.clip(o + g, -20, 20)), 0.0)
    return np.where(data, g, 0.0), 1 / (mu + 1 / s2), mu, o


def poisson_lognormal_rate(streets, prior, kcol="km", ncol="n"):
    """Per-edge Poisson-lognormal shrinkage — the Gaussian-RE sibling of EB.
    N ~ Poisson(E exp(m + γ)), γ ~ N(0, σ²_r). σ²_r is set from EB's per-regime
    dispersion (σ² = log(1 + cv²)), so #1 vs #2 isolates only the RE distribution."""
    by, gr = prior.set_index("edge_class"), prior.attrs["global_rate"]
    reg_s2 = np.log1p(by["var_between"] / by["rate"] ** 2)          # rate cv² -> log-space variance
    m = np.log(streets["edge_class"].map(by["rate"]).fillna(gr).to_numpy())
    s2 = streets["edge_class"].map(reg_s2).fillna(reg_s2.median()).to_numpy()
    N, E = streets[ncol].to_numpy(float), streets[kcol].to_numpy(float)

    g, v, _, _ = _laplace_lognormal(N, E, m, s2)
    center = m - s2 / 2 + g
    return pd.DataFrame({"posterior_mean_rate": np.exp(center + 0.5 * v),
                         "lo": np.exp(center - 1.96 * np.sqrt(v)),
                         "hi": np.exp(center + 1.96 * np.sqrt(v))}, index=streets.index)


def _subsample(n):
    return RNG.choice(n, N_TRAIN, replace=False) if n > N_TRAIN else np.arange(n)


def cross_val(tr, ev, cls, cents, frac=0.7, reps=4):
    """Held-out Poisson deviance for the three models: Poisson-Gamma (EB), Poisson-
    lognormal (independent Gaussian RE), and the spatial GP at each length-scale."""
    rides = tr["traj_id"].unique()
    scores = {n: [] for n in ["Poisson-Gamma", "Poisson-lognormal"] + [f"GP {L}km" for L in LENGTHS_KM]}
    for _ in range(reps):
        mask = pd.Series(RNG.random(len(rides)) < frac, index=rides)
        s = make_streets(tr, ev, cls, mask).merge(cents, on=DIRECTED_EDGE_KEY, how="left")
        s = s[s["x"].notna()]
        prior = fit_prior(s, "km_tr", "n_tr")
        reg = prior.set_index("edge_class")["rate"]
        gr = prior.attrs["global_rate"]

        trn, te = s[s["km_tr"] > 0], s[s["km_te"] > 0].copy()
        n_te, km_te = te["n_te"].to_numpy(float), te["km_te"].to_numpy(float)
        dev = lambda rate: poisson_pred_error(n_te, rate * km_te) / len(te)

        # independent random-effect models: same offset, Gamma vs Gaussian RE
        eb = posterior_rate(te, prior, "km_tr", "n_tr")["posterior_mean_rate"].to_numpy()
        pln = poisson_lognormal_rate(te, prior, "km_tr", "n_tr")["posterior_mean_rate"].to_numpy()
        scores["Poisson-Gamma"].append(dev(eb))
        scores["Poisson-lognormal"].append(dev(pln))

        # spatial GP: subsample train edges (Laplace is O(n^3)), scan the length-scale
        x0, y0 = trn["x"].mean(), trn["y"].mean()
        jit = RNG.normal(0, 2.0, (len(trn), 2))          # separate the two directions of a street
        Xtr = np.c_[(trn["x"].to_numpy() - x0 + jit[:, 0]) / 1000,
                    (trn["y"].to_numpy() - y0 + jit[:, 1]) / 1000]
        Xte = np.c_[(te["x"] - x0) / 1000, (te["y"] - y0) / 1000]
        m_tr = np.log(trn["edge_class"].map(reg).fillna(gr).to_numpy())   # log regime rate = GP mean
        m_te = np.log(te["edge_class"].map(reg).fillna(gr).to_numpy())
        logoff = np.log(trn["km_tr"].to_numpy()) + m_tr                   # exposure x regime baseline
        N_tr = trn["n_tr"].to_numpy(float)
        sub = _subsample(len(trn))
        for L in LENGTHS_KM:
            lam, _ = gp_rate(Xtr[sub], N_tr[sub], logoff[sub], Xte, m_te, L)
            scores[f"GP {L}km"].append(np.nan if lam is None else dev(lam))
    return pd.DataFrame({n: [np.nanmean(v), np.nanstd(v)] for n, v in scores.items()},
                        index=["mean_dev", "sd"]).T


def _plain(name):
    labels = {"Poisson-Gamma": "Poisson–Gamma  (independent Gamma RE)",
              "Poisson-lognormal": "Poisson-lognormal  (independent Gaussian RE)"}
    return labels.get(name) or f"Spatial GP  ({name.split()[1]} neighbourhood)"


def fig_cv(perf):
    """Ranked held-out deviance across the three models."""
    o = perf.sort_values("mean_dev")
    best = o["mean_dev"].idxmin()
    base = {"Poisson-Gamma", "Poisson-lognormal"}
    cols = [ACC if n == best else (ACC2 if n in base else "#c9c9c6") for n in o.index]

    fig, ax = plt.subplots(figsize=(8.6, 4))
    ax.barh(range(len(o)), o["mean_dev"], xerr=o["sd"], color=cols, alpha=0.95,
            error_kw=dict(lw=1.2, capsize=3))
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([_plain(i) for i in o.index])
    ax.invert_yaxis()
    ax.set_xlim((o["mean_dev"] - o["sd"]).min() - 0.004, (o["mean_dev"] + o["sd"]).max() + 0.004)
    ax.set_xlabel("held-out Poisson deviance   (lower is better)")
    win = "the spatial GP" if best.startswith("GP") else best
    ax.set_title(f"Gamma vs Gaussian vs spatial random effects — best: {win}",
                 fontsize=11.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gp_cv.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {FIG / 'fig_gp_cv.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tr, ev, cls = load()
    cents = centroids()

    print("=== held-out Poisson deviance: Gamma vs Gaussian vs spatial RE (lower=better) ===")
    perf = cross_val(tr, ev, cls, cents)
    perf.to_csv(OUT / "gp_cv_task5.csv")
    print(perf.round(4).to_string())

    gp_best = perf.filter(like="GP", axis=0)["mean_dev"].idxmin()      # best GP length-scale
    d = perf.loc[gp_best, "mean_dev"] / perf.loc["Poisson-Gamma", "mean_dev"] - 1
    print(f"\nbest: {perf['mean_dev'].idxmin()} | best GP ({gp_best}) is {d:+.1%} vs Poisson-Gamma "
          f"({'better' if d < 0 else 'worse'})")
    fig_cv(perf)


if __name__ == "__main__":
    main()
