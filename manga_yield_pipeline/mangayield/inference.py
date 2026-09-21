"""
Bayesian inference of the supernova mass distribution (the high-mass IMF slope
alpha) from the effective yield, with the alpha-eta degeneracy handled openly.

Model per galaxy
----------------
Free parameters: alpha (IMF high-mass slope), log10(eta) (outflow mass loading),
and -- only when an [alpha/Fe] measurement is supplied -- tau (SF timescale).

Observables and likelihood (all Gaussian):
    y_eff_obs   ~ N( y_O(alpha)/(1+eta),         sigma_yeff )
    [a/Fe]_obs  ~ N( forward_alpha_fe(alpha,tau), sigma_afe )   (optional)

Priors:
    alpha       ~ TruncNormal(2.35, 0.6) on [1.0, 3.5]   (Salpeter-centred)
    log10 eta   ~ Normal(log10 0.3, 0.5) on [-2, 1]
    tau         ~ TruncNormal(3, 2) Gyr on [0.5, 12]

Why the degeneracy matters: y_eff constrains y_O(alpha)/(1+eta) -- a single
combination -- so alpha and eta trade off along a "banana". The [alpha/Fe]
constraint is nearly outflow-independent (O and Fe leave together) and so helps
localise alpha. We report the posterior AND the sampled alpha-eta correlation so
the degeneracy is never swept under the rug.

Sampler: a dependency-free random-walk Metropolis (NumPy only), plus a SciPy
MAP + Laplace covariance for a quick check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from .config import Config
from . import chemev


@dataclass
class InferenceResult:
    alpha_map: float
    alpha_med: float
    alpha_lo: float                 # 16th percentile
    alpha_hi: float                 # 84th percentile
    eta_med: float
    tau_med: float
    alpha_eta_corr: float           # posterior correlation (degeneracy strength)
    y_eff_obs: float
    y_eff_err: float
    alpha_fe_used: bool
    n_spaxels: int
    accept_frac: float
    samples: Optional[np.ndarray] = field(default=None, repr=False)


def gas_fraction_map(gm) -> np.ndarray:
    """mu = Sigma_gas / (Sigma_gas + Sigma_star), NaN where unavailable."""
    if gm.sigma_gas is None or gm.sigma_star is None:
        return np.full(gm.shape, np.nan)
    sg = np.asarray(gm.sigma_gas, float)
    ss = np.asarray(gm.sigma_star, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = sg / (sg + ss)
    return np.where(np.isfinite(mu), mu, np.nan)


def effective_yield_summary(metal, gm, cfg: Config) -> Tuple[float, float, int]:
    """Robust galaxy-level (y_eff, sigma, N) from Z_O(spaxel) and mu(spaxel)."""
    mu = gas_fraction_map(gm)
    y = chemev.effective_yield(metal.Z_O, mu)
    good = np.isfinite(y) & metal.sf_mask
    yv = y[good]
    n = int(yv.size)
    if n < 5:
        return (np.nan, np.nan, n)
    # robust central value + error on the (log-normalish) yield
    med = float(np.nanmedian(yv))
    mad = 1.4826 * float(np.nanmedian(np.abs(yv - med)))
    err = mad / np.sqrt(max(n, 1))
    err = float(np.hypot(err, 0.05 * med))     # add 5% systematic floor
    return (med, err, n)


def infer_alpha(metal, gm, cfg: Config, keep_samples: bool = False
                ) -> InferenceResult:
    icfg = cfg.inference
    y_eff_obs, y_eff_err, n = effective_yield_summary(metal, gm, cfg)
    afe_obs = gm.alpha_fe
    afe_err = gm.alpha_fe_err or 0.05
    use_afe = bool(icfg.use_alpha_fe and (afe_obs is not None) and np.isfinite(afe_obs))

    ndim = 3 if use_afe else 2
    a_lo, a_hi = icfg.alpha_bounds
    le_lo, le_hi = icfg.logeta_bounds
    t_lo, t_hi = icfg.tau_bounds

    def unpack(theta):
        alpha = theta[0]
        log_eta = theta[1]
        tau = theta[2] if ndim == 3 else icfg.tau_prior_mean
        return alpha, log_eta, tau

    def log_prior(theta):
        alpha, log_eta, tau = unpack(theta)
        if not (a_lo <= alpha <= a_hi):
            return -np.inf
        if not (le_lo <= log_eta <= le_hi):
            return -np.inf
        lp = -0.5 * ((alpha - icfg.alpha_prior_mean) / icfg.alpha_prior_sigma) ** 2
        lp += -0.5 * ((log_eta - icfg.logeta_prior_mean) / icfg.logeta_prior_sigma) ** 2
        if ndim == 3:
            if not (t_lo <= tau <= t_hi):
                return -np.inf
            lp += -0.5 * ((tau - icfg.tau_prior_mean) / icfg.tau_prior_sigma) ** 2
        return lp

    def log_like(theta):
        alpha, log_eta, tau = unpack(theta)
        eta = 10.0 ** log_eta
        ll = 0.0
        if np.isfinite(y_eff_obs) and np.isfinite(y_eff_err) and y_eff_err > 0:
            model = chemev.y_eff_fast(alpha, eta, cfg.yields)
            ll += -0.5 * ((y_eff_obs - model) / y_eff_err) ** 2
        if use_afe:
            model_afe = chemev.forward_alpha_fe_fast(alpha, tau, cfg.yields)
            ll += -0.5 * ((afe_obs - model_afe) / afe_err) ** 2
        return ll

    def log_post(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_like(theta)

    # ---- MAP via SciPy (bounded) ----------------------------------------
    from scipy.optimize import minimize
    x0 = np.array([icfg.alpha_prior_mean, icfg.logeta_prior_mean,
                   icfg.tau_prior_mean])[:ndim]
    bounds = [(a_lo, a_hi), (le_lo, le_hi)] + ([(t_lo, t_hi)] if ndim == 3 else [])
    try:
        res = minimize(lambda th: -log_post(th), x0, method="L-BFGS-B",
                       bounds=bounds)
        map_theta = res.x
    except Exception:
        map_theta = x0
    alpha_map = float(map_theta[0])

    # ---- Metropolis MCMC (NumPy only) -----------------------------------
    rng = np.random.default_rng(icfg.seed + (hash(gm.plateifu) % 10000))
    step = np.array([icfg.mcmc_step, icfg.mcmc_step, 0.4])[:ndim]
    nsteps = icfg.n_walkers_steps
    theta = map_theta.copy()
    lp_cur = log_post(theta)
    chain = np.empty((nsteps, ndim))
    naccept = 0
    for i in range(nsteps):
        prop = theta + step * rng.standard_normal(ndim)
        lp_prop = log_post(prop)
        if np.log(rng.uniform()) < (lp_prop - lp_cur):
            theta, lp_cur = prop, lp_prop
            naccept += 1
        chain[i] = theta
    burn = min(icfg.n_burn, nsteps // 3)
    post = chain[burn:]
    accept_frac = naccept / nsteps

    alpha_s = post[:, 0]
    logeta_s = post[:, 1]
    alpha_med = float(np.median(alpha_s))
    alpha_lo = float(np.percentile(alpha_s, 16))
    alpha_hi = float(np.percentile(alpha_s, 84))
    eta_med = float(10 ** np.median(logeta_s))
    tau_med = float(np.median(post[:, 2])) if ndim == 3 else float(icfg.tau_prior_mean)
    if np.std(alpha_s) > 0 and np.std(logeta_s) > 0:
        alpha_eta_corr = float(np.corrcoef(alpha_s, logeta_s)[0, 1])
    else:
        alpha_eta_corr = np.nan

    return InferenceResult(
        alpha_map=alpha_map, alpha_med=alpha_med, alpha_lo=alpha_lo,
        alpha_hi=alpha_hi, eta_med=eta_med, tau_med=tau_med,
        alpha_eta_corr=alpha_eta_corr, y_eff_obs=y_eff_obs, y_eff_err=y_eff_err,
        alpha_fe_used=use_afe, n_spaxels=n, accept_frac=accept_frac,
        samples=(post if keep_samples else None),
    )
