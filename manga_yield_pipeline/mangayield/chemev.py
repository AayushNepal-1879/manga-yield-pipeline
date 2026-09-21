"""
Chemical-evolution and nucleosynthetic-yield model.

This module connects an *observable* (heavy-metal content + gas fraction) to a
*physical* parameter (the supernova / high-mass IMF slope) via two links:

  1. Effective yield.  In a one-zone chemical-evolution model the gas-phase
     metallicity obeys, in the instantaneous-recycling approximation,

         closed box :  Z_O = y_O * ln(1/mu)
         leaky box  :  Z_O = [y_O / (1 + eta)] * ln(1/mu)

     with mu = gas fraction and eta = mass-loading of a metal-carrying outflow.
     Inverting gives the *effective* yield  y_eff = Z_O / ln(1/mu) = y_O/(1+eta).

  2. IMF-weighted oxygen yield.  The *true* yield y_O is the oxygen mass ejected
     by core-collapse supernovae per unit mass of stars formed, integrated over
     the IMF.  Flattening the high-mass IMF slope alpha raises y_O (more massive
     stars -> more oxygen).  So y_eff constrains y_O(alpha)/(1+eta): alpha and
     eta are degenerate from y_eff alone.  The [alpha/Fe] ratio (link below)
     helps break it because outflows remove O and Fe together and largely cancel
     in the ratio.

  3. [alpha/Fe] as a supernova-timescale/IMF diagnostic.  Oxygen is a pure
     core-collapse (massive-star) product; iron has a large delayed Type-Ia
     contribution.  [O/Fe] therefore rises with a top-heavier IMF (smaller
     alpha) and with a shorter star-formation timescale tau.

The yield tables here are compact, log-log-interpolated approximations to the
published core-collapse grids (Woosley & Weaver 1995; Nomoto et al. 2006).
They are meant to be representative and are trivially swappable — see
docs/METHODOLOGY.md for the caveats. Nothing in this module is a substitute for
a full nucleosynthesis network.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import numpy as np

from .config import YieldCfg


# --------------------------------------------------------------------------
# Representative core-collapse ejecta (Msun) vs progenitor mass (Msun).
# Log-log interpolated; approximations to WW95 / Nomoto+06.
# --------------------------------------------------------------------------
_MASS_GRID = np.array([8, 10, 12, 13, 15, 18, 20, 25, 30, 35, 40, 60, 80, 100.0])
_O_EJECTA = np.array([0.10, 0.15, 0.20, 0.25, 0.40, 0.80, 1.0, 1.9,
                      2.5, 3.3, 4.0, 5.5, 6.5, 7.0])
_FE_EJECTA_CC = np.array([0.05, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09,
                          0.10, 0.10, 0.11, 0.12, 0.13, 0.13])

# Type-Ia iron per event (Msun) and solar O/Fe by mass (for [O/Fe] zero-point).
_M_FE_IA = 0.60
_OFE_SUN_MASS = (10 ** (8.69 - 7.50)) * (16.0 / 56.0)   # ~4.43


def _loglog_interp(m: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray) -> np.ndarray:
    """Interpolate y(m) linearly in log-log, clamped to the grid ends."""
    m = np.atleast_1d(np.asarray(m, float))
    lx, ly = np.log10(xgrid), np.log10(ygrid)
    out = np.interp(np.log10(np.clip(m, xgrid[0], xgrid[-1])), lx, ly)
    return 10.0 ** out


def oxygen_ejecta(m):
    """Oxygen mass (Msun) ejected by a CC-SN of initial mass m (Msun)."""
    return _loglog_interp(m, _MASS_GRID, _O_EJECTA)


def iron_ejecta_cc(m):
    """Iron mass (Msun) from a core-collapse SN of initial mass m."""
    return _loglog_interp(m, _MASS_GRID, _FE_EJECTA_CC)


# --------------------------------------------------------------------------
# IMF: broken power law with a variable high-mass slope alpha.
# --------------------------------------------------------------------------
def imf_number(m: np.ndarray, alpha: float, cfg: YieldCfg) -> np.ndarray:
    """Un-normalised IMF number density dN/dm, continuous across breaks.

    Segments: [0.08,0.5) slope imf_low_slope; [0.5,1.0) slope imf_mid_slope;
    [1.0, m_max] slope alpha (the free "supernova mass distribution" slope).
    Only ratios of IMF integrals are used downstream, so the overall constant
    is irrelevant.
    """
    m = np.atleast_1d(np.asarray(m, float))
    b0, b1, b2 = cfg.imf_break_masses          # (0.08, 0.5, 1.0)
    s0, s1, s2 = cfg.imf_low_slope, cfg.imf_mid_slope, alpha
    C0 = 1.0
    C1 = C0 * b1 ** (s1 - s0)                   # continuity at b1
    C2 = C1 * b2 ** (s2 - s1)                   # continuity at b2
    phi = np.zeros_like(m)
    seg0 = (m >= b0) & (m < b1)
    seg1 = (m >= b1) & (m < b2)
    seg2 = (m >= b2) & (m <= cfg.m_max)
    phi[seg0] = C0 * m[seg0] ** (-s0)
    phi[seg1] = C1 * m[seg1] ** (-s1)
    phi[seg2] = C2 * m[seg2] ** (-s2)
    return phi


def _mass_grid(cfg: YieldCfg, n: int = 400) -> np.ndarray:
    return np.logspace(np.log10(cfg.m_min), np.log10(cfg.m_max), n)


def oxygen_yield(alpha: float, cfg: YieldCfg) -> float:
    """y_O(alpha): oxygen mass produced per unit mass of stars formed.

    y_O = [ int_{sn_min}^{m_collapse} M_O(m) phi(m) dm ]
          / [ int_{m_min}^{m_max}     m       phi(m) dm ]
    Rises as alpha decreases (top-heavier IMF -> more massive stars).
    """
    m = _mass_grid(cfg)
    phi = imf_number(m, alpha, cfg)
    sn = (m >= cfg.sn_mass_min) & (m <= cfg.m_collapse)
    num = np.trapz(np.where(sn, oxygen_ejecta(m), 0.0) * phi, m)
    den = np.trapz(m * phi, m)
    return float(num / den)


def _cc_iron_yield(alpha: float, cfg: YieldCfg) -> float:
    m = _mass_grid(cfg)
    phi = imf_number(m, alpha, cfg)
    sn = (m >= cfg.sn_mass_min) & (m <= cfg.m_collapse)
    num = np.trapz(np.where(sn, iron_ejecta_cc(m), 0.0) * phi, m)
    den = np.trapz(m * phi, m)
    return float(num / den)


def _n_ia_progenitors(alpha: float, cfg: YieldCfg) -> float:
    """Number of Type-Ia progenitors (3-8 Msun) per unit stellar mass formed."""
    m = _mass_grid(cfg)
    phi = imf_number(m, alpha, cfg)
    prog = (m >= 3.0) & (m <= 8.0)
    num = np.trapz(np.where(prog, 1.0, 0.0) * phi, m)
    den = np.trapz(m * phi, m)
    return float(num / den)


@lru_cache(maxsize=1)
def _calibrate_A_ia(cfg_key: tuple) -> float:
    """Set the Ia normalisation so [O/Fe]=0 at (alpha=2.35, tau=3 Gyr)."""
    cfg = YieldCfg(*cfg_key)
    a0, tau0 = 2.35, 3.0
    O = oxygen_yield(a0, cfg)
    Fe_cc = _cc_iron_yield(a0, cfg)
    n_ia = _n_ia_progenitors(a0, cfg)
    # want O/(Fe_cc + A*tau0*n_ia*M_FE_IA) = OFE_SUN_MASS
    fe_total_needed = O / _OFE_SUN_MASS
    fe_ia_needed = max(fe_total_needed - Fe_cc, 1e-6)
    A = fe_ia_needed / (tau0 * n_ia * _M_FE_IA)
    return float(A)


def _cfg_key(cfg: YieldCfg) -> tuple:
    return (cfg.box_model, cfg.imf_low_slope, cfg.imf_mid_slope,
            cfg.imf_break_masses, cfg.m_min, cfg.m_max, cfg.sn_mass_min,
            cfg.m_collapse, cfg.return_fraction)


def forward_alpha_fe(alpha: float, tau_gyr: float, cfg: YieldCfg) -> float:
    """Predicted [alpha/Fe] (using [O/Fe] as proxy) for a given IMF slope and
    star-formation timescale. Increases for top-heavier IMF (smaller alpha) and
    shorter tau. Calibrated to [O/Fe]=0 at (alpha=2.35, tau=3 Gyr)."""
    A = _calibrate_A_ia(_cfg_key(cfg))
    O = oxygen_yield(alpha, cfg)
    Fe_cc = _cc_iron_yield(alpha, cfg)
    n_ia = _n_ia_progenitors(alpha, cfg)
    Fe_ia = A * tau_gyr * n_ia * _M_FE_IA
    ofe_mass = O / (Fe_cc + Fe_ia)
    return float(np.log10(ofe_mass / _OFE_SUN_MASS))


# --------------------------------------------------------------------------
# Effective-yield extraction from observables.
# --------------------------------------------------------------------------
def effective_yield(Z_O: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """y_eff = Z_O / ln(1/mu).  Valid for 0<mu<1; returns NaN otherwise.

    Z_O is the oxygen mass fraction; mu is the gas mass fraction
    M_gas/(M_gas+M_star).
    """
    Z_O = np.asarray(Z_O, float)
    mu = np.asarray(mu, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lninv = np.log(1.0 / mu)
        y = Z_O / lninv
    bad = ~np.isfinite(y) | (mu <= 0) | (mu >= 1) | (lninv <= 0)
    y = np.where(bad, np.nan, y)
    return y


def y_eff_from_alpha_eta(alpha: float, eta: float, cfg: YieldCfg) -> float:
    """Forward model for the effective yield: y_eff = y_O(alpha)/(1+eta)."""
    return oxygen_yield(alpha, cfg) / (1.0 + eta)


# --------------------------------------------------------------------------
# Fast cached interpolants (used by the MCMC in inference.py).
# oxygen_yield / iron / Ia integrals are smooth in alpha, so tabulate once.
# --------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _yield_tables(cfg_key: tuple):
    cfg = YieldCfg(*cfg_key)
    ag = np.linspace(0.8, 3.6, 141)
    O = np.array([oxygen_yield(a, cfg) for a in ag])
    Fecc = np.array([_cc_iron_yield(a, cfg) for a in ag])
    nia = np.array([_n_ia_progenitors(a, cfg) for a in ag])
    return ag, O, Fecc, nia


def oxygen_yield_fast(alpha, cfg: YieldCfg):
    ag, O, _, _ = _yield_tables(_cfg_key(cfg))
    return np.interp(alpha, ag, O)


def y_eff_fast(alpha, eta, cfg: YieldCfg):
    """Vectorised y_eff = y_O(alpha)/(1+eta) via the cached table."""
    return oxygen_yield_fast(alpha, cfg) / (1.0 + np.asarray(eta, float))


def forward_alpha_fe_fast(alpha, tau_gyr, cfg: YieldCfg):
    """Vectorised [alpha/Fe] forward model via cached tables."""
    A = _calibrate_A_ia(_cfg_key(cfg))
    ag, O, Fecc, nia = _yield_tables(_cfg_key(cfg))
    Oa = np.interp(alpha, ag, O)
    Fca = np.interp(alpha, ag, Fecc)
    nia_a = np.interp(alpha, ag, nia)
    Fe_ia = A * np.asarray(tau_gyr, float) * nia_a * _M_FE_IA
    ofe_mass = Oa / (Fca + Fe_ia)
    return np.log10(ofe_mass / _OFE_SUN_MASS)


def alpha_from_y_eff(y_eff_obs: float, eta: float, cfg: YieldCfg,
                     bracket: Tuple[float, float] = (0.8, 3.6)) -> float:
    """Invert y_eff = y_O(alpha)/(1+eta) for alpha at fixed eta (root find)."""
    from scipy.optimize import brentq
    target = y_eff_obs * (1.0 + eta)

    def f(a):
        return oxygen_yield(a, cfg) - target

    lo, hi = bracket
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        # target outside achievable range; clamp to nearest bound.
        return lo if abs(flo) < abs(fhi) else hi
    return float(brentq(f, lo, hi, xtol=1e-4))
