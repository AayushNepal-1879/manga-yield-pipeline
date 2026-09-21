"""
Radial Acceleration Relation (RAR).

The RAR (McGaugh, Lelli & Schombert 2016) is the empirical, remarkably tight
relation between the *observed* centripetal acceleration and the acceleration
predicted from the *baryons* alone:

    g_obs(R) = V_rot(R)^2 / R
    g_bar(R) = G M_bar(<R) / R^2   (spherical)   or the thin-disk expression
    g_obs = g_bar / (1 - exp(-sqrt(g_bar / g_dagger)))     [fitted g_dagger]

We compute both accelerations from the rotation curve and the stellar+gas
surface-density maps, fit the acceleration scale g_dagger, and record the
outer-disk mass discrepancy (g_obs/g_bar = (V_obs/V_bar)^2) and RAR residual.
Those become dynamical variables to correlate against the metal/yield results.

Default g_bar uses the spherical enclosed-mass approximation (robust, common).
A razor-thin exponential-disk option (Freeman 1970, via Bessel functions) is
available and is exact for an exponential disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .config import Config, CONST
from . import geometry
from .utils import kpc_per_arcsec


@dataclass
class RARResult:
    r_kpc: np.ndarray
    g_obs: np.ndarray                # m/s^2
    g_bar: np.ndarray                # m/s^2
    v_bar: np.ndarray                # km/s
    g_dagger: float                  # m/s^2
    g_dagger_err: float
    mass_discrepancy: np.ndarray     # g_obs/g_bar
    rar_residual: np.ndarray         # dex, log g_obs - log RAR(g_bar)
    md_outer: float                  # outer-disk mass discrepancy
    rar_resid_outer: float


def rar_function(g_bar: np.ndarray, g_dagger: float) -> np.ndarray:
    """McGaugh+2016 RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/g_dagger)))."""
    g_bar = np.asarray(g_bar, float)
    x = np.sqrt(np.clip(g_bar, 0, None) / g_dagger)
    with np.errstate(over="ignore", invalid="ignore"):
        denom = 1.0 - np.exp(-x)
    return g_bar / np.where(denom > 1e-12, denom, 1e-12)


def surface_density_profiles(gm, cfg: Config, r_kpc: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Azimuthally-averaged Sigma_star, Sigma_gas (Msun/pc^2) on an r_kpc grid."""
    inc = geometry.inclination_from_ba(gm.ba, cfg.kinematics.q0)
    R_arcsec, _ = geometry.deproject(gm.shape, gm.x0, gm.y0, gm.pa, inc,
                                     gm.spaxel_arcsec)
    kpc_as = kpc_per_arcsec(gm.z, cfg.cosmology)
    r_map_kpc = R_arcsec * kpc_as
    edges = np.concatenate([[0], 0.5 * (r_kpc[:-1] + r_kpc[1:]),
                            [r_kpc[-1] + (r_kpc[-1] - r_kpc[-2])]])

    def profile(m):
        if m is None:
            return np.full(r_kpc.size, np.nan)
        out = np.full(r_kpc.size, np.nan)
        for i in range(r_kpc.size):
            sel = (r_map_kpc >= edges[i]) & (r_map_kpc < edges[i + 1]) & np.isfinite(m)
            if np.count_nonzero(sel) >= 3:
                out[i] = float(np.nanmean(m[sel]))
        return out

    return profile(gm.sigma_star), profile(gm.sigma_gas)


def _vbar_spherical(r_kpc: np.ndarray, sig_tot: np.ndarray) -> np.ndarray:
    """Circular speed (km/s) from spherical enclosed mass of a Sigma profile."""
    r_pc = r_kpc * 1e3
    good = np.isfinite(sig_tot)
    sig = np.where(good, sig_tot, 0.0)
    # cumulative M(<r) = 2π ∫ Σ r dr  (trapezoid on the pc grid)
    integrand = 2.0 * np.pi * sig * r_pc            # Msun/pc
    # concatenate([[0], cumsum(trapezoid)]) is already length len(r_kpc):
    # Menc[0]=0 at r_kpc[0], Menc[i]=M(<r_kpc[i]). Do NOT slice further.
    Menc = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                          * np.diff(r_pc))])  # Msun
    r_m = r_kpc * CONST.kpc_m
    M_kg = Menc * CONST.Msun_kg
    with np.errstate(divide="ignore", invalid="ignore"):
        g = CONST.G * M_kg / r_m ** 2               # m/s^2
        v = np.sqrt(g * r_m) / 1e3                   # km/s
    return v


def _vbar_thindisk(r_kpc: np.ndarray, sig_tot: np.ndarray) -> np.ndarray:
    """Circular speed of a razor-thin exponential disk fit to Sigma (Freeman)."""
    from scipy.special import iv, kv
    good = np.isfinite(sig_tot) & (sig_tot > 0)
    if np.count_nonzero(good) < 3:
        return _vbar_spherical(r_kpc, sig_tot)
    # log-linear fit Sigma = Sigma0 exp(-r/h)
    A = np.vstack([np.ones(good.sum()), r_kpc[good]]).T
    b = np.log(sig_tot[good])
    (lnS0, minv_h), *_ = np.linalg.lstsq(A, b, rcond=None)
    h = -1.0 / minv_h if minv_h < 0 else r_kpc[good].mean()
    Sigma0 = np.exp(lnS0)                            # Msun/pc^2
    y = r_kpc / (2.0 * max(h, 1e-3))
    Sigma0_si = Sigma0 * CONST.Msun_kg / CONST.pc_m ** 2   # kg/m^2
    h_m = h * CONST.kpc_m
    r_m = r_kpc * CONST.kpc_m
    with np.errstate(over="ignore", invalid="ignore"):
        bess = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
        v2 = 4.0 * np.pi * CONST.G * Sigma0_si * h_m * y ** 2 * bess
    v = np.sqrt(np.clip(v2, 0, None)) / 1e3
    return v


def compute_rar(gm, kin, cfg: Config) -> RARResult:
    """Compute the RAR for one galaxy from its rotation curve + Sigma maps."""
    r_kpc = np.asarray(kin.r_kpc, float)
    vrot = np.asarray(kin.vrot, float)
    sig_star, sig_gas = surface_density_profiles(gm, cfg, r_kpc)
    if not cfg.rar.include_gas:
        sig_gas = np.zeros_like(sig_gas)
    sig_tot = np.nansum(np.vstack([np.nan_to_num(sig_star),
                                   np.nan_to_num(sig_gas)]), axis=0)
    sig_tot = np.where((np.isfinite(sig_star)) | (np.isfinite(sig_gas)),
                       sig_tot, np.nan)

    if cfg.rar.gbar_model == "thindisk":
        v_bar = _vbar_thindisk(r_kpc, sig_tot)
    else:
        v_bar = _vbar_spherical(r_kpc, sig_tot)

    r_m = r_kpc * CONST.kpc_m
    with np.errstate(divide="ignore", invalid="ignore"):
        g_obs = (vrot * 1e3) ** 2 / r_m
        g_bar = (v_bar * 1e3) ** 2 / r_m

    # Fit g_dagger in log space.
    g_dagger, g_dagger_err = cfg.rar.gdagger_guess, np.nan
    ok = np.isfinite(g_obs) & np.isfinite(g_bar) & (g_obs > 0) & (g_bar > 0)
    if cfg.rar.fit_gdagger and np.count_nonzero(ok) >= 3:
        from scipy.optimize import curve_fit

        def model(gbar, log_gd):
            return np.log10(rar_function(gbar, 10 ** log_gd))
        try:
            p, cov = curve_fit(model, g_bar[ok], np.log10(g_obs[ok]),
                               p0=[np.log10(cfg.rar.gdagger_guess)],
                               maxfev=10000)
            g_dagger = float(10 ** p[0])
            g_dagger_err = float(np.log(10) * g_dagger * np.sqrt(cov[0, 0]))
        except Exception:
            pass

    with np.errstate(divide="ignore", invalid="ignore"):
        md = g_obs / g_bar
        resid = np.log10(g_obs) - np.log10(rar_function(g_bar, g_dagger))

    outer = r_kpc >= (cfg.kinematics.outer_frac * np.nanmax(r_kpc))
    md_outer = float(np.nanmean(md[outer & ok]))
    resid_outer = float(np.nanmean(resid[outer & ok]))

    return RARResult(
        r_kpc=r_kpc, g_obs=g_obs, g_bar=g_bar, v_bar=v_bar,
        g_dagger=g_dagger, g_dagger_err=g_dagger_err,
        mass_discrepancy=md, rar_residual=resid,
        md_outer=md_outer, rar_resid_outer=resid_outer,
    )
