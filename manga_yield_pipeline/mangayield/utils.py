"""
Small numerical utilities: cosmological distances (astropy if available, else a
SciPy fallback), weighted linear fits, and robust helpers. Kept dependency-safe.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple
from .config import CosmologyCfg, CONST


def kpc_per_arcsec(z: float, cosmo: CosmologyCfg) -> float:
    """Physical kpc per arcsec at redshift z for a flat LCDM cosmology.

    Uses astropy.cosmology when importable; otherwise integrates the comoving
    distance directly with SciPy so the pipeline never hard-depends on astropy
    for this one number.
    """
    z = float(z)
    if z <= 0:
        z = 1e-4
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        ac = FlatLambdaCDM(H0=cosmo.H0, Om0=cosmo.Om0)
        return float(ac.kpc_proper_per_arcmin(z).to(u.kpc / u.arcsec).value)
    except Exception:
        pass
    # SciPy fallback: D_A = (c/H0) / (1+z) * int_0^z dz'/E(z')
    from scipy.integrate import quad
    c_km_s = 299792.458
    H0 = cosmo.H0
    Om0, Ode0 = cosmo.Om0, cosmo.Ode0

    def inv_E(zp):
        return 1.0 / np.sqrt(Om0 * (1 + zp) ** 3 + Ode0)

    dc = (c_km_s / H0) * quad(inv_E, 0.0, z)[0]      # comoving distance, Mpc
    d_a = dc / (1.0 + z)                              # angular diameter dist, Mpc
    # radians per arcsec
    rad_per_arcsec = np.pi / (180.0 * 3600.0)
    kpc_per_arcsec_val = d_a * 1000.0 * rad_per_arcsec
    return float(kpc_per_arcsec_val)


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray
                        ) -> Tuple[float, float, float, float]:
    """Weighted least-squares y = a + b x.

    Returns (intercept a, slope b, sigma_a, sigma_b). Weights are 1/variance.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[good], y[good], w[good]
    if x.size < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    delta = S * Sxx - Sx * Sx
    if abs(delta) < 1e-30:
        return (np.nan, np.nan, np.nan, np.nan)
    a = (Sxx * Sy - Sx * Sxy) / delta
    b = (S * Sxy - Sx * Sy) / delta
    sig_a = np.sqrt(Sxx / delta)
    sig_b = np.sqrt(S / delta)
    return (float(a), float(b), float(sig_a), float(sig_b))


def nanmad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to sigma (robust scatter estimate)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def oh_to_ZO(oh: np.ndarray) -> np.ndarray:
    """Convert 12+log(O/H) to oxygen mass fraction Z_O."""
    return CONST.Z_O_sun * 10.0 ** (np.asarray(oh, float) - CONST.OH_sun)


def ZO_to_oh(z_o: np.ndarray) -> np.ndarray:
    """Inverse of oh_to_ZO."""
    z_o = np.asarray(z_o, float)
    return CONST.OH_sun + np.log10(np.clip(z_o, 1e-12, None) / CONST.Z_O_sun)
