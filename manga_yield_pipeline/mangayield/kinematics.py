"""
Kinematics and "orbit levels".

From a MaNGA velocity field we extract an axisymmetric rotation curve by a
ring-by-ring first-order harmonic fit:

    V_los(x,y) = V_sys + V_rot(R) * sin(i) * cos(theta)

where theta is the in-plane azimuth from the major axis and i the inclination.
We solve one global least-squares system for V_sys and the per-ring amplitudes
A_k = V_rot,k * sin(i).

"Orbit levels" are defined by the specific angular momentum j = R * V_rot(R)
(kpc km/s). Angular momentum labels orbits far more robustly than radius alone
(it is approximately conserved under secular evolution / radial migration), so
binning the metal content by j answers "how much metal sits on each orbit."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .config import Config
from . import geometry
from .utils import kpc_per_arcsec


@dataclass
class KinResult:
    r_re: np.ndarray                 # ring centers, R/Re
    r_kpc: np.ndarray                # ring centers, kpc (in-plane)
    vrot: np.ndarray                 # deprojected rotation speed, km/s
    vrot_err: np.ndarray
    sigma: np.ndarray                # velocity dispersion profile, km/s
    jprofile: np.ndarray             # specific angular momentum, kpc km/s
    inclination: float               # radians
    vsys: float                      # km/s
    v_flat: float                    # outer-disk rotation speed, km/s
    v_flat_err: float
    sigma_outer: float               # outer-disk dispersion, km/s
    j_outer: float                   # outer-disk specific angular momentum
    v_over_sigma_outer: float
    kpc_per_re: float
    spaxel_j: np.ndarray = field(default=None)      # per-spaxel j map
    orbit_level: np.ndarray = field(default=None)   # integer orbit-level map
    orbit_level_edges: np.ndarray = field(default=None)

    def vrot_at(self, r_re: np.ndarray) -> np.ndarray:
        """Interpolate the rotation curve to arbitrary R/Re (clamped)."""
        good = np.isfinite(self.vrot)
        if good.sum() < 2:
            return np.full_like(np.asarray(r_re, float), np.nan)
        return np.interp(np.asarray(r_re, float), self.r_re[good], self.vrot[good],
                         left=self.vrot[good][0], right=self.vrot[good][-1])


def _velocity_map(gm, tracer: str):
    if tracer == "gas" and gm.gas_vel is not None:
        return gm.gas_vel, gm.gas_vel_ivar
    return gm.stellar_vel, gm.stellar_vel_ivar


def fit_rotation_curve(gm, cfg: Config) -> KinResult:
    """Ring-by-ring harmonic rotation-curve extraction + orbit levels."""
    kc = cfg.kinematics
    vel, ivar = _velocity_map(gm, kc.tracer)
    if vel is None:
        raise ValueError("no velocity map available for requested tracer")

    inc = geometry.inclination_from_ba(gm.ba, kc.q0)
    sini = max(np.sin(inc), 0.15)
    R_arcsec, theta = geometry.deproject(gm.shape, gm.x0, gm.y0, gm.pa, inc,
                                         gm.spaxel_arcsec)
    r_re_map = R_arcsec / max(gm.Re_arcsec, 1e-3)

    finite = np.isfinite(vel)
    if ivar is not None:
        w = np.where(np.isfinite(ivar), np.clip(ivar, 0, None), 0.0)
    else:
        w = np.ones(gm.shape)
    use = finite & (w > 0) & (r_re_map <= kc.rmax_re)

    # Ring edges in R/Re.
    edges = np.linspace(0.0, kc.rmax_re, kc.n_rings + 1)
    r_re = 0.5 * (edges[:-1] + edges[1:])
    ring_idx = np.digitize(r_re_map, edges) - 1  # 0..n_rings-1

    # Global least squares: unknowns [Vsys, A_0..A_{n-1}], V = Vsys + A_k cos(theta)
    xs = np.where(use)
    ring_of = ring_idx[xs]
    valid_ring = (ring_of >= 0) & (ring_of < kc.n_rings)
    rows = np.where(valid_ring)[0]
    npar = kc.n_rings + 1
    A = np.zeros((rows.size, npar))
    yv = vel[xs][valid_ring]
    wv = np.sqrt(w[xs][valid_ring])
    cth = np.cos(theta[xs][valid_ring])
    A[:, 0] = 1.0
    A[np.arange(rows.size), 1 + ring_of[valid_ring]] = cth
    # weight rows
    Aw = A * wv[:, None]
    yw = yv * wv
    sol, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    vsys = float(sol[0])
    amp = sol[1:]

    # Per-ring counts and amplitude errors (from residual scatter).
    vrot = amp / sini
    vrot_err = np.full(kc.n_rings, np.nan)
    sigma_prof = np.full(kc.n_rings, np.nan)
    npix = np.zeros(kc.n_rings, int)
    resid = yv - A.dot(sol)
    for k in range(kc.n_rings):
        m = ring_of == k
        npix[k] = int(np.count_nonzero(m & valid_ring))
        if npix[k] >= kc.min_spx_per_ring:
            rr = resid[m[valid_ring]] if m.shape == valid_ring.shape else resid
            sc = np.nanstd(resid[ring_of[valid_ring] == k]) if npix[k] > 2 else np.nan
            denom = np.sqrt(max(npix[k], 1)) * sini
            vrot_err[k] = sc / max(denom, 1e-3)
        else:
            vrot[k] = np.nan

    # Velocity dispersion profile (instrument-corrected sigma preferred).
    sig_map = gm.stellar_sigma if gm.stellar_sigma is not None else gm.gas_sigma
    if sig_map is not None:
        for k in range(kc.n_rings):
            sel = use & (ring_idx == k) & np.isfinite(sig_map)
            if np.count_nonzero(sel) >= kc.min_spx_per_ring:
                sigma_prof[k] = float(np.nanmedian(sig_map[sel]))

    # Physical scales.
    kpc_as = kpc_per_arcsec(gm.z, cfg.cosmology)
    kpc_per_re = kpc_as * gm.Re_arcsec
    r_kpc = r_re * kpc_per_re
    jprof = r_kpc * vrot

    # Outer-disk metrics.
    outer = r_re >= (kc.outer_frac * kc.rmax_re)
    v_flat = float(np.nanmean(vrot[outer]))
    v_flat_err = float(np.nanmean(vrot_err[outer]))
    sigma_outer = float(np.nanmean(sigma_prof[outer]))
    j_outer = float(np.nanmax(jprof[np.isfinite(jprof)])) if np.any(np.isfinite(jprof)) else np.nan
    vsig_outer = v_flat / sigma_outer if (sigma_outer and np.isfinite(sigma_outer)) else np.nan

    res = KinResult(
        r_re=r_re, r_kpc=r_kpc, vrot=vrot, vrot_err=vrot_err, sigma=sigma_prof,
        jprofile=jprof, inclination=inc, vsys=vsys, v_flat=v_flat,
        v_flat_err=v_flat_err, sigma_outer=sigma_outer, j_outer=j_outer,
        v_over_sigma_outer=vsig_outer, kpc_per_re=kpc_per_re,
    )

    # Per-spaxel angular momentum and orbit-level assignment.
    vrot_spaxel = res.vrot_at(r_re_map)
    spaxel_j = (r_re_map * kpc_per_re) * vrot_spaxel
    res.spaxel_j = np.where(use, spaxel_j, np.nan)
    finite_j = np.isfinite(res.spaxel_j)
    if finite_j.sum() > kc.n_orbit_levels:
        qs = np.linspace(0, 100, kc.n_orbit_levels + 1)
        j_edges = np.nanpercentile(res.spaxel_j[finite_j], qs)
        j_edges[0] -= 1e-6
        level = np.digitize(res.spaxel_j, j_edges) - 1
        level = np.where(finite_j, np.clip(level, 0, kc.n_orbit_levels - 1), -1)
        res.orbit_level = level
        res.orbit_level_edges = j_edges
    return res


def metallicity_by_orbit_level(metal, kin: KinResult):
    """Mean 12+log(O/H) in each angular-momentum ("orbit") level.

    Returns (level_index, mean_j, mean_oh, oh_err, n) arrays. This is the direct
    "how much metal on each orbit" product linking metals to kinematics.
    """
    if kin.orbit_level is None:
        return (np.array([]),) * 5
    nlev = int(np.nanmax(kin.orbit_level)) + 1
    lev = np.arange(nlev)
    mean_j = np.full(nlev, np.nan)
    mean_oh = np.full(nlev, np.nan)
    oh_err = np.full(nlev, np.nan)
    n = np.zeros(nlev, int)
    oh = metal.oh
    for L in lev:
        sel = (kin.orbit_level == L) & np.isfinite(oh)
        k = int(np.count_nonzero(sel))
        n[L] = k
        if k >= 3:
            mean_j[L] = float(np.nanmean(kin.spaxel_j[sel]))
            mean_oh[L] = float(np.nanmean(oh[sel]))
            oh_err[L] = float(np.nanstd(oh[sel]) / np.sqrt(k))
    return lev, mean_j, mean_oh, oh_err, n
