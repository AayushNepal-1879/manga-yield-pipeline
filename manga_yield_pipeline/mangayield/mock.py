"""
Synthetic MaNGA-like data generator.

Each mock galaxy is built *forward* from a set of injected truths so the full
pipeline can be validated against ground truth:

  * alpha_true  -> oxygen yield y_O(alpha) (chemev)
  * eta_true    -> effective yield y_eff = y_O/(1+eta)
  * a gas-fraction profile mu(R) -> Z_O(R) = y_eff ln(1/mu) -> 12+log(O/H)(R)
  * O/H(R)      -> emission-line ratios (consistent with PP04/Marino) -> line maps
  * Sigma_*(R), Sigma_gas(R) from mu(R)
  * an injected RAR acceleration scale g_dagger_true -> rotation curve V(R)
  * tau_true    -> [alpha/Fe] via the chemev forward model

Population trends are injected on purpose (e.g. top-heavier IMF in more massive,
faster-rotating disks) so the statistics module can be checked for its ability
to *recover* a known correlation. These injected trends are a test harness, not
a claim about real galaxies.
"""
from __future__ import annotations

from typing import Optional, List
import numpy as np

from .config import Config, CONST
from . import geometry, chemev
from .ingest import GalaxyMaps, CANONICAL_LINES
from .utils import kpc_per_arcsec, oh_to_ZO, ZO_to_oh


# --------------------------------------------------------------------------
# Truth sampling
# --------------------------------------------------------------------------
def _draw_truth(rng: np.random.Generator, cfg: Config,
                logmstar: Optional[float] = None) -> dict:
    if logmstar is None:
        logmstar = rng.uniform(9.3, 11.2)
    # Stellar-mass Tully-Fisher-ish outer rotation speed.
    v_flat = 10 ** (0.27 * (logmstar - 10.5) + 2.2) * rng.lognormal(0, 0.05)
    # Injected IMF trend: top-heavier (smaller alpha) in more massive disks.
    alpha_true = float(np.clip(2.45 - 0.33 * (logmstar - 10.3)
                               + rng.normal(0, 0.08), 1.5, 3.1))
    # Outflow loading: stronger in low-mass galaxies.
    eta_true = float(np.clip(0.55 * (10.8 - logmstar) + rng.normal(0, 0.1),
                             0.05, 3.0))
    tau_true = float(np.clip(rng.uniform(1.5, 6.0)
                             + 0.4 * (logmstar - 10.3), 0.6, 10.0))
    # Central O/H from a mass-metallicity relation (sets the gas-fraction norm).
    # Ceiling ~8.62: the O3N2 strong-line calibration saturates near 8.6-8.7,
    # and keeping the center below this keeps every spaxel inside the Kauffmann
    # star-forming region (whose asymptote at log[NII]/Ha=0.05 would otherwise
    # mask metal-rich centers).
    oh0_target = float(np.clip(8.50 + 0.24 * (logmstar - 10.0)
                               + rng.normal(0, 0.03), 8.30, 8.62))
    mu_slope = float(rng.uniform(0.05, 0.11))           # gas fraction rises outward
    g_dagger = float(1.2e-10 * rng.lognormal(0, 0.06))  # ~universal + small scatter

    # Geometry / observing.
    ba = float(rng.uniform(0.40, 0.85))
    pa = float(rng.uniform(0.0, 180.0))
    z = float(rng.uniform(0.02, 0.06))
    Re_arcsec = float(rng.uniform(4.0, 8.0))
    sigma0 = float(np.clip(0.30 * v_flat + rng.normal(0, 8), 30, 220))

    return dict(logmstar=logmstar, v_flat=v_flat, alpha_true=alpha_true,
                eta_true=eta_true, tau_true=tau_true, oh0_target=oh0_target,
                mu_slope=mu_slope, g_dagger_true=g_dagger, ba=ba, pa=pa, z=z,
                Re_arcsec=Re_arcsec, sigma0=sigma0)


# --------------------------------------------------------------------------
# Line synthesis from O/H
# --------------------------------------------------------------------------
def _line_ratios_from_oh(oh: np.ndarray):
    """Return intrinsic NII/Ha and OIII/Hb consistent with PP04 (N2 & O3N2)."""
    N2 = (oh - 8.90) / 0.57                      # PP04_N2 inverse
    nii_ha = 10 ** N2
    o3n2 = (8.73 - oh) / 0.32                     # PP04_O3N2 inverse
    oiii_hb = 10 ** o3n2 * nii_ha                 # since O3N2 = log((OIII/Hb)/(NII/Ha))
    return nii_ha, oiii_hb


def make_mock_galaxy(plateifu: str = "0000-00000",
                     cfg: Optional[Config] = None,
                     seed: Optional[int] = None,
                     logmstar: Optional[float] = None,
                     truth: Optional[dict] = None) -> GalaxyMaps:
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    t = truth or _draw_truth(rng, cfg, logmstar=logmstar)

    # ---- geometry / grid -------------------------------------------------
    Re = t["Re_arcsec"]
    half = int(np.clip(round(2.4 * Re / cfg_spaxel(cfg)), 20, 36))
    npix = 2 * half + 1
    shape = (npix, npix)
    x0 = y0 = half
    inc = geometry.inclination_from_ba(t["ba"], cfg.kinematics.q0)
    R_arcsec, theta = geometry.deproject(shape, x0, y0, t["pa"], inc, cfg_spaxel(cfg))
    r_re = R_arcsec / Re
    kpc_as = kpc_per_arcsec(t["z"], cfg.cosmology)
    r_kpc = R_arcsec * kpc_as

    # ---- chemistry: y_eff, mu(R), O/H(R) --------------------------------
    y_O = chemev.oxygen_yield(t["alpha_true"], cfg.yields)
    y_eff_true = y_O / (1.0 + t["eta_true"])
    Z_O0 = oh_to_ZO(t["oh0_target"])
    mu_center = float(np.clip(np.exp(-Z_O0 / y_eff_true), 0.03, 0.95))
    mu = np.clip(mu_center + t["mu_slope"] * r_re, 0.03, 0.85)
    Z_O = y_eff_true * np.log(1.0 / mu)
    oh = ZO_to_oh(Z_O)
    oh += rng.normal(0, 0.02, size=shape)          # small intrinsic scatter
    # Pure numerical guard only. With the mu-range and oh0 ceiling above, O/H
    # stays ~[8.0, 8.65] by construction (the gradient is y_eff-independent),
    # so this clip does not trigger in normal operation and self-consistency
    # (Z_O = y_eff * ln(1/mu)) is preserved on every star-forming spaxel.
    oh = np.clip(oh, 7.60, 8.80)

    # measured truth gradient (linear fit over R<2Re)
    fit = (r_re <= 2.0)
    A = np.vstack([np.ones(fit.sum()), r_re[fit]]).T
    coef, *_ = np.linalg.lstsq(A, oh[fit], rcond=None)
    oh0_true, grad_true = float(coef[0]), float(coef[1])

    # ---- surface densities ----------------------------------------------
    h_star_arcsec = Re / 1.678
    h_star_pc = h_star_arcsec * kpc_as * 1e3
    Mstar = 10 ** t["logmstar"]
    Sigma0_star = Mstar / (2 * np.pi * h_star_pc ** 2)      # Msun/pc^2
    sigma_star = Sigma0_star * np.exp(-R_arcsec / h_star_arcsec)
    sigma_gas = sigma_star * mu / (1.0 - mu)

    # ---- rotation curve from the injected RAR ---------------------------
    vrot_map = _rar_rotation(r_kpc, R_arcsec, h_star_arcsec, Sigma0_star,
                             mu_center, t, kpc_as)
    sini = max(np.sin(inc), 0.15)
    vlos = vrot_map * sini * np.cos(theta)
    vlos += rng.normal(0, 8.0, size=shape)
    vlos_ivar = np.full(shape, 1.0 / 8.0 ** 2)

    # velocity dispersion map
    sig_map = t["sigma0"] * np.exp(-R_arcsec / (2 * h_star_arcsec)) + 25.0
    sig_map += rng.normal(0, 5.0, size=shape)
    sig_ivar = np.full(shape, 1.0 / 5.0 ** 2)

    # ---- emission-line maps ---------------------------------------------
    nii_ha, oiii_hb = _line_ratios_from_oh(oh)
    ha_int = 100.0 * np.exp(-R_arcsec / h_star_arcsec)      # arbitrary flux units
    ebv = np.clip(0.35 * np.exp(-R_arcsec / (1.5 * h_star_arcsec)), 0, 0.6)
    from .metallicity import k_calzetti, _WAVE
    kk = {ln: k_calzetti(np.array([_WAVE[ln]]))[0] for ln in CANONICAL_LINES}

    hb_int = ha_int / 2.86
    intrinsic = {
        "Ha": ha_int,
        "Hb": hb_int,
        "NII6583": nii_ha * ha_int,
        "OIII5007": oiii_hb * hb_int,
        "OII3727": 1.2 * hb_int,
        "SII6717": 0.18 * ha_int,
        "SII6731": 0.13 * ha_int,
    }
    noise_abs = (ha_int.max() * 10 ** (-0.4 * kk["Ha"] * ebv.max())) / 60.0
    lines, ivars = {}, {}
    for ln, f_int in intrinsic.items():
        f_obs = f_int * 10 ** (-0.4 * kk[ln] * ebv)         # apply reddening
        f_obs = f_obs + rng.normal(0, noise_abs, size=shape)
        lines[ln] = f_obs
        ivars[ln] = np.full(shape, 1.0 / noise_abs ** 2)

    # ---- [alpha/Fe] ------------------------------------------------------
    afe_true = chemev.forward_alpha_fe(t["alpha_true"], t["tau_true"], cfg.yields)
    afe_obs = float(afe_true + rng.normal(0, 0.05))

    truth_out = dict(t)
    truth_out.update(alpha_true=t["alpha_true"], eta_true=t["eta_true"],
                     tau_true=t["tau_true"], y_O_true=y_O, y_eff_true=y_eff_true,
                     oh0_true=oh0_true, grad_true=grad_true,
                     mu_center=mu_center, g_dagger_true=t["g_dagger_true"],
                     v_flat_true=float(np.nanmedian(vrot_map[r_re > 1.3])),
                     alpha_fe_true=afe_true, inclination=inc)

    gm = GalaxyMaps(
        plateifu=plateifu, z=t["z"], Re_arcsec=Re, ba=t["ba"], pa=t["pa"],
        logmstar=t["logmstar"], shape=shape,
        lines=lines, lines_ivar=ivars, lines_mask={},
        stellar_vel=vlos, stellar_vel_ivar=vlos_ivar,
        stellar_sigma=sig_map, stellar_sigma_ivar=sig_ivar,
        sigma_star=sigma_star, sigma_gas=sigma_gas,
        alpha_fe=afe_obs, alpha_fe_err=0.05,
        x0=x0, y0=y0, spaxel_arcsec=cfg_spaxel(cfg),
        provenance="mock", truth=truth_out,
    )
    return gm


def _rar_rotation(r_kpc, R_arcsec, h_star_arcsec, Sigma0_star, mu_center,
                  t, kpc_as) -> np.ndarray:
    """Build V_rot per spaxel from baryons + the injected RAR scale g_dagger."""
    # Fine symmetric radial grid for enclosed-mass baryonic curve.
    rmax_kpc = float(np.nanmax(r_kpc)) * 1.05 + 1e-3
    rf_kpc = np.linspace(1e-3, rmax_kpc, 400)
    rf_arcsec = rf_kpc / kpc_as
    sig_star_f = Sigma0_star * np.exp(-rf_arcsec / h_star_arcsec)
    # gas fraction along fine grid (same law as map, in R/Re units):
    rf_re = rf_arcsec / (h_star_arcsec * 1.678)
    mu_f = np.clip(mu_center + t["mu_slope"] * rf_re, 0.03, 0.85)
    sig_gas_f = sig_star_f * mu_f / (1.0 - mu_f)
    sig_tot = sig_star_f + sig_gas_f                       # Msun/pc^2

    r_pc = rf_kpc * 1e3
    integrand = 2 * np.pi * sig_tot * r_pc                 # Msun/pc
    # length == len(rf_kpc); Menc[0]=0 at rf_kpc[0]. Do NOT slice.
    Menc = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1])
                                          * np.diff(r_pc))])
    r_m = rf_kpc * CONST.kpc_m
    g_bar = CONST.G * (Menc * CONST.Msun_kg) / r_m ** 2
    # McGaugh RAR forward
    x = np.sqrt(g_bar / t["g_dagger_true"])
    g_obs = g_bar / (1.0 - np.exp(-x))
    v_fine = np.sqrt(g_obs * r_m) / 1e3                    # km/s
    return np.interp(r_kpc, rf_kpc, v_fine)


def cfg_spaxel(cfg: Config) -> float:
    return CONST.spaxel_arcsec


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------
def make_mock_sample(n: int, cfg: Optional[Config] = None,
                     seed: int = 0) -> List[GalaxyMaps]:
    """Generate a synthetic MaNGA-like sample of n galaxies."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    gals = []
    for i in range(n):
        gs = int(rng.integers(0, 2 ** 31 - 1))
        gm = make_mock_galaxy(plateifu=f"MOCK-{i:05d}", cfg=cfg, seed=gs)
        gals.append(gm)
    return gals
