"""
Gas-phase (and stellar) metallicity from MaNGA emission-line maps.

Pipeline for each galaxy:
  1. Balmer-decrement dust correction (Calzetti or CCM89).
  2. BPT classification -> keep star-forming spaxels (strong-line calibrations
     are only valid for H II-region-like gas; AGN/LINER spaxels are rejected).
  3. Strong-line O/H via a selectable calibration (PP04, Marino13, Dopita16).
  4. Radial gradient fit of 12+log(O/H) vs R/Re, and binned radial profiles.

All calibrations return 12 + log(O/H); convert to oxygen mass fraction with
utils.oh_to_ZO for the yield analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .config import Config
from . import geometry
from .utils import weighted_linear_fit, oh_to_ZO

# Rest-frame vacuum-ish wavelengths (Angstrom) for the dust law.
_WAVE = {"OII3727": 3727.0, "Hb": 4861.0, "OIII5007": 5007.0,
         "Ha": 6563.0, "NII6583": 6583.0, "SII6717": 6717.0, "SII6731": 6731.0}


# --------------------------------------------------------------------------
# Dust / extinction
# --------------------------------------------------------------------------
def k_calzetti(wave_ang: np.ndarray, Rv: float = 4.05) -> np.ndarray:
    """Calzetti et al. (2000) starburst attenuation curve k(lambda)."""
    lam = np.asarray(wave_ang, float) * 1e-4          # micron
    k = np.empty_like(lam)
    hi = lam >= 0.63
    lo = ~hi
    k[hi] = 2.659 * (-1.857 + 1.040 / lam[hi]) + Rv
    k[lo] = 2.659 * (-2.156 + 1.509 / lam[lo]
                     - 0.198 / lam[lo] ** 2 + 0.011 / lam[lo] ** 3) + Rv
    return k


def k_ccm(wave_ang: np.ndarray, Rv: float = 3.1) -> np.ndarray:
    """Cardelli, Clayton & Mathis (1989) Galactic extinction k(lambda)=A(lam)/E(B-V)
    for the optical/NIR (1.1 <= x <= 3.3 um^-1)."""
    x = 1.0 / (np.asarray(wave_ang, float) * 1e-4)     # um^-1
    y = x - 1.82
    a = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3
         + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6
         + 0.32999 * y ** 7)
    b = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3
         - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6
         - 2.09002 * y ** 7)
    return (a + b / Rv) * Rv


def _k_law(law: str) -> callable:
    return k_ccm if law.lower().startswith("card") or law.lower() == "ccm" else k_calzetti


def ebv_from_balmer(ha: np.ndarray, hb: np.ndarray, law: str,
                    intrinsic: float = 2.86) -> np.ndarray:
    """E(B-V) from the Balmer decrement Ha/Hb. Negative values (noise) -> 0."""
    kfun = _k_law(law)
    kHa = kfun(np.array([_WAVE["Ha"]]))[0]
    kHb = kfun(np.array([_WAVE["Hb"]]))[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ha / hb
        ebv = (2.5 / (kHb - kHa)) * np.log10(ratio / intrinsic)
    ebv = np.where(np.isfinite(ebv), ebv, 0.0)
    return np.clip(ebv, 0.0, 2.0)


def deredden(gm, cfg: Config) -> Dict[str, np.ndarray]:
    """Return dust-corrected line maps (dict keyed by canonical line names)."""
    law = cfg.metallicity.extinction_law
    ha = gm.lines.get("Ha")
    hb = gm.lines.get("Hb")
    corrected = {}
    if ha is None or hb is None:
        # No Balmer pair -> return raw (ratios of nearby lines still ~ok).
        return {k: np.array(v, float) for k, v in gm.lines.items()}
    ebv = ebv_from_balmer(ha, hb, law)
    kfun = _k_law(law)
    for name, flux in gm.lines.items():
        k = kfun(np.array([_WAVE.get(name, 5500.0)]))[0]
        corrected[name] = np.asarray(flux, float) * 10.0 ** (0.4 * k * ebv)
    return corrected


# --------------------------------------------------------------------------
# BPT star-forming mask
# --------------------------------------------------------------------------
def bpt_sf_mask(lines: Dict[str, np.ndarray], mode: str = "kauffmann") -> np.ndarray:
    """Boolean mask of star-forming spaxels on the [NII] BPT diagram."""
    ha, hb = lines.get("Ha"), lines.get("Hb")
    nii, oiii = lines.get("NII6583"), lines.get("OIII5007")
    if any(v is None for v in (ha, hb, nii, oiii)):
        # cannot classify -> accept all finite spaxels
        base = next(iter(lines.values()))
        return np.isfinite(base)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(nii / ha)
        y = np.log10(oiii / hb)
    if mode == "none":
        return np.isfinite(x) & np.isfinite(y)
    if mode == "kewley":
        line = 0.61 / (x - 0.47) + 1.19
        sf = (y < line) | (x >= 0.47)
        sf = np.where(x >= 0.47, False, sf)
    else:  # kauffmann (default, purest SF)
        line = 0.61 / (x - 0.05) + 1.30
        sf = np.where(x >= 0.05, False, y < line)
    return sf & np.isfinite(x) & np.isfinite(y)


# --------------------------------------------------------------------------
# Strong-line O/H calibrations -> 12 + log(O/H)
# --------------------------------------------------------------------------
def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


def strong_line_oh(lines: Dict[str, np.ndarray], calibration: str
                   ) -> Tuple[np.ndarray, str]:
    """Return (12+log(O/H) map, ratio_used) for the requested calibration."""
    ha, hb = lines.get("Ha"), lines.get("Hb")
    nii = lines.get("NII6583")
    oiii = lines.get("OIII5007")
    sii = None
    if lines.get("SII6717") is not None and lines.get("SII6731") is not None:
        sii = lines["SII6717"] + lines["SII6731"]
    cal = calibration.upper()

    with np.errstate(divide="ignore", invalid="ignore"):
        if cal in ("PP04_N2", "M13_N2"):
            N2 = np.log10(_ratio(nii, ha))
            if cal == "PP04_N2":
                oh = 8.90 + 0.57 * N2
            else:
                oh = 8.743 + 0.462 * N2
            return oh, "N2"
        if cal in ("PP04_O3N2", "M13_O3N2"):
            O3N2 = np.log10(_ratio(_ratio(oiii, hb), _ratio(nii, ha)))
            if cal == "PP04_O3N2":
                oh = 8.73 - 0.32 * O3N2
            else:
                oh = 8.533 - 0.214 * O3N2
            return oh, "O3N2"
        if cal == "D16":
            if sii is None:
                raise ValueError("D16 calibration requires [SII] doublet")
            yv = np.log10(_ratio(nii, sii)) + 0.264 * np.log10(_ratio(nii, ha))
            oh = 8.77 + yv
            return oh, "D16"
    raise ValueError(f"unknown calibration '{calibration}'")


def _oh_sigma(gm, calibration: str, ratio_used: str, sn_min: float) -> np.ndarray:
    """Per-spaxel O/H uncertainty (dex) propagated from line S/N, with a 0.05
    dex systematic floor for calibration scatter."""
    slope = {"N2": 0.57, "O3N2": 0.32, "D16": 1.0}.get(ratio_used, 0.5)
    lines_in = {"N2": ["NII6583", "Ha"],
                "O3N2": ["OIII5007", "Hb", "NII6583", "Ha"],
                "D16": ["NII6583", "SII6717", "SII6731", "Ha"]}[ratio_used]
    var = np.zeros(gm.shape)
    for ln in lines_in:
        sn = gm.line_sn(ln)
        with np.errstate(divide="ignore", invalid="ignore"):
            var += (0.434 / np.clip(sn, 1e-3, None)) ** 2
    sig = slope * np.sqrt(var)
    return np.sqrt(sig ** 2 + 0.05 ** 2)


@dataclass
class MetalResult:
    oh: np.ndarray                 # 12+log(O/H) map (NaN where invalid)
    oh_err: np.ndarray             # per-spaxel dex uncertainty
    Z_O: np.ndarray                # oxygen mass fraction map
    radius_re: np.ndarray          # R/Re map
    sf_mask: np.ndarray            # star-forming spaxels used
    oh0: float                     # gradient-fit central O/H (R=0)
    grad: float                    # gradient dex / Re
    oh0_err: float
    grad_err: float
    scatter: float                 # residual scatter (dex)
    n_used: int
    calibration: str
    ratio_used: str


def measure_metallicity(gm, cfg: Config) -> MetalResult:
    """Full gas-phase metallicity measurement for one galaxy."""
    mcfg = cfg.metallicity
    lines_corr = deredden(gm, cfg)

    # Validity: S/N floor on every line feeding the chosen calibration.
    ratio_used = {"PP04_N2": "N2", "M13_N2": "N2", "PP04_O3N2": "O3N2",
                  "M13_O3N2": "O3N2", "D16": "D16"}[mcfg.calibration.upper()]
    needed = {"N2": ["NII6583", "Ha"],
              "O3N2": ["OIII5007", "Hb", "NII6583", "Ha"],
              "D16": ["NII6583", "SII6717", "SII6731", "Ha"]}[ratio_used]
    valid = np.ones(gm.shape, bool)
    for ln in needed:
        valid &= gm.valid_line(ln, mcfg.sn_min)
    # Balmer decrement sanity (reject unphysical Ha/Hb below Case B).
    if gm.lines.get("Ha") is not None and gm.lines.get("Hb") is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            dec = gm.lines["Ha"] / gm.lines["Hb"]
        valid &= np.isfinite(dec) & (dec >= mcfg.ha_hb_min * 0.9)

    sf = bpt_sf_mask(lines_corr, mcfg.bpt_class)
    use = valid & sf

    oh, ratio_used2 = strong_line_oh(lines_corr, mcfg.calibration)
    oh = np.where(use, oh, np.nan)
    oh_err = _oh_sigma(gm, mcfg.calibration, ratio_used, mcfg.sn_min)
    Z_O = np.where(np.isfinite(oh), oh_to_ZO(oh), np.nan)

    radius_re = geometry.radius_re_map(gm, cfg)

    # Gradient fit within the configured radial range.
    fit_ok = np.isfinite(oh) & (radius_re <= mcfg.fit_gradient_rmax_re) & (radius_re >= 0)
    a, b, sa, sb = weighted_linear_fit(radius_re[fit_ok], oh[fit_ok],
                                       1.0 / oh_err[fit_ok] ** 2)
    if np.isfinite(a):
        resid = oh[fit_ok] - (a + b * radius_re[fit_ok])
        scatter = float(np.nanstd(resid))
    else:
        scatter = np.nan

    return MetalResult(
        oh=oh, oh_err=oh_err, Z_O=Z_O, radius_re=radius_re, sf_mask=use,
        oh0=a, grad=b, oh0_err=sa, grad_err=sb, scatter=scatter,
        n_used=int(np.count_nonzero(fit_ok)),
        calibration=mcfg.calibration, ratio_used=ratio_used,
    )


def radial_profile(values: np.ndarray, radius_re: np.ndarray,
                   errors: Optional[np.ndarray] = None,
                   bins: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin `values` in R/Re. Returns (r_center, mean, err_on_mean, n_per_bin)."""
    if bins is None:
        bins = np.linspace(0, 2.5, 11)
    r_c = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(r_c.size, np.nan)
    err = np.full(r_c.size, np.nan)
    n = np.zeros(r_c.size, int)
    finite = np.isfinite(values) & np.isfinite(radius_re)
    for i in range(r_c.size):
        sel = finite & (radius_re >= bins[i]) & (radius_re < bins[i + 1])
        k = int(np.count_nonzero(sel))
        n[i] = k
        if k >= 3:
            v = values[sel]
            mean[i] = np.nanmean(v)
            err[i] = np.nanstd(v) / np.sqrt(k)
    return r_c, mean, err, n
