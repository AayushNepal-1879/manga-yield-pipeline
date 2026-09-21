"""
Ingestion layer.

Three back-ends, one container (`GalaxyMaps`):

  * "marvin"  -> live SDSS access via the official `sdss-marvin` package
  * "fits"    -> raw DAP MAPS FITS files you downloaded yourself (astropy)
  * "mock"    -> synthetic MaNGA-like maps (see mock.py) for testing/demo

Everything downstream consumes `GalaxyMaps`, so adding a new back-end just means
producing one of these objects. Emission-line fluxes follow the MaNGA DAP
convention (1e-17 erg/s/cm^2/spaxel) with matched inverse-variance maps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import warnings

import numpy as np

from .config import Config, CONST

# Internal canonical emission-line keys used everywhere downstream.
CANONICAL_LINES = ["OII3727", "Hb", "OIII5007", "Ha", "NII6583", "SII6717", "SII6731"]

# Aliases seen in DAP MAPS channel headers / Marvin property names.
_LINE_ALIASES: Dict[str, List[str]] = {
    "OII3727": ["oii-3727", "oii3727", "oiid-3728", "oii-3729", "oii"],
    "Hb": ["hb-4862", "hb-4861", "hbeta", "hb"],
    "OIII5007": ["oiii-5008", "oiii-5007", "oiii5007", "oiii"],
    "Ha": ["ha-6564", "ha-6563", "halpha", "ha"],
    "NII6583": ["nii-6585", "nii-6583", "nii6583", "nii"],
    "SII6717": ["sii-6718", "sii-6717", "sii6717"],
    "SII6731": ["sii-6732", "sii-6731", "sii6731"],
}


@dataclass
class GalaxyMaps:
    """Unified per-galaxy map container in the DAP convention."""
    plateifu: str
    z: float
    Re_arcsec: float
    ba: float                       # minor/major axis ratio b/a
    pa: float                       # position angle (deg, N->E)
    logmstar: float
    shape: tuple                    # (ny, nx)

    # Emission-line flux maps and inverse variances, keyed by CANONICAL_LINES.
    lines: Dict[str, np.ndarray] = field(default_factory=dict)
    lines_ivar: Dict[str, np.ndarray] = field(default_factory=dict)
    lines_mask: Dict[str, np.ndarray] = field(default_factory=dict)

    # Kinematics (km/s). *_corr sigmas are instrument-corrected.
    stellar_vel: Optional[np.ndarray] = None
    stellar_vel_ivar: Optional[np.ndarray] = None
    stellar_sigma: Optional[np.ndarray] = None
    stellar_sigma_ivar: Optional[np.ndarray] = None
    gas_vel: Optional[np.ndarray] = None
    gas_vel_ivar: Optional[np.ndarray] = None
    gas_sigma: Optional[np.ndarray] = None

    # Surface densities (Msun/pc^2) and stellar populations (optional).
    sigma_star: Optional[np.ndarray] = None
    sigma_gas: Optional[np.ndarray] = None
    stellar_ZH: Optional[np.ndarray] = None      # [Z/H] dex map
    alpha_fe: Optional[float] = None             # global [alpha/Fe] (dex)
    alpha_fe_err: Optional[float] = None

    # Geometry
    x0: Optional[float] = None                   # center col (pixel)
    y0: Optional[float] = None                   # center row (pixel)
    spaxel_arcsec: float = CONST.spaxel_arcsec
    radius_re: Optional[np.ndarray] = None       # precomputed R/Re, else derived

    provenance: str = "unknown"
    truth: Dict[str, Any] = field(default_factory=dict)  # injected params (mock only)

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = (self.shape[1] - 1) / 2.0
        if self.y0 is None:
            self.y0 = (self.shape[0] - 1) / 2.0

    def line_sn(self, name: str) -> np.ndarray:
        """Per-spaxel S/N for a line = flux * sqrt(ivar)."""
        f = self.lines.get(name)
        iv = self.lines_ivar.get(name)
        if f is None or iv is None:
            return np.zeros(self.shape)
        with np.errstate(invalid="ignore"):
            sn = f * np.sqrt(np.clip(iv, 0, None))
        sn[~np.isfinite(sn)] = 0.0
        return sn

    def valid_line(self, name: str, sn_min: float) -> np.ndarray:
        """Boolean map: line present, positive, S/N above floor, not masked."""
        ok = self.line_sn(name) >= sn_min
        f = self.lines.get(name)
        if f is not None:
            ok &= np.isfinite(f) & (f > 0)
        m = self.lines_mask.get(name)
        if m is not None:
            ok &= ~m.astype(bool)
        return ok


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------
def load_galaxy(plateifu: str,
                source: str = "mock",
                cfg: Optional[Config] = None,
                fits_path: Optional[str] = None,
                **kwargs) -> GalaxyMaps:
    """Load one galaxy's maps from the requested back-end."""
    cfg = cfg or Config()
    source = source.lower()
    if source == "mock":
        from .mock import make_mock_galaxy
        return make_mock_galaxy(plateifu=plateifu, cfg=cfg, **kwargs)
    if source == "fits":
        if fits_path is None:
            raise ValueError("source='fits' requires fits_path=<DAP MAPS file>")
        return _load_from_fits(plateifu, fits_path, cfg, **kwargs)
    if source == "marvin":
        return _load_from_marvin(plateifu, cfg, **kwargs)
    raise ValueError(f"unknown source '{source}' (use mock|fits|marvin)")


# --------------------------------------------------------------------------
# FITS back-end
# --------------------------------------------------------------------------
def _channel_index(header, aliases: List[str]) -> Optional[int]:
    """Find the 1-based channel whose 'C##' header value matches an alias."""
    for key in header:
        if not str(key).startswith("C"):
            continue
        rest = str(key)[1:]
        if not rest.isdigit():
            continue
        val = str(header[key]).strip().lower()
        for a in aliases:
            if a == val or a in val:
                return int(rest)
    return None


def _load_from_fits(plateifu: str, fits_path: str, cfg: Config,
                    meta: Optional[dict] = None) -> GalaxyMaps:
    from astropy.io import fits

    meta = meta or {}
    with fits.open(fits_path) as hdul:
        ext = {h.name.upper(): i for i, h in enumerate(hdul)}

        def get_cube(name):
            return hdul[ext[name]].data if name in ext else None

        egflux = get_cube("EMLINE_GFLUX")
        egivar = get_cube("EMLINE_GFLUX_IVAR")
        egmask = get_cube("EMLINE_GFLUX_MASK")
        eg_hdr = hdul[ext["EMLINE_GFLUX"]].header if "EMLINE_GFLUX" in ext else None

        lines, ivars, masks = {}, {}, {}
        if egflux is not None and eg_hdr is not None:
            for key, aliases in _LINE_ALIASES.items():
                ci = _channel_index(eg_hdr, aliases)
                if ci is None:
                    continue
                lines[key] = np.asarray(egflux[ci - 1], float)
                if egivar is not None:
                    ivars[key] = np.asarray(egivar[ci - 1], float)
                if egmask is not None:
                    masks[key] = np.asarray(egmask[ci - 1]) != 0

        shape = None
        if lines:
            shape = next(iter(lines.values())).shape

        # Stellar kinematics
        stellar_vel = get_cube("STELLAR_VEL")
        stellar_vel_ivar = get_cube("STELLAR_VEL_IVAR")
        stellar_sigma = get_cube("STELLAR_SIGMA")
        stellar_sigma_ivar = get_cube("STELLAR_SIGMA_IVAR")
        stellar_sigma_corr = get_cube("STELLAR_SIGMA_CORR")
        if (stellar_sigma is not None and stellar_sigma_corr is not None):
            # DAP: subtract instrumental dispersion in quadrature.
            with np.errstate(invalid="ignore"):
                stellar_sigma = np.sqrt(
                    np.clip(stellar_sigma ** 2 - stellar_sigma_corr ** 2, 0, None))

        # Gas (Halpha) velocity from EMLINE_GVEL cube channel = Ha
        gvel = get_cube("EMLINE_GVEL")
        gvel_hdr = hdul[ext["EMLINE_GVEL"]].header if "EMLINE_GVEL" in ext else None
        gas_vel = None
        if gvel is not None and gvel_hdr is not None:
            ci = _channel_index(gvel_hdr, _LINE_ALIASES["Ha"])
            if ci is not None:
                gas_vel = np.asarray(gvel[ci - 1], float)

        # Elliptical radius R/Re
        radius_re = None
        ell = get_cube("SPX_ELLCOO")
        ell_hdr = hdul[ext["SPX_ELLCOO"]].header if "SPX_ELLCOO" in ext else None
        if ell is not None and ell_hdr is not None:
            ci = _channel_index(ell_hdr, ["r/re", "elliptical radius (r_e)",
                                          "radius normalized", "r_re"])
            if ci is not None:
                radius_re = np.asarray(ell[ci - 1], float)

        if shape is None:
            for cand in (stellar_vel, gas_vel, radius_re):
                if cand is not None:
                    shape = cand.shape
                    break
        if shape is None:
            raise RuntimeError(f"could not determine map shape from {fits_path}")

    gm = GalaxyMaps(
        plateifu=plateifu,
        z=float(meta.get("z", 0.03)),
        Re_arcsec=float(meta.get("Re_arcsec", 5.0)),
        ba=float(meta.get("ba", 0.6)),
        pa=float(meta.get("pa", 0.0)),
        logmstar=float(meta.get("logmstar", 10.0)),
        shape=tuple(shape),
        lines=lines, lines_ivar=ivars, lines_mask=masks,
        stellar_vel=stellar_vel, stellar_vel_ivar=stellar_vel_ivar,
        stellar_sigma=stellar_sigma, stellar_sigma_ivar=stellar_sigma_ivar,
        gas_vel=gas_vel,
        sigma_star=meta.get("sigma_star"),
        stellar_ZH=meta.get("stellar_ZH"),
        alpha_fe=meta.get("alpha_fe"),
        alpha_fe_err=meta.get("alpha_fe_err"),
        radius_re=radius_re,
        provenance="fits",
    )
    return gm


# --------------------------------------------------------------------------
# Marvin back-end (optional dependency)
# --------------------------------------------------------------------------
def _load_from_marvin(plateifu: str, cfg: Config, **kwargs) -> GalaxyMaps:
    try:
        from marvin.tools.maps import Maps
    except Exception as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "sdss-marvin is not installed/configured. Install `sdss-marvin` and set "
            "SAS_BASE_DIR / MANGA_SPECTRO_ANALYSIS, or use source='fits'/'mock'."
        ) from exc

    maps = Maps(plateifu, bintype=kwargs.get("bintype", "HYB10"))
    hdr = maps.header

    def marv(prop, channel=None):
        try:
            m = maps[prop] if channel is None else maps[f"{prop}_{channel}"]
            return np.asarray(m.value, float), np.asarray(getattr(m, "ivar", None), float) \
                if getattr(m, "ivar", None) is not None else None, \
                np.asarray(getattr(m, "mask", None), float) != 0 \
                if getattr(m, "mask", None) is not None else None
        except Exception:
            return None, None, None

    lines, ivars, masks = {}, {}, {}
    marvin_line_prop = {
        "OII3727": "emline_gflux_oiid_3728",
        "Hb": "emline_gflux_hb_4862",
        "OIII5007": "emline_gflux_oiii_5008",
        "Ha": "emline_gflux_ha_6564",
        "NII6583": "emline_gflux_nii_6585",
        "SII6717": "emline_gflux_sii_6718",
        "SII6731": "emline_gflux_sii_6732",
    }
    for key, prop in marvin_line_prop.items():
        f, iv, mk = marv(prop)
        if f is not None:
            lines[key] = f
            if iv is not None:
                ivars[key] = iv
            if mk is not None:
                masks[key] = mk

    sv, sv_iv, _ = marv("stellar_vel")
    ss, ss_iv, _ = marv("stellar_sigma")
    shape = None
    for cand in list(lines.values()) + [sv]:
        if cand is not None:
            shape = cand.shape
            break

    # NSA metadata for geometry (fall back to kwargs).
    def nsa(k, default):
        try:
            return float(maps.nsa[k])
        except Exception:
            return float(kwargs.get(k, default))

    gm = GalaxyMaps(
        plateifu=plateifu,
        z=nsa("z", kwargs.get("z", 0.03)),
        Re_arcsec=nsa("elpetro_th50_r", kwargs.get("Re_arcsec", 5.0)),
        ba=nsa("elpetro_ba", kwargs.get("ba", 0.6)),
        pa=nsa("elpetro_phi", kwargs.get("pa", 0.0)),
        logmstar=np.log10(nsa("elpetro_mass", 1e10)),
        shape=tuple(shape) if shape else (0, 0),
        lines=lines, lines_ivar=ivars, lines_mask=masks,
        stellar_vel=sv, stellar_vel_ivar=sv_iv,
        stellar_sigma=ss, stellar_sigma_ivar=ss_iv,
        provenance="marvin",
    )
    return gm


# --------------------------------------------------------------------------
# Sample selection
# --------------------------------------------------------------------------
def select_sample(cfg: Config,
                  drpall_path: Optional[str] = None,
                  dapall_path: Optional[str] = None) -> List[dict]:
    """Return a list of galaxy metadata dicts passing the selection cuts.

    With no catalogue paths, returns an empty list (use mock sampling instead).
    """
    if drpall_path is None:
        warnings.warn("select_sample: no drpall_path given; returning empty list. "
                      "Use mock.make_mock_sample() for a synthetic catalogue.")
        return []

    from astropy.io import fits
    with fits.open(drpall_path) as hdul:
        data = hdul[1].data
    s = cfg.sample

    def col(name, alt=None):
        for n in ([name] + ([alt] if alt else [])):
            if n in data.names:
                return data[n]
        return None

    plateifu = col("plateifu")
    z = col("nsa_z", "z")
    ba = col("nsa_elpetro_ba")
    phi = col("nsa_elpetro_phi")
    th50 = col("nsa_elpetro_th50_r")
    mass = col("nsa_elpetro_mass")

    out = []
    n = len(plateifu)
    for i in range(n):
        try:
            logm = np.log10(mass[i]) if mass[i] > 0 else np.nan
        except Exception:
            logm = np.nan
        if not (s.z_min <= z[i] <= s.z_max):
            continue
        if not (s.ba_min <= ba[i] <= s.ba_max):
            continue
        if np.isfinite(logm) and not (s.logmstar_min <= logm <= s.logmstar_max):
            continue
        out.append(dict(plateifu=str(plateifu[i]).strip(), z=float(z[i]),
                        ba=float(ba[i]), pa=float(phi[i]),
                        Re_arcsec=float(th50[i]), logmstar=float(logm)))
        if s.max_galaxies and len(out) >= s.max_galaxies:
            break
    return out
