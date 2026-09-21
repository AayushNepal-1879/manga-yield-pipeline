"""
Disk geometry: inclination from axis ratio, and deprojection of sky
coordinates into the galaxy plane (in-plane radius and azimuth).

Shared by the metallicity and kinematics modules so both use identical radii.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def inclination_from_ba(ba: float, q0: float = 0.13) -> float:
    """Inclination (radians) from the observed axis ratio b/a.

    cos^2 i = ((b/a)^2 - q0^2) / (1 - q0^2), with q0 the intrinsic edge-on
    thickness. Clamped to [0, 90] deg equivalents.
    """
    ba = float(np.clip(ba, q0 + 1e-3, 1.0))
    cos2 = (ba ** 2 - q0 ** 2) / (1.0 - q0 ** 2)
    cos2 = float(np.clip(cos2, 0.0, 1.0))
    return float(np.arccos(np.sqrt(cos2)))


def sky_grid(shape: Tuple[int, int], x0: float, y0: float,
             spaxel_arcsec: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dx, dy) offsets from center in arcsec (x increases with column)."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = (xx - x0) * spaxel_arcsec
    dy = (yy - y0) * spaxel_arcsec
    return dx, dy


def deproject(shape: Tuple[int, int], x0: float, y0: float,
              pa_deg: float, inclination: float,
              spaxel_arcsec: float) -> Tuple[np.ndarray, np.ndarray]:
    """Deproject sky coordinates into the disk plane.

    Returns (R_arcsec, theta) where R_arcsec is the in-plane galactocentric
    radius (arcsec) and theta is the in-plane azimuthal angle measured from the
    major axis (radians). Standard tilted-disk transform.
    """
    dx, dy = sky_grid(shape, x0, y0, spaxel_arcsec)
    pa = np.deg2rad(pa_deg)
    # Rotate so the major axis lies along x'. PA measured N(up)->E; with x to the
    # right, y up, the rotation below aligns the kinematic major axis to x'.
    x_maj = dx * np.sin(pa) + dy * np.cos(pa)
    x_min = dx * np.cos(pa) - dy * np.sin(pa)
    cosi = np.cos(inclination)
    cosi = cosi if abs(cosi) > 1e-3 else 1e-3
    x_min_dep = x_min / cosi                      # stretch minor axis
    R = np.sqrt(x_maj ** 2 + x_min_dep ** 2)
    theta = np.arctan2(x_min_dep, x_maj)          # 0 along major axis
    return R, theta


def radius_re_map(gm, cfg) -> np.ndarray:
    """Return an R/Re map, using the precomputed one if present else geometry."""
    if getattr(gm, "radius_re", None) is not None:
        return np.asarray(gm.radius_re, float)
    inc = inclination_from_ba(gm.ba, cfg.kinematics.q0)
    R_arcsec, _ = deproject(gm.shape, gm.x0, gm.y0, gm.pa, inc, gm.spaxel_arcsec)
    return R_arcsec / max(gm.Re_arcsec, 1e-3)
