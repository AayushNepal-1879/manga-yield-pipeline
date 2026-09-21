"""
Population-level statistics.

Links the per-galaxy inference (supernova/IMF slope alpha, effective yield,
metallicity gradient) to outer-disk kinematics and RAR metrics across the
sample, with three safeguards against fooling ourselves:

  * Spearman rank correlation (robust to non-linearity / outliers).
  * Partial correlation controlling for stellar mass -- the dominant confounder;
    almost everything correlates with M_star, so a raw correlation between alpha
    and kinematics can be entirely mass-driven.
  * Bootstrap confidence intervals on every coefficient.

For mock runs it also reports recovery metrics (inferred vs injected truth),
which is how we validate the whole chain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np


def _get(records: List[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in records], float)


def _clean(*arrays) -> Tuple[np.ndarray, ...]:
    mask = np.ones(len(arrays[0]), bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in arrays)


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(a)


def pearson(x, y) -> Tuple[float, float]:
    from scipy.stats import pearsonr
    x, y = _clean(np.asarray(x, float), np.asarray(y, float))
    if x.size < 4:
        return (np.nan, np.nan)
    r, p = pearsonr(x, y)
    return (float(r), float(p))


def spearman(x, y) -> Tuple[float, float]:
    from scipy.stats import spearmanr
    x, y = _clean(np.asarray(x, float), np.asarray(y, float))
    if x.size < 4:
        return (np.nan, np.nan)
    r, p = spearmanr(x, y)
    return (float(r), float(p))


def partial_correlation(x, y, z, method: str = "spearman") -> Tuple[float, float]:
    """Correlation of x and y controlling for z.

    For Spearman we rank-transform first, then compute the partial Pearson
    correlation of the ranks (a standard partial-Spearman estimator).
    """
    from scipy.stats import pearsonr
    x, y, z = _clean(np.asarray(x, float), np.asarray(y, float),
                     np.asarray(z, float))
    if x.size < 5:
        return (np.nan, np.nan)
    if method == "spearman":
        x, y, z = _rank(x), _rank(y), _rank(z)

    def resid(a, b):
        A = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A.dot(coef)

    rx = resid(x, z)
    ry = resid(y, z)
    r, _ = pearsonr(rx, ry)
    # p-value via t-distribution with n-3 dof
    from scipy.stats import t as tdist
    n = x.size
    dof = n - 3
    if dof <= 0 or abs(r) >= 1:
        return (float(r), np.nan)
    tstat = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * tdist.sf(abs(tstat), dof)
    return (float(r), float(p))


def bootstrap_ci(x, y, stat: Callable, z: Optional[np.ndarray] = None,
                 n_boot: int = 2000, seed: int = 0
                 ) -> Tuple[float, float, float]:
    """Bootstrap (point, lo, hi) 68% CI for a correlation statistic."""
    rng = np.random.default_rng(seed)
    if z is None:
        x, y = _clean(np.asarray(x, float), np.asarray(y, float))
        data = (x, y)
    else:
        x, y, z = _clean(np.asarray(x, float), np.asarray(y, float),
                         np.asarray(z, float))
        data = (x, y, z)
    n = data[0].size
    if n < 5:
        return (np.nan, np.nan, np.nan)
    point = stat(*data)[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals[b] = stat(*[d[idx] for d in data])[0]
        except Exception:
            vals[b] = np.nan
    lo, hi = np.nanpercentile(vals, [16, 84])
    return (float(point), float(lo), float(hi))


@dataclass
class CorrResult:
    x: str
    y: str
    control: Optional[str]
    method: str
    r: float
    p: float
    ci_lo: float
    ci_hi: float
    n: int

    def significant(self, alpha: float = 0.05) -> bool:
        return np.isfinite(self.p) and self.p < alpha


def correlate(records: List[dict], xkey: str, ykey: str,
              control: Optional[str] = "logmstar", method: str = "spearman",
              n_boot: int = 1500) -> CorrResult:
    x, y = _get(records, xkey), _get(records, ykey)
    if control is None:
        stat = spearman if method == "spearman" else pearson
        r, p = stat(x, y)
        pt, lo, hi = bootstrap_ci(x, y, stat, n_boot=n_boot)
        xc, yc = _clean(x, y)
        n = xc.size
    else:
        z = _get(records, control)
        r, p = partial_correlation(x, y, z, method=method)
        stat = lambda a, b, c: partial_correlation(a, b, c, method=method)
        pt, lo, hi = bootstrap_ci(x, y, stat, z=z, n_boot=n_boot)
        xc, yc, zc = _clean(x, y, z)
        n = xc.size
    return CorrResult(x=xkey, y=ykey, control=control, method=method,
                      r=r, p=p, ci_lo=lo, ci_hi=hi, n=n)


# Default set of scientifically-motivated pairs to test.
DEFAULT_PAIRS = [
    # (x, y, control) : supernova mass distribution vs outer-disk kinematics
    ("alpha_med", "v_flat", "logmstar"),
    ("alpha_med", "sigma_outer", "logmstar"),
    ("alpha_med", "j_outer", "logmstar"),
    ("alpha_med", "v_over_sigma_outer", "logmstar"),
    # yields / gradients vs kinematics
    ("y_eff_obs", "v_flat", "logmstar"),
    ("grad", "v_flat", "logmstar"),
    ("grad", "j_outer", "logmstar"),
    # RAR vs metals
    ("md_outer", "alpha_med", "logmstar"),
    ("md_outer", "grad", "logmstar"),
    ("g_dagger", "oh0", None),
    ("rar_resid_outer", "y_eff_obs", "logmstar"),
    # sanity checks (should be strong)
    ("y_eff_obs", "logmstar", None),
    ("alpha_med", "logmstar", None),
]


def population_correlations(records: List[dict],
                            pairs: Optional[List[tuple]] = None,
                            method: str = "spearman") -> List[CorrResult]:
    pairs = pairs or DEFAULT_PAIRS
    out = []
    for x, y, ctrl in pairs:
        try:
            out.append(correlate(records, x, y, control=ctrl, method=method))
        except Exception:
            out.append(CorrResult(x, y, ctrl, method, np.nan, np.nan,
                                  np.nan, np.nan, 0))
    return out


def recovery_metrics(records: List[dict]) -> Dict[str, dict]:
    """Compare inferred quantities to injected truth (mock only)."""
    pairs = {
        "alpha": ("alpha_med", "alpha_true"),
        "y_eff": ("y_eff_obs", "y_eff_true"),
        "gradient": ("grad", "grad_true"),
        "oh0": ("oh0", "oh0_true"),
        "g_dagger": ("g_dagger", "g_dagger_true"),
        "v_flat": ("v_flat", "v_flat_true"),
    }
    out = {}
    for name, (obs_k, true_k) in pairs.items():
        obs, tru = _get(records, obs_k), _get(records, true_k)
        obs, tru = _clean(obs, tru)
        if obs.size < 3:
            out[name] = dict(n=int(obs.size), bias=np.nan, scatter=np.nan,
                             r=np.nan)
            continue
        diff = obs - tru
        r, _ = pearson(obs, tru)
        out[name] = dict(n=int(obs.size), bias=float(np.mean(diff)),
                         scatter=float(np.std(diff)), r=float(r),
                         frac_bias=float(np.mean(diff) / (np.mean(np.abs(tru)) + 1e-9)))
    return out
