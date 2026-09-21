"""
Pipeline orchestration.

process_galaxy: metallicity -> kinematics/orbit-levels -> Bayesian IMF-slope
inference -> RAR, assembled into one flat record.

run_pipeline: ingest a sample (mock | fits | marvin), process every galaxy,
compute population correlations + (for mock) recovery metrics, and write CSV/JSON
outputs and optional figures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os
import json
import warnings
import numpy as np

from .config import Config
from . import ingest, metallicity, kinematics, inference, rar, stats, mock


TRUTH_KEYS = ["alpha_true", "eta_true", "tau_true", "y_O_true", "y_eff_true",
              "oh0_true", "grad_true", "g_dagger_true", "v_flat_true",
              "alpha_fe_true"]


def _orbit_metal_slope(metal, kin) -> float:
    """d(12+log O/H)/d(log10 j) across angular-momentum ("orbit") levels."""
    lev, mean_j, mean_oh, oh_err, n = kinematics.metallicity_by_orbit_level(metal, kin)
    good = np.isfinite(mean_j) & np.isfinite(mean_oh) & (mean_j > 0) & (n >= 3)
    if np.count_nonzero(good) < 3:
        return np.nan
    x = np.log10(mean_j[good])
    y = mean_oh[good]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1])


def process_galaxy(gm, cfg: Config, keep_samples: bool = False) -> Dict[str, Any]:
    """Run the full per-galaxy analysis; return a flat record of scalars."""
    rec: Dict[str, Any] = dict(plateifu=gm.plateifu, z=gm.z, ba=gm.ba,
                               logmstar=gm.logmstar, provenance=gm.provenance)
    # 1) Metallicity
    met = metallicity.measure_metallicity(gm, cfg)
    rec.update(calibration=met.calibration, oh0=met.oh0, grad=met.grad,
               grad_err=met.grad_err, oh_scatter=met.scatter, n_metal=met.n_used)
    # 2) Kinematics + orbit levels
    kin = kinematics.fit_rotation_curve(gm, cfg)
    rec.update(v_flat=kin.v_flat, v_flat_err=kin.v_flat_err,
               sigma_outer=kin.sigma_outer, j_outer=kin.j_outer,
               v_over_sigma_outer=kin.v_over_sigma_outer,
               kpc_per_re=kin.kpc_per_re)
    rec["oh_logj_slope"] = _orbit_metal_slope(met, kin)
    # 3) Bayesian supernova/IMF-slope inference
    inf = inference.infer_alpha(met, gm, cfg, keep_samples=keep_samples)
    rec.update(alpha_map=inf.alpha_map, alpha_med=inf.alpha_med,
               alpha_lo=inf.alpha_lo, alpha_hi=inf.alpha_hi, eta_med=inf.eta_med,
               tau_med=inf.tau_med, alpha_eta_corr=inf.alpha_eta_corr,
               y_eff_obs=inf.y_eff_obs, y_eff_err=inf.y_eff_err,
               alpha_fe_used=inf.alpha_fe_used, n_yield_spx=inf.n_spaxels,
               mcmc_accept=inf.accept_frac)
    # 4) RAR
    rr = rar.compute_rar(gm, kin, cfg)
    rec.update(g_dagger=rr.g_dagger, g_dagger_err=rr.g_dagger_err,
               md_outer=rr.md_outer, rar_resid_outer=rr.rar_resid_outer)
    # 5) Truth (mock)
    for k in TRUTH_KEYS:
        if k in gm.truth:
            rec[k] = gm.truth[k]
    rec["_metal"] = met
    rec["_kin"] = kin
    rec["_inf"] = inf
    rec["_rar"] = rr
    return rec


@dataclass
class PipelineOutput:
    records: List[dict]
    correlations: List[stats.CorrResult]
    recovery: Dict[str, dict] = field(default_factory=dict)
    outdir: str = "results"
    figures: List[str] = field(default_factory=list)


def _strip_private(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def run_pipeline(cfg: Optional[Config] = None,
                 source: str = "mock",
                 plateifus: Optional[List[str]] = None,
                 fits_paths: Optional[Dict[str, str]] = None,
                 metas: Optional[Dict[str, dict]] = None,
                 mock_n: int = 60,
                 seed: int = 0,
                 make_plots: bool = True,
                 keep_samples_first: int = 6) -> PipelineOutput:
    """End-to-end run. For source='mock' a synthetic sample is generated."""
    cfg = cfg or Config()
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)

    # ---- ingest ----------------------------------------------------------
    galaxies = []
    if source == "mock":
        galaxies = mock.make_mock_sample(mock_n, cfg=cfg, seed=seed)
    else:
        if not plateifus:
            raise ValueError("provide plateifus for source='fits'|'marvin'")
        for pid in plateifus:
            try:
                fp = (fits_paths or {}).get(pid)
                meta = (metas or {}).get(pid, {})
                gm = ingest.load_galaxy(pid, source=source, cfg=cfg,
                                        fits_path=fp, meta=meta)
                galaxies.append(gm)
            except Exception as exc:  # keep going on per-galaxy failures
                warnings.warn(f"load failed for {pid}: {exc}")

    # ---- process ---------------------------------------------------------
    records = []
    for i, gm in enumerate(galaxies):
        try:
            rec = process_galaxy(gm, cfg, keep_samples=(i < keep_samples_first))
            records.append(rec)
        except Exception as exc:
            warnings.warn(f"processing failed for {gm.plateifu}: {exc}")

    # ---- population statistics ------------------------------------------
    clean_records = [_strip_private(r) for r in records]
    corrs = stats.population_correlations(clean_records)
    recov = stats.recovery_metrics(clean_records) if source == "mock" else {}

    # ---- write outputs ---------------------------------------------------
    _write_csv(os.path.join(outdir, "galaxy_records.csv"), clean_records)
    _write_corr_csv(os.path.join(outdir, "correlations.csv"), corrs)
    with open(os.path.join(outdir, "recovery.json"), "w") as fh:
        json.dump(recov, fh, indent=2)
    with open(os.path.join(outdir, "config_used.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)

    figures = []
    if make_plots:
        try:
            from . import plotting
            figures = plotting.make_all(records, corrs, recov, cfg, outdir,
                                        is_mock=(source == "mock"))
        except Exception as exc:
            warnings.warn(f"plotting failed: {exc}")

    return PipelineOutput(records=records, correlations=corrs, recovery=recov,
                          outdir=outdir, figures=figures)


# --------------------------------------------------------------------------
# Lightweight writers (no pandas dependency).
# --------------------------------------------------------------------------
def _write_csv(path: str, records: List[dict]) -> None:
    if not records:
        open(path, "w").close()
        return
    keys = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def _write_corr_csv(path: str, corrs: List[stats.CorrResult]) -> None:
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "control", "method", "r", "p_value",
                    "ci16", "ci84", "n", "significant_0.05"])
        for c in corrs:
            w.writerow([c.x, c.y, c.control, c.method, f"{c.r:.4f}",
                        f"{c.p:.4g}", f"{c.ci_lo:.4f}", f"{c.ci_hi:.4f}",
                        c.n, c.significant()])
