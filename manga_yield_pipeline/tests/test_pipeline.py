"""
Sanity + validation tests. Runs under pytest, or standalone:  python tests/test_pipeline.py

The scientific validation test generates a mock sample and checks that the
pipeline recovers the injected IMF slope, effective yield, gradient and RAR
acceleration scale within tolerance.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangayield.config import Config
from mangayield import chemev, mock, pipeline, metallicity, kinematics, stats


def test_oxygen_yield_monotonic():
    cfg = Config().yields
    alphas = np.linspace(1.2, 3.4, 12)
    y = np.array([chemev.oxygen_yield(a, cfg) for a in alphas])
    # top-heavier IMF (smaller alpha) => larger oxygen yield
    assert np.all(np.diff(y) < 0), "oxygen yield should decrease with alpha"


def test_effective_yield_roundtrip():
    cfg = Config().yields
    for alpha, eta in [(2.35, 0.3), (1.8, 1.0), (2.9, 0.1)]:
        y_eff = chemev.y_eff_from_alpha_eta(alpha, eta, cfg)
        a_rec = chemev.alpha_from_y_eff(y_eff, eta, cfg)
        assert abs(a_rec - alpha) < 0.05, (alpha, eta, a_rec)


def test_fast_matches_slow():
    cfg = Config().yields
    for a in (1.5, 2.0, 2.35, 3.0):
        assert abs(chemev.oxygen_yield(a, cfg)
                   - float(chemev.oxygen_yield_fast(a, cfg))) < 1e-4


def test_alpha_fe_trends():
    cfg = Config().yields
    # top-heavier IMF -> higher [alpha/Fe]
    assert chemev.forward_alpha_fe(1.8, 3.0, cfg) > chemev.forward_alpha_fe(3.0, 3.0, cfg)
    # longer SF timescale -> lower [alpha/Fe]
    assert chemev.forward_alpha_fe(2.35, 1.0, cfg) > chemev.forward_alpha_fe(2.35, 8.0, cfg)


def test_mock_galaxy_wellformed():
    cfg = Config()
    gm = mock.make_mock_galaxy(seed=1, cfg=cfg)
    assert gm.stellar_vel is not None and gm.stellar_vel.shape == gm.shape
    met = metallicity.measure_metallicity(gm, cfg)
    oh = met.oh[np.isfinite(met.oh)]
    assert oh.size > 50
    assert 7.8 < np.nanmedian(oh) < 9.2, np.nanmedian(oh)


def test_single_galaxy_recovers_alpha():
    cfg = Config()
    gm = mock.make_mock_galaxy(seed=3, cfg=cfg)
    rec = pipeline.process_galaxy(gm, cfg)
    assert np.isfinite(rec["alpha_med"])
    # within ~0.5 of truth for a single galaxy (loose; degeneracy-limited)
    assert abs(rec["alpha_med"] - rec["alpha_true"]) < 0.6, rec


def test_population_recovery():
    cfg = Config()
    out = pipeline.run_pipeline(cfg=cfg, source="mock", mock_n=40, seed=7,
                                make_plots=False)
    recov = out.recovery
    # effective yield and g_dagger should be tightly recovered
    assert recov["y_eff"]["r"] > 0.8, recov["y_eff"]
    assert recov["g_dagger"]["r"] > 0.7, recov["g_dagger"]
    assert recov["gradient"]["r"] > 0.6, recov["gradient"]
    # IMF slope recovered with positive correlation to truth
    assert recov["alpha"]["r"] > 0.4, recov["alpha"]
    assert abs(recov["alpha"]["bias"]) < 0.4, recov["alpha"]


def test_injected_correlation_detected():
    cfg = Config()
    out = pipeline.run_pipeline(cfg=cfg, source="mock", mock_n=50, seed=11,
                                make_plots=False)
    cmap = {(c.x, c.y): c for c in out.correlations}
    # injected alpha-mass trend should show up as alpha_med vs logMstar
    c = cmap.get(("alpha_med", "logmstar"))
    assert c is not None and np.isfinite(c.r)
    assert abs(c.r) > 0.3, c


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            npass += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{npass}/{len(fns)} tests passed")
    sys.exit(0 if npass == len(fns) else 1)
