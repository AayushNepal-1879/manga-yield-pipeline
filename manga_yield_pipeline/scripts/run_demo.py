#!/usr/bin/env python3
"""
End-to-end demo on synthetic MaNGA-like data.

Generates a mock sample, runs the full pipeline (metallicity -> kinematics /
orbit-levels -> Bayesian SN/IMF-slope inference -> RAR -> population statistics),
writes CSV/JSON/figures, and prints a recovery + correlation summary.

Usage:
    python -m scripts.run_demo --n 60 --seed 0 --outdir results
    python scripts/run_demo.py --n 40 --calibration PP04_O3N2
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running both as module and as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mangayield.config import Config
from mangayield import pipeline


def main():
    ap = argparse.ArgumentParser(description="MaNGA yield -> SN mass distribution demo")
    ap.add_argument("--n", type=int, default=60, help="number of mock galaxies")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--calibration", default="PP04_O3N2",
                    help="PP04_O3N2|PP04_N2|M13_O3N2|M13_N2|D16")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-alpha-fe", action="store_true",
                    help="disable the [alpha/Fe] constraint (widen the degeneracy)")
    args = ap.parse_args()

    cfg = Config()
    cfg.outdir = args.outdir
    cfg.metallicity.calibration = args.calibration
    if args.no_alpha_fe:
        cfg.inference.use_alpha_fe = False

    print(f"Running mock pipeline: N={args.n}, calibration={args.calibration}, "
          f"alpha/Fe constraint={cfg.inference.use_alpha_fe}")
    out = pipeline.run_pipeline(cfg=cfg, source="mock", mock_n=args.n,
                                seed=args.seed, make_plots=not args.no_plots)

    print(f"\nProcessed {len(out.records)} galaxies -> {os.path.abspath(cfg.outdir)}")

    print("\n=== RECOVERY (inferred vs injected truth) ===")
    print(f"{'quantity':12s} {'N':>4s} {'bias':>10s} {'scatter':>10s} {'r(obs,true)':>12s}")
    for name, d in out.recovery.items():
        print(f"{name:12s} {d['n']:4d} {d['bias']:10.4f} {d['scatter']:10.4f} "
              f"{d['r']:12.3f}")

    print("\n=== POPULATION CORRELATIONS (partial, controlling for logMstar unless noted) ===")
    print(f"{'y vs x':40s} {'ctrl':9s} {'r':>7s} {'p':>10s} {'signif':>7s}")
    for c in out.correlations:
        tag = "  <--" if c.significant() else ""
        ctrl = c.control or "-"
        print(f"{(c.y+' vs '+c.x):40s} {ctrl:9s} {c.r:7.3f} {c.p:10.3g} "
              f"{str(c.significant()):>7s}{tag}")

    if out.figures:
        print("\nFigures written:")
        for f in out.figures:
            print("  ", f)


if __name__ == "__main__":
    main()
