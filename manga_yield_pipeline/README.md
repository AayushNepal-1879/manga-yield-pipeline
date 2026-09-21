# mangayield

**An automated SciPy/NumPy pipeline that ingests MaNGA spectroscopic maps, models
heavy-metal (oxygen) yields across galactic disks, infers the supernova mass distribution
(the high-mass IMF slope) from those yields, resolves each disk into angular-momentum
"orbit levels", computes the Radial Acceleration Relation (RAR), and hunts for
population-level correlations between the inferred supernova mass distribution, outer-disk
kinematics, and the RAR.**

The scientific question: *can you recover the supernova mass distribution from how much
heavy metal a galaxy has locked into gas at different orbit levels, and what does the RAR
add to that picture?* The pipeline answers it end-to-end — with an explicit, quantified
account of the one degeneracy (IMF slope vs. gas outflows) that makes the question hard.
Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) before trusting any number.

---

## Install

Dependency-light — the core stack is all you need:

```bash
pip install numpy scipy astropy matplotlib pyyaml
```

`sdss-marvin` is **optional** and only needed for live SDSS/MaNGA access. Raw-FITS and
synthetic ("mock") modes run on the core stack alone.

```bash
pip install sdss-marvin      # optional, for real SDSS data
```

---

## Quickstart — synthetic demo

The full pipeline runs end-to-end on self-consistent mock galaxies (no download required):

```bash
python scripts/run_demo.py --n 60 --seed 0 --outdir results
```

This generates 60 mock galaxies, runs metallicity → kinematics/orbit-levels → Bayesian
SN/IMF-slope inference → RAR → population statistics, writes CSV/JSON/figures to `results/`,
and prints a recovery table (inferred vs. injected truth) plus the population correlations.

To *see the α–η degeneracy widen*, drop the `[α/Fe]` constraint:

```bash
python scripts/run_demo.py --n 60 --no-alpha-fe
```

Switch the abundance calibration to bracket that systematic:

```bash
python scripts/run_demo.py --n 60 --calibration PP04_N2   # or M13_O3N2, D16, ...
```

### Run the tests

```bash
python tests/test_pipeline.py        # standalone runner, prints PASS/FAIL per test
# or, if you have pytest:
pytest tests/test_pipeline.py -v
```

The suite checks yield monotonicity, the effective-yield round-trip, `[α/Fe]` trends, mock
well-formedness, single-galaxy α recovery, and — the key scientific validation — population
recovery of `y_eff`, gradient, `g_dagger`, and α within documented tolerances.

> **Note on this build.** The demo and test suite were **not executed in the session that
> generated this package** — the sandbox VM's file mount was wedged, so no code could be run
> there. Every module was instead verified by static review and hand-tracing (including the
> array-shape fixes in `rar.py` / `mock.py` and the BPT-mask/O-H range consistency). Please
> run the two commands above locally to confirm on your machine; they need only the core
> stack and complete in well under a minute for `--n 60`.

---

## Using real MaNGA data

```python
from mangayield.config import load_config
from mangayield import pipeline

cfg = load_config("config/default.yaml")

# Marvin (needs sdss-marvin configured):
out = pipeline.run_pipeline(cfg=cfg, source="marvin",
                            plateifus=["8485-1901", "7443-12703"])

# Raw DAP MAPS FITS files you have on disk:
out = pipeline.run_pipeline(cfg=cfg, source="fits",
                            plateifus=["8485-1901"],
                            fits_paths={"8485-1901": "/path/to/manga-8485-1901-MAPS-*.fits.gz"},
                            metas={"8485-1901": {"z": 0.019, "Re_arcsec": 6.2,
                                                 "ba": 0.7, "pa": 42.0, "logmstar": 9.6}})

print(out.recovery)        # {} for real data (no ground truth)
for c in out.correlations:
    print(c.y, "vs", c.x, "  r=", round(c.r, 3), " p=", c.p)
```

Outputs land in `cfg.outdir` (`results/` by default).

---

## Package layout

```
manga_yield_pipeline/
├── mangayield/
│   ├── config.py        # dataclass config + YAML loader + physical constants
│   ├── ingest.py        # GalaxyMaps container; marvin / fits / mock loaders; sample cuts
│   ├── mock.py          # self-consistent synthetic MaNGA-like generator (validation truth)
│   ├── geometry.py      # inclination, deprojection, R/Re maps
│   ├── metallicity.py   # dust correction, BPT SF mask, 5 strong-line calibrations, gradients
│   ├── kinematics.py    # ring-by-ring rotation-curve fit; angular-momentum "orbit levels"
│   ├── chemev.py        # IMF broken power law; oxygen/iron yields; y_eff; [α/Fe] forward model
│   ├── inference.py     # Bayesian α (+η,τ) inference: priors, likelihood, MAP + NumPy MCMC
│   ├── rar.py           # baryonic g_bar (spherical/thin-disk), McGaugh RAR, g_dagger fit
│   ├── stats.py         # Pearson/Spearman/partial correlations, bootstrap CIs, recovery metrics
│   ├── pipeline.py      # per-galaxy orchestration + full-sample run + CSV/JSON writers
│   └── plotting.py      # diagnostic figures (recovery, α-vs-kinematics, RAR, degeneracy)
├── scripts/run_demo.py  # end-to-end synthetic demo (CLI)
├── tests/test_pipeline.py
├── config/default.yaml  # every tunable, documented
├── docs/METHODOLOGY.md  # models, assumptions, and the caveats that matter
└── requirements.txt
```

---

## Configuration

Everything has a sane default, so the pipeline runs with zero configuration. Any field in
[`config/default.yaml`](config/default.yaml) can be overridden — abundance calibration, BPT
mode, extinction law, ring count, orbit-level count, IMF break masses and slopes, SN mass
floor, priors on α/η/τ, MCMC length, RAR `g_bar` model, and more. Load with
`mangayield.load_config("config/default.yaml")`.

---

## Outputs

Written to `cfg.outdir` on every run:

- **`galaxy_records.csv`** — one row per galaxy: metallicity gradient, `V_flat`, orbit-level
  metal slope `oh_logj_slope`, inferred `alpha_med` (with 16th/84th percentiles), `eta_med`,
  `y_eff_obs`, `alpha_eta_corr` (per-galaxy degeneracy strength), `g_dagger`, and — for mock
  runs — the injected `*_true` columns alongside.
- **`correlations.csv`** — every tested pair with `r`, `p`, 68% bootstrap CI, and the control
  variable (partial correlations control for `log M★`).
- **`recovery.json`** — inferred-vs-injected bias/scatter/`r` (mock only).
- **`config_used.json`** — the exact configuration, for reproducibility.
- **`fig_*.png`** — recovery, α vs. outer-disk kinematics, the population RAR, the α–η
  posterior, and a correlation forest plot.

---

## The one caveat you must not skip

Inferring an IMF slope from abundances is **degenerate with gas outflows**: the effective
yield constrains only `y_O(α)/(1+η)`, so a top-heavy IMF with weak outflows mimics a normal
IMF with strong outflows. This pipeline treats that degeneracy openly — it reports the
posterior α–η correlation per galaxy, uses `[α/Fe]` (nearly outflow-independent) to help
break it, and lets you switch it off to watch the banana widen. See
[`docs/METHODOLOGY.md` §7](docs/METHODOLOGY.md) for this and the other limitations (yield-table
approximations, leaky-box idealisation, strong-line systematics, kinematic simplifications,
and why the RAR here is a consistency check rather than an independent dark-matter test).
