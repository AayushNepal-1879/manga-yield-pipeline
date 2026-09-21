# Methodology and Scientific Caveats

**Package:** `mangayield` — a MaNGA heavy-metal-yield → supernova-mass-distribution pipeline
**Version:** 0.1.0

This document describes what the pipeline actually computes, the assumptions behind
each step, and — most importantly — the limits of what the results can support. The
headline science question is deliberately hard: *can the supernova mass distribution
(the high-mass stellar initial mass function, or IMF) be inferred from how much heavy
metal a galaxy has locked into gas at different orbital radii, and how does that picture
compare with what the Radial Acceleration Relation says about the same disks?* The honest
short answer is **partially, and only under stated assumptions** — the reasons are the
subject of the [Scientific limitations](#7-scientific-limitations-read-this-before-trusting-a-number)
section, which you should read before quoting any number this code produces.

---

## 1. Scientific goal and logical chain

The pipeline is built around a single inferential chain, executed per galaxy and then
aggregated across the population:

```
emission-line maps
   → dust correction → BPT star-forming selection → strong-line 12+log(O/H)
   → oxygen mass fraction Z_O(spaxel)   and   gas fraction μ(spaxel)
   → effective yield  y_eff = Z_O / ln(1/μ)          (leaky-box inversion)
   → Bayesian inference of α (IMF high-mass slope) and η (outflow loading)
      using  y_eff = y_O(α)/(1+η)  and, optionally, [α/Fe]
   ⟂ independently: rotation curve → orbit ("angular-momentum") levels
   ⟂ independently: rotation curve + Σ maps → Radial Acceleration Relation (g_dagger)
   → population statistics: correlate the inferred α, y_eff and gradient
      against outer-disk kinematics and RAR quantities, controlling for M★.
```

The "supernova mass distribution" is operationalised as the **high-mass slope α of a
broken-power-law IMF**. A flatter (smaller-α, "top-heavy") IMF produces relatively more
massive stars, which are the core-collapse supernova (CC-SN) progenitors that make oxygen.
So the oxygen budget of a disk is a lever on α — but, as Section 7 makes clear, a lever
that is partly shared with gas outflows.

"Orbit levels" are angular-momentum bins. A spaxel at deprojected radius `R` with local
rotation speed `V_rot` has specific angular momentum `j = R · V_rot`; spaxels are binned by
percentile of `j`, and the run reports `d(12+log O/H)/d(log j)` — the metal content as a
function of orbital level, which is the quantity the original request framed as "how much
heavy metals a galaxy has in their different orbit levels".

---

## 2. Data ingestion

Three interchangeable sources feed a common `GalaxyMaps` container (`ingest.py`), so every
downstream module is agnostic to provenance:

**Marvin (official SDSS tool).** When `sdss-marvin` is installed and configured, the
pipeline pulls DAP `MAPS` (emission-line fluxes + inverse variances, stellar kinematics)
and DRPall/DAPall metadata directly. This is the intended path for real science.

**Raw FITS (astropy).** For users who have downloaded DAP `MAPS` / DRP `LOGCUBE` products,
the FITS loader reads the same extensions with `astropy.io.fits`, with no Marvin dependency.

**Synthetic "mock" mode (`mock.py`).** Because the full MaNGA dataset is multi-terabyte
and cannot be downloaded inside a sandboxed session, the package ships a self-consistent
generator that builds each galaxy *forward* from injected ground truth (see Section 8).
This is what the demo and the test suite exercise. It is a validation harness, not data.

Sample selection (`sample.SampleCfg`) mirrors standard MaNGA practice: redshift
`0.01 < z < 0.15`, `9.0 < log M★ < 11.5`, and axis-ratio cuts `0.25 < b/a < 0.95` to reject
near-edge-on disks (deprojection unstable) and near-face-on disks (no rotation signal).

---

## 3. Gas-phase metallicity (`metallicity.py`)

**Dust correction.** Line maps are de-reddened from the Balmer decrement Hα/Hβ, assuming a
Case-B intrinsic ratio of 2.86, using either the Calzetti et al. (2000) starburst
attenuation curve (default) or the Cardelli, Clayton & Mathis (1989) Galactic law. `E(B−V)`
is derived per spaxel and clipped to physical, non-negative values.

**Star-forming selection.** Strong-line calibrations are only valid for H II-region-like
gas, so spaxels are classified on the [N II] BPT diagram and non-star-forming spaxels
(AGN/LINER/composite) are rejected. The default is the Kauffmann et al. (2003) demarcation
`log([O III]/Hβ) = 0.61/(log([N II]/Hα) − 0.05) + 1.30`, with everything at
`log([N II]/Hα) ≥ 0.05` (right of the vertical asymptote) treated as non-SF. The Kewley
et al. (2001) maximum-starburst line is also available.

**Strong-line abundances.** Five calibrations are selectable and return `12 + log(O/H)`:
PP04 N2 and O3N2 (Pettini & Pagel 2004), the Marino et al. (2013) recalibrations of both,
and Dopita et al. (2016). Per-spaxel uncertainties are propagated from line S/N with a
0.05 dex systematic floor for calibration scatter.

**Gradients.** A `1/σ²`-weighted linear fit of `12+log(O/H)` versus `R/Re` (within a
configurable `R < 2 Re`) yields the central abundance `oh0`, the gradient `grad` (dex per
`Re`), and the residual scatter. Oxygen mass fraction `Z_O` is obtained from `12+log(O/H)`
via the Asplund et al. (2009) solar anchor (`utils.oh_to_ZO`, with
`OH_sun = 8.69`, `Z_O,sun = 0.0057`).

---

## 4. Kinematics and orbit levels (`kinematics.py`)

Inclination is estimated from the axis ratio via `cos²i = (q² − q0²)/(1 − q0²)` with an
intrinsic disk thickness `q0 = 0.13`. The line-of-sight velocity field is fit ring-by-ring
with a first-order harmonic model `V_los = V_sys + V_rot·sin(i)·cos(θ)`, solved as one
global linear least-squares system with a shared systemic velocity and one `V_rot`
amplitude per ring. The outer rotation speed `V_flat`, outer velocity dispersion, and
`V/σ` are read off the outer rings.

**Orbit ("angular-momentum") levels.** Each spaxel is assigned specific angular momentum
`j = R · V_rot(R)`, and spaxels are binned into `n_orbit_levels` (default 6) by percentile
of `j`. The mean metallicity per level gives the orbit-level metal distribution, and the
run reports its slope `d(12+log O/H)/d(log j)`.

---

## 5. Chemical evolution and the yield → IMF link (`chemev.py`)

**Effective yield.** In a one-zone model under the instantaneous-recycling approximation,
the gas-phase oxygen mass fraction obeys

- closed box: `Z_O = y_O · ln(1/μ)`
- leaky box: `Z_O = [y_O / (1 + η)] · ln(1/μ)`

where `μ` is the gas fraction `M_gas/(M_gas+M★)` and `η` is the mass-loading of a
metal-carrying outflow. Inverting gives the **effective yield**
`y_eff = Z_O / ln(1/μ) = y_O/(1+η)`, computed robustly (median over star-forming spaxels
with a MAD-based error) in `inference.effective_yield_summary`.

**IMF-weighted oxygen yield.** The *true* yield `y_O(α)` is the oxygen mass ejected by
CC-SNe per unit mass of stars formed, integrated over a broken-power-law IMF with fixed
low/mid slopes (Kroupa-like: 1.3 on 0.08–0.5 M⊙, 2.3 on 0.5–1 M⊙) and a **free high-mass
slope α** above 1 M⊙:

```
y_O(α) = ∫_{8}^{100} M_O(m) φ(m|α) dm  /  ∫_{0.08}^{100} m φ(m|α) dm
```

The IMF is continuous across break masses by construction. `M_O(m)` and the CC iron yield
`M_Fe(m)` come from compact log-log interpolations of published CC-SN grids (Woosley &
Weaver 1995; Nomoto et al. 2006). `y_O` decreases monotonically with α (a top-heavier IMF
makes more oxygen), which is the physical lever the inference uses.

**[α/Fe] as a second constraint.** Oxygen is a near-pure massive-star product; iron has a
large delayed Type-Ia contribution. The forward model `forward_alpha_fe(α, τ)` uses
`[O/Fe]` as an `[α/Fe]` proxy, with the Type-Ia normalisation calibrated so that
`[O/Fe] = 0` at `(α = 2.35, τ = 3 Gyr)`. `[O/Fe]` rises for a top-heavier IMF (smaller α)
and a shorter star-formation timescale τ. Crucially, outflows remove O and Fe together and
largely cancel in the ratio, which is why `[α/Fe]` helps break the α–η degeneracy.

The oxygen/iron/Type-Ia integrals are smooth in α, so they are tabulated once on an α grid
(0.8–3.6, 141 points) and interpolated inside the sampler for speed.

---

## 6. Bayesian inference of the supernova mass distribution (`inference.py`)

Per galaxy, the free parameters are α, `log₁₀η`, and — only when an `[α/Fe]` measurement is
supplied — τ. The likelihood is Gaussian in the observables:

```
y_eff_obs   ~ N( y_O(α)/(1+η),          σ_yeff )
[α/Fe]_obs  ~ N( forward_alpha_fe(α,τ),  σ_afe )     (optional)
```

Priors are explicit and physically motivated:

| Parameter | Prior | Support |
|---|---|---|
| α (IMF high-mass slope) | TruncNormal(2.35, 0.6) | [1.0, 3.5] |
| log₁₀η (outflow loading) | Normal(log₁₀ 0.3, 0.5) | [−2, 1] → η ∈ [0.01, 10] |
| τ (SF e-folding time, Gyr) | TruncNormal(3, 2) | [0.5, 12] |

The α prior is Salpeter-centred; the η prior encodes a typical mass-loading of order unity.
Inference uses a SciPy L-BFGS-B MAP estimate as a starting point, followed by a
dependency-free NumPy random-walk Metropolis sampler (6000 steps, 1500 burn-in). The
posterior median and 16th/84th percentiles are reported for α, along with the **posterior
α–η correlation coefficient** so the degeneracy is quantified for every galaxy, not hidden.

---

## 7. Scientific limitations (read this before trusting a number)

This is the most important section. The pipeline is internally consistent and recovers its
own injected truths on mock data, but several assumptions stand between it and a defensible
statement about real galaxies.

**(a) The α–η degeneracy is fundamental, not a bug.** The effective yield constrains a
single combination, `y_O(α)/(1+η)`. From `y_eff` alone, a top-heavy IMF (small α, large
`y_O`) and weak outflows are *indistinguishable* from a normal IMF with strong outflows.
The posterior is a "banana" in the α–η plane. The `[α/Fe]` constraint tightens α because it
is nearly outflow-independent, but it introduces its own dependence on the star-formation
timescale τ and on the assumed Type-Ia delay-time normalisation. The practical consequence,
visible in the mock recovery, is that `y_eff` is recovered tightly while α is recovered only
with a positive-but-modest correlation to truth. **Any α reported without an `[α/Fe]`
constraint should be read as a statement about `y_O/(1+η)`, not about the IMF.**

**(b) The yield tables are approximations.** `M_O(m)` and `M_Fe(m)` are compact log-log
interpolations of published grids, not a nucleosynthesis network. Real CC-SN yields depend
on metallicity, rotation, explosion energy, mass cut, and the (uncertain) mass range that
collapses directly to black holes without ejecting its oxygen. These choices shift `y_O(α)`
by factors that map directly onto systematic α offsets. The tables are deliberately easy to
swap; the inference is only as good as the grid it is handed.

**(c) The leaky-box / instantaneous-recycling model is a caricature.** Real disks have
radial gas inflows, time-varying accretion, non-instantaneous recycling, and radial
migration of stars — none of which are modelled. The gas fraction μ is itself uncertain:
here it is derived from Σ_gas and Σ_star maps, and Σ_gas in real MaNGA data must be inferred
(e.g. from dust or an assumed molecular-gas scaling) rather than measured directly. Errors
in μ propagate straight into `y_eff` through `ln(1/μ)`, which is steep as μ → 1.

**(d) Strong-line abundances carry large systematic offsets.** Different calibrations
disagree by up to ~0.6 dex in absolute `12+log(O/H)`, and the shape of the mass–metallicity
relation depends on the calibration chosen. The pipeline lets you switch calibrations
precisely so this systematic can be bracketed; it does not remove it. Gradients (relative
abundances) are more robust than absolute normalisations.

**(e) Kinematic simplifications.** The rotation model is a first-order harmonic (pure
rotation, no explicit non-circular or warp terms), inclination comes from a single-`q0`
axis-ratio estimate, and there is no asymmetric-drift correction for the stellar tracer.
Each biases `V_rot`, and therefore both the orbit-level assignment (through `j`) and the RAR
(through `g_obs ∝ V_rot²`).

**(f) The RAR is not an independent dark-matter test here.** `g_bar` is built from the same
baryonic surface-density maps used elsewhere, under a spherical-enclosed-mass (or thin-disk)
approximation and an assumed stellar M/L. The recovered `g_dagger` therefore reflects those
modelling choices; agreement with the McGaugh et al. (2016) value is a consistency check on
the mock, not a measurement.

**(g) Correlation is not causation, and look-elsewhere effects are real.** The population
module tests many pairs. Partial correlations control for `log M★`, and bootstrap CIs are
reported, but with a modest sample and many tested pairs the usual multiple-comparisons
cautions apply. A significant `p` is a prompt for follow-up, not a discovery.

---

## 8. What the mock does and does not prove (`mock.py`)

Each mock galaxy is generated **forward** from injected truths — `α_true`, `η_true`,
`τ_true`, a gas-fraction profile μ(R), an injected RAR scale `g_dagger,true`, and geometry —
so that `Z_O(R) = y_eff · ln(1/μ(R))` holds by construction and the emission-line maps are
synthesised from the PP04 (N2 and O3N2) inverse relations. (Running the demo with a
different calibration — e.g. `M13` or `D16` — therefore recovers an O/H offset slightly from
the injected value by the known calibration-to-calibration difference; this is expected, and
is itself a compact illustration of the strong-line systematic discussed in §7d.) Population
trends
(e.g. a top-heavier IMF in more massive, faster-rotating disks) are injected *on purpose* so
the statistics module can be checked for its ability to *recover a known correlation*.

This validates the **software and the estimators**: it shows the code inverts its own
generative model, propagates uncertainties sensibly, and recovers injected `y_eff`,
gradient, `g_dagger` tightly and α more loosely (exactly as the degeneracy predicts). It
proves **nothing about real galaxies** — the mock cannot contain physics the model does not
already assume. The `--no-alpha-fe` demo flag deliberately widens the α–η banana so you can
*see* the degeneracy open up when the second constraint is removed.

The test suite (`tests/test_pipeline.py`) encodes these expectations as tolerances:
`y_eff` recovered with `r > 0.8`, `g_dagger` with `r > 0.7`, gradient with `r > 0.6`, and α
with `r > 0.4` and `|bias| < 0.4` — the deliberately looser α threshold is the degeneracy,
made quantitative.

---

## 9. References

- Asplund, Grevesse, Sauval & Scott (2009), *ARA&A* 47, 481 — solar abundances.
- Calzetti et al. (2000), *ApJ* 533, 682 — starburst attenuation curve.
- Cardelli, Clayton & Mathis (1989), *ApJ* 345, 245 — Galactic extinction.
- Dopita et al. (2016), *Ap&SS* 361, 61 — strong-line abundance calibration.
- Kauffmann et al. (2003), *MNRAS* 346, 1055 — BPT star-forming demarcation.
- Kewley et al. (2001), *ApJ* 556, 121 — maximum-starburst line.
- Marino et al. (2013), *A&A* 559, A114 — N2 and O3N2 recalibrations.
- McGaugh, Lelli & Schombert (2016), *PRL* 117, 201101 — Radial Acceleration Relation.
- Nomoto et al. (2006), *Nucl. Phys. A* 777, 424 — core-collapse/hypernova yields.
- Pettini & Pagel (2004), *MNRAS* 348, L59 — PP04 N2 and O3N2 calibrations.
- Woosley & Weaver (1995), *ApJS* 101, 181 — massive-star nucleosynthesis grid.
