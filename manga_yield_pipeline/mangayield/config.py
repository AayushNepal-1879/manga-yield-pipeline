"""
Configuration objects for the mangayield pipeline.

Everything has a sane default so the pipeline runs with zero configuration.
A YAML file (see config/default.yaml) can override any field; PyYAML is
imported lazily so the package still works if PyYAML is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import math


# --------------------------------------------------------------------------
# Physical constants (SI unless noted). Kept local to avoid unit surprises.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Constants:
    G: float = 6.67430e-11            # m^3 kg^-1 s^-2
    Msun_kg: float = 1.98892e30       # kg
    pc_m: float = 3.0856775814913673e16  # m
    kpc_m: float = 3.0856775814913673e19  # m
    km_m: float = 1.0e3
    # Solar abundance anchors (Asplund et al. 2009)
    OH_sun: float = 8.69              # 12 + log(O/H)_sun
    Z_O_sun: float = 0.0057           # solar oxygen *mass* fraction
    X_H: float = 0.7381               # solar hydrogen mass fraction
    # MaNGA instrument
    spaxel_arcsec: float = 0.5        # reconstructed spaxel size


CONST = Constants()


@dataclass
class CosmologyCfg:
    """Flat LCDM used for arcsec->kpc and luminosity/mass scaling."""
    H0: float = 70.0      # km/s/Mpc
    Om0: float = 0.3
    Ode0: float = 0.7


@dataclass
class SampleCfg:
    """Selection cuts applied to the DRPall/DAPall catalogue."""
    z_min: float = 0.01
    z_max: float = 0.15
    logmstar_min: float = 9.0
    logmstar_max: float = 11.5
    ba_min: float = 0.25          # reject near-edge-on (deprojection unstable)
    ba_max: float = 0.95          # reject near-face-on (no rotation signal)
    require_dap: bool = True
    daptype: str = "HYB10-MILESHC-MASTARSSP"  # DAP binning scheme
    max_galaxies: Optional[int] = None        # None = all


@dataclass
class MetallicityCfg:
    calibration: str = "PP04_O3N2"   # PP04_O3N2 | PP04_N2 | M13_O3N2 | M13_N2 | D16
    sn_min: float = 3.0              # per-line S/N floor
    ha_hb_min: float = 2.86          # theoretical Balmer decrement (Case B)
    extinction_law: str = "calzetti"  # calzetti | cardelli
    bpt_class: str = "kauffmann"     # kauffmann (SF only) | kewley (SF+composite) | none
    fit_gradient_rmax_re: float = 2.0  # radial fit range in units of Re


@dataclass
class KinematicsCfg:
    tracer: str = "stellar"          # stellar | gas (Halpha)
    q0: float = 0.13                 # intrinsic disk thickness for inclination
    n_rings: int = 12                # radial rings for rotation-curve fit
    rmax_re: float = 2.2             # outer edge of ring fit in Re
    min_spx_per_ring: int = 12
    outer_frac: float = 0.5          # "outer disk" = R/Re above this fraction of rmax
    n_orbit_levels: int = 6          # angular-momentum bins


@dataclass
class YieldCfg:
    """Chemical-evolution + IMF-yield model settings."""
    box_model: str = "leaky"         # closed | leaky
    imf_low_slope: float = 1.3       # fixed low-mass Kroupa slope (0.08-0.5 Msun)
    imf_mid_slope: float = 2.3       # fixed mid slope (0.5-1 Msun)
    imf_break_masses: tuple = (0.08, 0.5, 1.0)
    m_min: float = 0.08              # Msun
    m_max: float = 100.0             # Msun
    sn_mass_min: float = 8.0         # CC-SN progenitor floor
    m_collapse: float = 100.0        # above this, direct collapse (no O ejection)
    return_fraction: float = 0.4     # R, mass returned to ISM (approx, IMF-avg)


@dataclass
class InferenceCfg:
    # Priors on the free parameters.
    alpha_prior_mean: float = 2.35   # Salpeter high-mass slope
    alpha_prior_sigma: float = 0.6
    alpha_bounds: tuple = (1.0, 3.5)
    logeta_prior_mean: float = math.log10(0.3)  # mass-loading ~ 0.3
    logeta_prior_sigma: float = 0.5
    logeta_bounds: tuple = (-2.0, 1.0)           # eta in [0.01, 10]
    use_alpha_fe: bool = True        # include [alpha/Fe] constraint if available
    tau_prior_mean: float = 3.0      # SF e-folding time, Gyr
    tau_prior_sigma: float = 2.0
    tau_bounds: tuple = (0.5, 12.0)
    # MCMC
    n_walkers_steps: int = 6000
    n_burn: int = 1500
    mcmc_step: float = 0.08
    seed: int = 12345


@dataclass
class RARCfg:
    gbar_model: str = "spherical"    # spherical (enclosed mass) | thindisk
    ml_stellar: float = 1.0          # stellar M/L already applied if using Sigma_star maps
    include_gas: bool = True
    fit_gdagger: bool = True
    gdagger_guess: float = 1.2e-10   # m/s^2 (McGaugh+2016 ~ 1.2e-10)


@dataclass
class Config:
    cosmology: CosmologyCfg = field(default_factory=CosmologyCfg)
    sample: SampleCfg = field(default_factory=SampleCfg)
    metallicity: MetallicityCfg = field(default_factory=MetallicityCfg)
    kinematics: KinematicsCfg = field(default_factory=KinematicsCfg)
    yields: YieldCfg = field(default_factory=YieldCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    rar: RARCfg = field(default_factory=RARCfg)
    outdir: str = "results"
    n_jobs: int = 1                  # reserved for future parallelism

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTION_TYPES = {
    "cosmology": CosmologyCfg,
    "sample": SampleCfg,
    "metallicity": MetallicityCfg,
    "kinematics": KinematicsCfg,
    "yields": YieldCfg,
    "inference": InferenceCfg,
    "rar": RARCfg,
}


def load_config(path: Optional[str] = None) -> Config:
    """Load a Config, optionally overriding defaults from a YAML file."""
    cfg = Config()
    if path is None:
        return cfg
    try:
        import yaml  # lazy: PyYAML optional
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read a config file; install pyyaml or pass path=None"
        ) from exc
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    for section, subtype in _SECTION_TYPES.items():
        if section in raw and isinstance(raw[section], dict):
            current = getattr(cfg, section)
            for k, v in raw[section].items():
                if hasattr(current, k):
                    setattr(current, k, v)
    for scalar in ("outdir", "n_jobs"):
        if scalar in raw:
            setattr(cfg, scalar, raw[scalar])
    return cfg
