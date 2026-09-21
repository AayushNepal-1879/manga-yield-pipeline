"""
mangayield: a MaNGA heavy-metal-yield -> supernova-mass-distribution pipeline.

An automated, SciPy/NumPy-based pipeline that ingests MaNGA spatially-resolved
spectroscopic maps, models gas-phase (and stellar) heavy-metal content across
galactic disks, infers a supernova mass distribution (the high-mass IMF slope)
from effective nucleosynthetic yields, resolves the disk into angular-momentum
"orbit levels", computes the Radial Acceleration Relation (RAR), and then hunts
for population-level statistical correlations between the inferred supernova
mass distribution, outer-disk kinematics, and the RAR.

The package is deliberately dependency-light: numpy + scipy + astropy +
matplotlib. `sdss-marvin` is optional and only needed for live SDSS access;
raw-FITS and synthetic ("mock") modes work with the core stack alone.

Scientific caveats live in docs/METHODOLOGY.md. Read them before trusting a
number. In particular, inferring an IMF slope from abundances is degenerate
with gas outflows (the alpha-eta degeneracy); this pipeline treats that
degeneracy explicitly rather than hiding it.
"""

from .config import Config, load_config
from .ingest import GalaxyMaps, load_galaxy, select_sample
from . import mock, metallicity, kinematics, chemev, inference, rar, stats, pipeline

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "GalaxyMaps",
    "load_galaxy",
    "select_sample",
    "mock",
    "metallicity",
    "kinematics",
    "chemev",
    "inference",
    "rar",
    "stats",
    "pipeline",
    "__version__",
]
