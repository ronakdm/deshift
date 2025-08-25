"""Deshift: distributionally robust optimization for PyTorch."""

from ._src.pav import (
    l2_centered_isotonic_regression,
    neg_entropy_centered_isotonic_regression,
)
from ._src.spectral_risk import (
    make_spectral_risk_measure,
    spectral_risk_measure_maximization_oracle,
    make_group_spectral_risk_measure,  
)
from ._src.spectra import (
    make_esrm_spectrum,
    make_extremile_spectrum,
    make_superquantile_spectrum,
)
from ._src.distributed import ddp_max_oracle

__version__ = "0.0.1.dev"

__all__ = [
    "l2_centered_isotonic_regression",
    "neg_entropy_centered_isotonic_regression",
    "make_esrm_spectrum",
    "make_extremile_spectrum",
    "make_spectral_risk_measure",
    "make_superquantile_spectrum",
    "spectral_risk_measure_maximization_oracle",
    "make_group_spectral_risk_measure",
    "ddp_max_oracle",
]