"""
SU(3) Gauge-Covariant Resonance GNN v3.4
Gluon = tactile | Real photon = words | Virtual photon = thinking
"""

from .core import (
    SU3GaugeField,
    SU3ResonanceLayer,
    SU3GaugeBlock,
    SU3ResonanceGNN,
    ResonanceDiagnostics,
    GaugeDiagnostics,
    get_gell_mann_matrices,
    make_undirected,
)

__version__ = "3.4.0"
__all__ = [
    "SU3GaugeField",
    "SU3ResonanceLayer", 
    "SU3GaugeBlock",
    "SU3ResonanceGNN",
    "ResonanceDiagnostics",
    "GaugeDiagnostics",
    "get_gell_mann_matrices",
    "make_undirected",
]
