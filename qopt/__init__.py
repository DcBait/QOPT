"""qopt - a small peephole optimizer for quantum circuits."""
from .ir import Circuit, Gate
from .manager import PassManager, optimize, preset
from .passes import CancelInverses, MergeRotations, PeepholePass, commutes
from .sim import equivalent, unitary

__all__ = ["Circuit", "Gate", "PassManager", "optimize", "preset", "CancelInverses",
           "MergeRotations", "PeepholePass", "commutes", "equivalent", "unitary"]
__version__ = "0.1.0"
