"""Dense unitary simulation, used to *prove* that an optimization preserved behaviour.

Qubit 0 is the most significant bit.  Practical up to ~10 qubits.
"""
from __future__ import annotations

import math

import numpy as np

from .ir import Circuit, Gate

_S2 = 1 / math.sqrt(2)
_FIXED = {
    "h": np.array([[_S2, _S2], [_S2, -_S2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.diag([1, -1]).astype(complex),
    "s": np.diag([1, 1j]),
    "sdg": np.diag([1, -1j]),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]),
    "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]),
    "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
    "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
}


def gate_matrix(g: Gate) -> np.ndarray:
    if g.name in _FIXED:
        return _FIXED[g.name]
    th = g.params[0]
    c, s = math.cos(th / 2), math.sin(th / 2)
    if g.name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if g.name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if g.name == "rz":
        return np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)])
    raise ValueError(g.name)


def unitary(circ: Circuit) -> np.ndarray:
    n = circ.n_qubits
    dim = 2 ** n
    # Tensor with one axis per qubit plus a trailing "column" axis.
    u = np.eye(dim, dtype=complex).reshape([2] * n + [dim])
    for g in circ.gates:
        k = len(g.qubits)
        m = gate_matrix(g).reshape([2] * (2 * k))
        u = np.tensordot(m, u, axes=(list(range(k, 2 * k)), list(g.qubits)))
        u = np.moveaxis(u, list(range(k)), list(g.qubits))
    return u.reshape(dim, dim)


def equivalent(a: Circuit, b: Circuit, atol: float = 1e-8) -> bool:
    """True if the circuits implement the same unitary up to a global phase."""
    if a.n_qubits != b.n_qubits:
        return False
    ua, ub = unitary(a), unitary(b)
    overlap = np.vdot(ua, ub)  # = trace(ua^dagger ub)
    if abs(overlap) < 1e-12:
        return False
    phase = overlap / abs(overlap)
    return bool(np.allclose(ua * phase, ub, atol=atol))
