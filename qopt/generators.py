"""Seeded circuit generators used by the tests and the benchmark suite."""
from __future__ import annotations

import math

import numpy as np

from .ir import Circuit

CLIFFORD_T = ["h", "s", "sdg", "t", "tdg", "x", "z", "cx"]


def ccx(c: Circuit, a: int, b: int, t: int) -> Circuit:
    """Toffoli via the standard 7-T, 6-CNOT Clifford+T decomposition."""
    c.add("h", t).add("cx", b, t).add("tdg", t).add("cx", a, t).add("t", t)
    c.add("cx", b, t).add("tdg", t).add("cx", a, t).add("t", b).add("t", t)
    c.add("h", t).add("cx", a, b).add("t", a).add("tdg", b).add("cx", a, b)
    return c


def cphase(c: Circuit, theta: float, a: int, b: int) -> Circuit:
    """Controlled phase via RZ + CNOT (one common textbook lowering)."""
    c.add("rz", a, param=theta / 2).add("cx", a, b).add("rz", b, param=-theta / 2)
    c.add("cx", a, b).add("rz", b, param=theta / 2)
    return c


def random_clifford_t(n: int, n_gates: int, seed: int) -> Circuit:
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    for _ in range(n_gates):
        name = CLIFFORD_T[rng.integers(len(CLIFFORD_T))]
        if name == "cx":
            a, b = rng.choice(n, 2, replace=False)
            c.add("cx", int(a), int(b))
        else:
            c.add(name, int(rng.integers(n)))
    return c


def compute_uncompute(n: int, n_gates: int, seed: int) -> Circuit:
    """U · oracle · U^-1, the shape of many oracle-based algorithms."""
    u = random_clifford_t(n, n_gates, seed)
    c = u.copy()
    c.add("cz", 0, n - 1)
    return c.extend(u.inverse())


def qft(n: int) -> Circuit:
    c = Circuit(n)
    for i in range(n):
        c.add("h", i)
        for j in range(i + 1, n):
            cphase(c, math.pi / 2 ** (j - i), j, i)
    return c


def trotter_ising(n: int, steps: int, dt: float = 0.1, j: float = 1.0, hx: float = 0.7) -> Circuit:
    """First-order Trotterization of H = J·sum Z_i Z_{i+1} + h·sum X_i."""
    c = Circuit(n)
    for _ in range(steps):
        for i in range(n - 1):
            c.add("cx", i, i + 1).add("rz", i + 1, param=2 * j * dt).add("cx", i, i + 1)
        for i in range(n):
            c.add("rx", i, param=2 * hx * dt)
    return c


def mcx_vchain(k: int) -> Circuit:
    """k-controlled X using k-2 clean ancillas (V-chain), with Toffolis decomposed."""
    if k < 3:
        raise ValueError("k must be >= 3")
    controls = list(range(k))
    anc = list(range(k, 2 * k - 2))
    target = 2 * k - 2
    c = Circuit(2 * k - 1)
    chain = [(controls[0], controls[1], anc[0])]
    for i in range(2, k - 1):
        chain.append((controls[i], anc[i - 2], anc[i - 1]))
    for a, b, t in chain:
        ccx(c, a, b, t)
    ccx(c, controls[-1], anc[-1], target)
    for a, b, t in reversed(chain):
        ccx(c, a, b, t)
    return c


def random_redundant(n: int, n_gates: int, seed: int, redundancy: float = 0.35) -> Circuit:
    """Random circuit over the full gate set with injected inverse / rotation pairs.

    Used by the randomized correctness tests.
    """
    rng = np.random.default_rng(seed)
    names = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz", "swap"]
    c = Circuit(n)
    while len(c) < n_gates:
        if c.gates and rng.random() < redundancy:
            g = c.gates[-int(rng.integers(1, min(4, len(c)) + 1))]
            if g.params and rng.random() < 0.5:
                c.add(g.name, *g.qubits, param=float(rng.uniform(-math.pi, math.pi)))
            else:
                c.append(g.inverse())
            continue
        name = names[rng.integers(len(names))]
        if name in ("cx", "cz", "swap"):
            if n < 2:
                continue
            a, b = rng.choice(n, 2, replace=False)
            c.add(name, int(a), int(b))
        elif name in ("rx", "ry", "rz"):
            angle = float(rng.choice([rng.uniform(-math.pi, math.pi), math.pi / 4 * rng.integers(-8, 9)]))
            c.add(name, int(rng.integers(n)), param=angle)
        else:
            c.add(name, int(rng.integers(n)))
    return c
