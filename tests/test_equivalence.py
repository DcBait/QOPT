"""Randomized correctness tests: optimization must never change the unitary."""
import pytest

from qopt import equivalent, preset
from qopt.generators import (compute_uncompute, mcx_vchain, qft, random_clifford_t,
                             random_redundant, trotter_ising)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("seed", range(150))
def test_random_circuits_preserve_unitary(seed, level):
    c = random_redundant(n=4, n_gates=40, seed=seed)
    out, _ = preset(level).run(c)
    assert len(out) <= len(c)
    assert equivalent(c, out)


@pytest.mark.parametrize("circ", [qft(5), trotter_ising(5, 4), mcx_vchain(3),
                                  compute_uncompute(5, 50, 7), random_clifford_t(5, 80, 3)],
                         ids=["qft", "trotter", "mcx", "compute_uncompute", "clifford_t"])
def test_benchmark_families_preserve_unitary(circ):
    out, _ = preset(3).run(circ)
    assert equivalent(circ, out)


def test_levels_are_monotone_on_compute_uncompute():
    c = compute_uncompute(6, 80, 11)
    sizes = [len(preset(level).run(c)[0]) for level in (0, 1, 2, 3)]
    assert sizes == sorted(sizes, reverse=True)


def test_idempotent():
    c = random_redundant(5, 60, seed=42)
    once = preset(3).run(c)[0]
    assert preset(3).run(once)[0].gates == once.gates


def test_simulator_detects_a_wrong_rewrite():
    # Guard against a vacuous equivalence checker.
    a = random_clifford_t(3, 20, 0)
    b = a.copy().add("t", 0)
    assert not equivalent(a, b)


def test_cross_check_with_qiskit():
    qb = pytest.importorskip("qopt.qiskit_bridge")
    for seed in range(20):
        c = random_redundant(4, 40, seed)
        assert qb.qiskit_equivalent(c, preset(3).run(c)[0])
