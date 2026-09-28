import pytest

from qopt import Circuit, equivalent
from qopt.ir import Gate


def test_depth_counts_parallel_gates_once():
    c = Circuit(3).add("h", 0).add("h", 1).add("h", 2).add("cx", 0, 1).add("x", 2)
    assert c.depth() == 2


def test_inverse_circuit_undoes_circuit():
    c = Circuit(2).add("h", 0).add("t", 0).add("cx", 0, 1).add("rz", 1, param=0.3).add("s", 1)
    assert equivalent(c.copy().extend(c.inverse()), Circuit(2))


def test_t_count():
    import math
    c = Circuit(1).add("t", 0).add("tdg", 0).add("s", 0).add("rz", 0, param=3 * math.pi / 4)
    assert c.t_count() == 3


@pytest.mark.parametrize("bad", [lambda: Gate("cx", (0,)), lambda: Gate("cx", (1, 1)),
                                 lambda: Gate("rz", (0,)), lambda: Gate("foo", (0,))])
def test_invalid_gates_rejected(bad):
    with pytest.raises(ValueError):
        bad()


def test_known_identity_hxh_equals_z():
    assert equivalent(Circuit(1).add("h", 0).add("x", 0).add("h", 0), Circuit(1).add("z", 0))
