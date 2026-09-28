import math

import pytest

from qopt import Circuit, PeepholePass, commutes, equivalent, optimize
from qopt.ir import Gate
from qopt.passes import CancelInverses, MergeRotations


def names(c: Circuit) -> list[str]:
    return [g.name for g in c.gates]


def O3(c: Circuit) -> Circuit:
    return optimize(c, level=3)


# --- adjacent cancellation --------------------------------------------------

@pytest.mark.parametrize("a,b", [("h", "h"), ("x", "x"), ("s", "sdg"), ("t", "tdg"), ("y", "y")])
def test_single_qubit_inverse_pairs_cancel(a, b):
    c = Circuit(1).add(a, 0).add(b, 0)
    assert len(CancelInverses().run(c)) == 0


def test_nested_cancellation_cascades():
    c = Circuit(1).add("h", 0).add("x", 0).add("t", 0).add("tdg", 0).add("x", 0).add("h", 0)
    assert len(CancelInverses().run(c)) == 0


def test_cx_cancels_only_with_same_orientation():
    assert len(O3(Circuit(2).add("cx", 0, 1).add("cx", 0, 1))) == 0
    assert len(O3(Circuit(2).add("cx", 0, 1).add("cx", 1, 0))) == 2


def test_symmetric_gates_cancel_in_either_order():
    assert len(O3(Circuit(2).add("cz", 0, 1).add("cz", 1, 0))) == 0
    assert len(O3(Circuit(2).add("swap", 0, 1).add("swap", 1, 0))) == 0


def test_blocked_by_noncommuting_gate():
    c = Circuit(1).add("t", 0).add("h", 0).add("tdg", 0)
    assert names(O3(c)) == ["t", "h", "tdg"]


# --- rotation merging ------------------------------------------------------

def test_rz_merge_sums_angles():
    c = Circuit(1).add("rz", 0, param=0.3).add("rz", 0, param=0.4)
    out = MergeRotations().run(c)
    assert names(out) == ["rz"] and out.gates[0].params[0] == pytest.approx(0.7)


def test_t_t_becomes_s_and_s_s_becomes_z():
    assert names(O3(Circuit(1).add("t", 0).add("t", 0))) == ["s"]
    assert names(O3(Circuit(1).add("s", 0).add("s", 0))) == ["z"]


def test_rotation_summing_to_identity_is_removed():
    c = Circuit(1).add("rx", 0, param=1.2).add("rx", 0, param=2 * math.pi - 1.2)
    assert len(O3(c)) == 0


def test_zero_rotation_removed():
    assert len(O3(Circuit(1).add("rz", 0, param=0.0))) == 0


def test_different_axes_do_not_merge():
    c = Circuit(1).add("rx", 0, param=0.3).add("rz", 0, param=0.3)
    assert len(O3(c)) == 2


# --- commutation ------------------------------------------------------------

def test_z_rotation_commutes_through_cx_control():
    c = Circuit(2).add("t", 0).add("cx", 0, 1).add("tdg", 0)
    assert names(O3(c)) == ["cx"]


def test_x_commutes_through_cx_target():
    c = Circuit(2).add("x", 1).add("cx", 0, 1).add("x", 1)
    assert names(O3(c)) == ["cx"]


def test_z_does_not_commute_through_cx_target():
    c = Circuit(2).add("z", 1).add("cx", 0, 1).add("z", 1)
    assert len(O3(c)) == 3


def test_cx_cancels_through_commuting_gates():
    c = Circuit(3).add("cx", 0, 1).add("t", 0).add("cx", 0, 2).add("rx", 1, param=0.2).add("cx", 0, 1)
    out = O3(c)
    assert names(out).count("cx") == 1 and equivalent(c, out)


def test_commutes_rules():
    assert commutes(Gate("cx", (0, 1)), Gate("cx", (0, 2)))      # shared control
    assert commutes(Gate("cx", (0, 2)), Gate("cx", (1, 2)))      # shared target
    assert not commutes(Gate("cx", (0, 1)), Gate("cx", (1, 2)))  # target meets control
    assert commutes(Gate("cz", (0, 1)), Gate("cx", (0, 2)))
    assert not commutes(Gate("h", (0,)), Gate("z", (0,)))


def test_without_commute_flag_no_lookthrough():
    c = Circuit(2).add("t", 0).add("cx", 0, 1).add("tdg", 0)
    assert len(PeepholePass(commute=False).run(c)) == 3
