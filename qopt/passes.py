"""Optimization passes.

All passes share one peephole engine.  It streams gates in program order and,
for each incoming gate, looks backwards along the qubit "wires" for an earlier
gate it can combine with:

* **cancel**  - an earlier gate that is exactly its inverse (H·H, S·Sdg, CX·CX ...)
                is deleted together with the incoming gate;
* **merge**   - an earlier rotation about the same axis on the same qubit is
                fused (RZ(a)·RZ(b) -> RZ(a+b), T·T -> S, X·RX(a) -> RX(a+pi) ...).
                A fused rotation that becomes the identity is deleted;
* **commute** - while looking backwards the engine may step over gates that
                provably commute with the incoming gate, so cancellations are
                found even when the two gates are not adjacent.

Commutation rule (sound, not complete): on every qubit two gates share, each
gate must act "diagonally" in the same basis.  Z-basis actions are diagonal
phase gates, CZ, and the control of CX; X-basis actions are X/RX and the target
of CX.  Everything else (H, Y, RY, SWAP) is treated as non-commuting.

Every rewrite holds up to a global phase, which is physically unobservable.
"""
from __future__ import annotations

import math

from .ir import SYMMETRIC, Z_ANGLE, Circuit, Gate

EPS = 1e-9


# helpers ---------------------------------------------------------------------

def _norm(theta: float) -> float:
    """Map an angle into (-pi, pi]."""
    t = math.fmod(theta, 2 * math.pi)
    if t <= -math.pi:
        t += 2 * math.pi
    elif t > math.pi:
        t -= 2 * math.pi
    return t


def rotation(g: Gate) -> tuple[str, float] | None:
    """Return (axis, angle) if g is a single-qubit rotation, up to global phase."""
    if g.name == "rz":
        return "Z", g.params[0]
    if g.name in Z_ANGLE:
        return "Z", Z_ANGLE[g.name]
    if g.name == "rx":
        return "X", g.params[0]
    if g.name == "x":
        return "X", math.pi
    if g.name == "ry":
        return "Y", g.params[0]
    if g.name == "y":
        return "Y", math.pi
    return None


_NAMED_Z = [(math.pi, "z"), (math.pi / 2, "s"), (-math.pi / 2, "sdg"),
            (math.pi / 4, "t"), (-math.pi / 4, "tdg")]


def make_rotation(axis: str, theta: float, q: int) -> Gate | None:
    """Build the cheapest gate for a rotation, or None if it is the identity."""
    t = _norm(theta)
    if abs(t) < EPS:
        return None
    if axis == "Z":
        for angle, name in _NAMED_Z:
            if abs(t - angle) < EPS:
                return Gate(name, (q,))
        return Gate("rz", (q,), (t,))
    if abs(t - math.pi) < EPS:
        return Gate(axis.lower(), (q,))
    return Gate("r" + axis.lower(), (q,), (t,))


def _action(g: Gate, q: int) -> str | None:
    """Basis in which g acts diagonally on qubit q ('Z', 'X') or None."""
    if g.name == "cx":
        return "Z" if g.qubits[0] == q else "X"
    if g.name == "cz":
        return "Z"
    rot = rotation(g)
    if rot and rot[0] in ("Z", "X"):
        return rot[0]
    return None


def commutes(a: Gate, b: Gate) -> bool:
    for q in set(a.qubits) & set(b.qubits):
        ka = _action(a, q)
        if ka is None or ka != _action(b, q):
            return False
    return True


def _same_wires(a: Gate, b: Gate) -> bool:
    if a.name in SYMMETRIC and b.name == a.name:
        return set(a.qubits) == set(b.qubits)
    return a.qubits == b.qubits


def _is_inverse(a: Gate, b: Gate) -> bool:
    if not _same_wires(a, b):
        return False
    inv = b.inverse()
    if inv.name != a.name:
        return False
    if a.params:
        return abs(_norm(a.params[0] - inv.params[0])) < EPS
    return True


# engine ----------------------------------------------------------------------

class PeepholePass:
    """Configurable peephole optimizer. See module docstring."""

    def __init__(self, cancel: bool = True, merge: bool = True, commute: bool = True):
        self.cancel, self.merge, self.commute = cancel, merge, commute
        flags = [f for f, on in (("cancel", cancel), ("merge", merge), ("commute", commute)) if on]
        self.name = "peephole[" + "+".join(flags) + "]"

    def _combine(self, prev: Gate, g: Gate):
        """How prev and g combine: ('drop',), ('replace', gate) or None."""
        if self.merge and len(g.qubits) == 1 and prev.qubits == g.qubits:
            rp, rg = rotation(prev), rotation(g)
            if rp and rg and rp[0] == rg[0]:
                merged = make_rotation(rp[0], rp[1] + rg[1], g.qubits[0])
                return ("drop",) if merged is None else ("replace", merged)
        if self.cancel and _is_inverse(prev, g):
            return ("drop",)
        return None

    def run(self, circ: Circuit) -> Circuit:
        out: list[Gate | None] = []
        wires: list[list[int]] = [[] for _ in range(circ.n_qubits)]  # indices into out

        for g in circ.gates:
            rot = rotation(g)
            if self.merge and rot is not None:
                g = make_rotation(rot[0], rot[1], g.qubits[0])  # canonical form
                if g is None:  # e.g. RZ(0) or RX(2*pi)
                    continue

            partner, action = None, None
            for q in g.qubits:
                found = None
                for idx in reversed(wires[q]):
                    prev = out[idx]
                    act = self._combine(prev, g)
                    if act is not None:
                        found = (idx, act)
                        break
                    if self.commute and commutes(prev, g):
                        continue
                    break
                if found is None or (partner is not None and found[0] != partner):
                    partner = None
                    break
                partner, action = found

            if partner is None:
                out.append(g)
                for q in g.qubits:
                    wires[q].append(len(out) - 1)
            elif action[0] == "drop":
                for q in out[partner].qubits:
                    wires[q].remove(partner)
                out[partner] = None
            else:
                out[partner] = action[1]

        return Circuit(circ.n_qubits, [g for g in out if g is not None])


def CancelInverses(commute: bool = False) -> PeepholePass:
    return PeepholePass(cancel=True, merge=False, commute=commute)


def MergeRotations(commute: bool = False) -> PeepholePass:
    return PeepholePass(cancel=False, merge=True, commute=commute)
