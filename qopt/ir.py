"""A minimal intermediate representation (IR) for quantum circuits.

A circuit is a list of gates in program order.  Each gate has a name, a tuple of
qubit indices and (for rotations) a tuple of angle parameters.  The IR is
deliberately small so that every optimization pass is easy to reason about.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

# Gate set --------------------------------------------------------------------

SINGLE_QUBIT = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz"}
TWO_QUBIT = {"cx", "cz", "swap"}
PARAMETRIC = {"rx", "ry", "rz"}
ALL_GATES = SINGLE_QUBIT | TWO_QUBIT

SELF_INVERSE = {"h", "x", "y", "z", "cx", "cz", "swap"}
INVERSE_NAME = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
SYMMETRIC = {"cz", "swap"}  # gates whose qubit order does not matter

# Diagonal (Z-axis) phase gates expressed as RZ angles, equal up to global phase.
Z_ANGLE = {"z": math.pi, "s": math.pi / 2, "sdg": -math.pi / 2,
           "t": math.pi / 4, "tdg": -math.pi / 4}


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in ALL_GATES:
            raise ValueError(f"unsupported gate {self.name!r}")
        arity = 1 if self.name in SINGLE_QUBIT else 2
        if len(self.qubits) != arity:
            raise ValueError(f"{self.name} expects {arity} qubit(s), got {self.qubits}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{self.name} applied to repeated qubits {self.qubits}")
        n_params = 1 if self.name in PARAMETRIC else 0
        if len(self.params) != n_params:
            raise ValueError(f"{self.name} expects {n_params} parameter(s)")

    def inverse(self) -> "Gate":
        if self.name in SELF_INVERSE:
            return self
        if self.name in INVERSE_NAME:
            return Gate(INVERSE_NAME[self.name], self.qubits)
        return Gate(self.name, self.qubits, (-self.params[0],))

    def __str__(self) -> str:
        p = f"({self.params[0]:.4f})" if self.params else ""
        return f"{self.name}{p} " + ",".join(f"q{q}" for q in self.qubits)


@dataclass
class Circuit:
    n_qubits: int
    gates: list[Gate] = field(default_factory=list)

    # construction ------------------------------------------------------------
    def add(self, name: str, *qubits: int, param: float | None = None) -> "Circuit":
        for q in qubits:
            if not 0 <= q < self.n_qubits:
                raise ValueError(f"qubit {q} out of range for {self.n_qubits}-qubit circuit")
        params = () if param is None else (float(param),)
        self.gates.append(Gate(name, tuple(qubits), params))
        return self

    def append(self, gate: Gate) -> "Circuit":
        return self.add(gate.name, *gate.qubits, param=gate.params[0] if gate.params else None)

    def extend(self, other: "Circuit") -> "Circuit":
        if other.n_qubits != self.n_qubits:
            raise ValueError("qubit count mismatch")
        self.gates.extend(other.gates)
        return self

    def inverse(self) -> "Circuit":
        return Circuit(self.n_qubits, [g.inverse() for g in reversed(self.gates)])

    def copy(self) -> "Circuit":
        return Circuit(self.n_qubits, list(self.gates))

    # metrics -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gates)

    def count_ops(self) -> dict[str, int]:
        return dict(Counter(g.name for g in self.gates))

    def two_qubit_count(self) -> int:
        return sum(1 for g in self.gates if len(g.qubits) == 2)

    def t_count(self) -> int:
        """Number of non-Clifford Z rotations (T, T-dagger, or RZ by an odd multiple of pi/4)."""
        count = 0
        for g in self.gates:
            if g.name in ("t", "tdg"):
                count += 1
            elif g.name == "rz":
                k = g.params[0] / (math.pi / 4)
                if abs(k - round(k)) < 1e-9 and round(k) % 2 == 1:
                    count += 1
        return count

    def depth(self) -> int:
        """Length of the critical path when gates on disjoint qubits run in parallel."""
        level = [0] * self.n_qubits
        for g in self.gates:
            d = max(level[q] for q in g.qubits) + 1
            for q in g.qubits:
                level[q] = d
        return max(level, default=0)

    def metrics(self) -> dict[str, int]:
        return {"gates": len(self), "two_qubit": self.two_qubit_count(),
                "depth": self.depth(), "t_count": self.t_count()}

    def __str__(self) -> str:
        return f"Circuit({self.n_qubits} qubits)\n" + "\n".join(f"  {g}" for g in self.gates)
