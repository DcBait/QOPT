"""Conversion to and from Qiskit (optional dependency).

Used to load OpenQASM 2 files, to cross-check correctness with an independent
simulator, and to compare against Qiskit's own transpiler.
"""
from __future__ import annotations

from .ir import ALL_GATES, Circuit

BASIS = sorted(ALL_GATES)


def to_qiskit(circ: Circuit):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(circ.n_qubits)
    for g in circ.gates:
        getattr(qc, g.name)(*g.params, *g.qubits)
    return qc


def from_qiskit(qc) -> Circuit:
    """Convert a Qiskit circuit, lowering unsupported gates (ccx, cp, u, ...) to the qopt basis first."""
    from qiskit import transpile

    if any(inst.operation.name not in ALL_GATES for inst in qc.data):
        qc = transpile(qc, basis_gates=BASIS, optimization_level=0)
    circ = Circuit(qc.num_qubits)
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        param = float(op.params[0]) if op.params else None
        circ.add(op.name, *qubits, param=param)
    return circ


def load_qasm(path: str) -> Circuit:
    from qiskit import qasm2

    return from_qiskit(qasm2.load(path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))


def to_qasm(circ: Circuit) -> str:
    from qiskit import qasm2

    return qasm2.dumps(to_qiskit(circ))


def qiskit_optimize(circ: Circuit, level: int) -> Circuit:
    from qiskit import transpile

    out = transpile(to_qiskit(circ), basis_gates=BASIS, optimization_level=level, seed_transpiler=0)
    return from_qiskit(out)


def qiskit_equivalent(a: Circuit, b: Circuit) -> bool:
    """Independent equivalence check using Qiskit's Operator class."""
    from qiskit.quantum_info import Operator

    return Operator(to_qiskit(a)).equiv(Operator(to_qiskit(b)))
