"""Pass manager: runs a pipeline of passes repeatedly until nothing changes."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .ir import Circuit
from .passes import CancelInverses, MergeRotations, PeepholePass


@dataclass
class RunReport:
    before: dict[str, int]
    after: dict[str, int]
    iterations: int
    seconds: float
    per_pass: list[tuple[str, int]] = field(default_factory=list)  # (pass name, gates removed)


class PassManager:
    def __init__(self, passes, max_iterations: int = 20):
        self.passes = list(passes)
        self.max_iterations = max_iterations

    def run(self, circ: Circuit) -> tuple[Circuit, RunReport]:
        start = time.perf_counter()
        before = circ.metrics()
        per_pass: list[tuple[str, int]] = []
        iterations = 0
        current = circ
        for iterations in range(1, self.max_iterations + 1):
            snapshot = current.gates
            for p in self.passes:
                n = len(current)
                current = p.run(current)
                per_pass.append((p.name, n - len(current)))
            if current.gates == snapshot:  # fixed point reached
                break
        report = RunReport(before, current.metrics(), iterations,
                           time.perf_counter() - start, per_pass)
        return current, report


def preset(level: int) -> PassManager:
    """Optimization levels used in the benchmarks.

    O1 - cancel adjacent inverse gates
    O2 - O1 + merge adjacent rotations
    O3 - O2, but looking through commuting gates
    """
    if level == 0:
        return PassManager([])
    if level == 1:
        return PassManager([CancelInverses()])
    if level == 2:
        return PassManager([CancelInverses(), MergeRotations()])
    if level == 3:
        return PassManager([PeepholePass(cancel=True, merge=True, commute=True)])
    raise ValueError(f"unknown level {level}")


def optimize(circ: Circuit, level: int = 3) -> Circuit:
    return preset(level).run(circ)[0]
