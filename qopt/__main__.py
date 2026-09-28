"""Command line: optimize an OpenQASM 2 file.

    python -m qopt examples/toffoli_pair.qasm --level 3 -o optimized.qasm
"""
from __future__ import annotations

import argparse

from .manager import preset
from .qiskit_bridge import load_qasm, to_qasm
from .sim import equivalent


def main() -> None:
    ap = argparse.ArgumentParser(prog="qopt", description=__doc__.splitlines()[0])
    ap.add_argument("qasm", help="input OpenQASM 2 file")
    ap.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("-o", "--output", help="write the optimized circuit here")
    ap.add_argument("--no-verify", action="store_true", help="skip the unitary equivalence check")
    args = ap.parse_args()

    circ = load_qasm(args.qasm)
    out, report = preset(args.level).run(circ)
    print(f"{'metric':<10}{'before':>8}{'after':>8}")
    for k in report.before:
        print(f"{k:<10}{report.before[k]:>8}{report.after[k]:>8}")
    print(f"{report.iterations} iteration(s), {report.seconds * 1000:.2f} ms")
    if not args.no_verify and circ.n_qubits <= 10:
        print("equivalent:", equivalent(circ, out))
    if args.output:
        with open(args.output, "w") as f:
            f.write(to_qasm(out))
        print("wrote", args.output)


if __name__ == "__main__":
    main()
