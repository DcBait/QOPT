"""Reproducible benchmark: qopt O1/O2/O3 vs Qiskit's transpiler (levels 1-3).

Every circuit is generated from a fixed seed, every optimized circuit is checked
for unitary equivalence, and results are written to benchmarks/results/.

    python benchmarks/run.py
"""
from __future__ import annotations

import csv
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qiskit  # noqa: E402

from qopt import equivalent, preset  # noqa: E402
from qopt.generators import (compute_uncompute, mcx_vchain, qft,  # noqa: E402
                             random_clifford_t, trotter_ising)
from qopt.qiskit_bridge import qiskit_optimize  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

SUITE = [
    ("compute_uncompute", "cu_n6", lambda: compute_uncompute(6, 120, seed=1)),
    ("compute_uncompute", "cu_n8", lambda: compute_uncompute(8, 200, seed=2)),
    ("random_clifford_t", "rand_n6", lambda: random_clifford_t(6, 300, seed=3)),
    ("random_clifford_t", "rand_n8", lambda: random_clifford_t(8, 500, seed=4)),
    ("qft", "qft_n6", lambda: qft(6)),
    ("qft", "qft_n8", lambda: qft(8)),
    ("mcx_vchain", "mcx_k4", lambda: mcx_vchain(4)),
    ("mcx_vchain", "mcx_k5", lambda: mcx_vchain(5)),
    ("trotter_ising", "ising_n6", lambda: trotter_ising(6, 10)),
    ("trotter_ising", "ising_n8", lambda: trotter_ising(8, 10)),
]

OPTIMIZERS = [
    ("qopt-O1", lambda c: preset(1).run(c)[0]),
    ("qopt-O2", lambda c: preset(2).run(c)[0]),
    ("qopt-O3", lambda c: preset(3).run(c)[0]),
    ("qiskit-L1", lambda c: qiskit_optimize(c, 1)),
    ("qiskit-L2", lambda c: qiskit_optimize(c, 2)),
    ("qiskit-L3", lambda c: qiskit_optimize(c, 3)),
]


def timed(fn, circ, repeats=3):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(circ)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    for family, name, make in SUITE:
        circ = make()
        base = circ.metrics()
        rows.append({"family": family, "circuit": name, "qubits": circ.n_qubits,
                     "optimizer": "original", **base, "ms": 0.0, "verified": True})
        for opt_name, fn in OPTIMIZERS:
            out, secs = timed(fn, circ)
            ok = equivalent(circ, out)
            rows.append({"family": family, "circuit": name, "qubits": circ.n_qubits,
                         "optimizer": opt_name, **out.metrics(), "ms": round(secs * 1000, 2),
                         "verified": ok})
            print(f"{name:10s} {opt_name:10s} {base['gates']:5d} -> {out.metrics()['gates']:5d}"
                  f"  {secs * 1000:8.1f} ms  verified={ok}")
            if not ok:
                raise SystemExit(f"{opt_name} produced a non-equivalent circuit for {name}")

    fields = list(rows[0].keys())
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    write_markdown(rows)
    plot(rows)


def _reduction(rows, circuit, optimizer, metric="gates"):
    base = next(r for r in rows if r["circuit"] == circuit and r["optimizer"] == "original")[metric]
    val = next(r for r in rows if r["circuit"] == circuit and r["optimizer"] == optimizer)[metric]
    return 100 * (base - val) / base if base else 0.0


def write_markdown(rows) -> None:
    circuits = list(dict.fromkeys(r["circuit"] for r in rows))
    opts = [o for o, _ in OPTIMIZERS]
    lines = [f"Environment: Python {platform.python_version()}, Qiskit {qiskit.__version__}, "
             f"{platform.system()} {platform.machine()}. All outputs verified equivalent "
             "to the input unitary (up to global phase).", ""]
    for metric, title in [("gates", "Total gates"), ("two_qubit", "Two-qubit gates"),
                          ("t_count", "T-count"), ("depth", "Depth")]:
        lines += [f"### {title}", "", "| circuit | qubits | original | " + " | ".join(opts) + " |",
                  "|---|---:|---:|" + "---:|" * len(opts)]
        for c in circuits:
            orig = next(r for r in rows if r["circuit"] == c and r["optimizer"] == "original")
            cells = [str(next(r for r in rows if r["circuit"] == c and r["optimizer"] == o)[metric])
                     for o in opts]
            lines.append(f"| {c} | {orig['qubits']} | {orig[metric]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["### Runtime (ms, best of 3)", "", "| circuit | " + " | ".join(opts) + " |",
              "|---|" + "---:|" * len(opts)]
    for c in circuits:
        cells = [f"{next(r for r in rows if r['circuit'] == c and r['optimizer'] == o)['ms']:.1f}"
                 for o in opts]
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")


def plot(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = [("qopt-O1", "#2a78d6"), ("qopt-O2", "#eb6834"),
              ("qopt-O3", "#1baf7a"), ("qiskit-L3", "#eda100")]
    circuits = list(dict.fromkeys(r["circuit"] for r in rows))
    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    width = 0.8 / len(series)
    for i, (opt, color) in enumerate(series):
        xs = [k + (i - (len(series) - 1) / 2) * width for k in range(len(circuits))]
        ys = [_reduction(rows, c, opt) for c in circuits]
        ax.bar(xs, ys, width * 0.88, color=color, label=opt, zorder=3)
        if opt == "qopt-O3":
            for x, y in zip(xs, ys):
                ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=7.5, color=ink)
    ax.set_xticks(range(len(circuits)), circuits, color=muted, fontsize=9)
    ax.set_ylabel("Gate-count reduction (%)", color=muted)
    ax.set_ylim(0, 100)
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=muted, linewidth=0.8, zorder=4)
    ax.set_title("Gate-count reduction by optimizer (higher is better)", color=ink,
                 loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(OUT / "gate_reduction.png", facecolor=surface)


if __name__ == "__main__":
    main()
