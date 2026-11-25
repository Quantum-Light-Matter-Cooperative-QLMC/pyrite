"""Fused per-chunk characteristic kernel (#1455, option B) against the grouped chain (A).

Usage:
  characteristic_chunk_kernel_1455.py check OUT_JSON [--segments N] [--twin-rows N] [--repeats R]
  characteristic_chunk_kernel_1455.py launches NSYS_SQLITE

``check`` (CuPy float32 device) builds fixed-seed synthetic segment sets:
tungsten and silicon emitting from a 100 A W film on a 900 A Si substrate, each
with and without channeling flux weights, plus carbon in a single slab. For
each it scores ``_accumulate_line_yields`` with the grouped chain (A,
``PYRITE_MC_CHARACTERISTIC_KERNEL=grouped``) and with the fused kernel (B,
``=fused``) at a small multi-chunk size (997) and at the runner's automatic
device chunk. It records the largest relative line-yield difference B vs A
(gate: 1e-12), B's bitwise rerun determinism, B against the NumPy twin
(``host_chunk_line_yields``, same reduction tree; gate 1e-12) on the first
``--twin-rows`` rows, the chunk count, and warm synchronized wall times per
arm (median of ``--repeats``). Each arm's first timed call runs in an NVTX
range ``char1455.<case>.<arm>.<chunk>``.

``launches`` reads an ``nsys export --type sqlite`` of a ``check`` run and
prints kernels per NVTX range and per chunk as JSON.

Exit status 1 when a gate fails. Engineering evidence; no physics-validation
claim.
"""

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

RTOL = 1.0e-12
LAYERS = [(0.0, 100.0, [("W", 0.063)]), (100.0, 1000.0, [("Si", 0.05)])]
CASES = (
    ("W-layered", "W", LAYERS, False),
    ("W-layered-weighted", "W", LAYERS, True),
    ("Si-layered", "Si", LAYERS, False),
    ("Si-layered-weighted", "Si", LAYERS, True),
    ("C-slab", "C", [(0.0, 5000.0, [("C", 0.1135)])], False),
)


def _inputs(element, layers, weighted, n, seed=1455):
    from pyrite._backend import REAL, xp
    from pyrite.montecarlo.spectrum import characteristic
    from pyrite.montecarlo.spectrum.segment_escape import segment_escape_paths

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    depth = layers[-1][1]
    segments = {
        "r_mid": np.column_stack((np.zeros(n), np.zeros(n), rng.uniform(1.0, depth - 1.0, n))),
        "v_hat": v / np.linalg.norm(v, axis=1, keepdims=True),
        "L_ang": rng.uniform(0.5, 50.0, n),
        "E_keV": rng.uniform(1.0, 300.0, n),
        "elec_id": np.repeat(np.arange(n // 4 + 1), 4)[:n],
        "layer": np.zeros(n, dtype=int),
        "Ne": n // 4 + 1,
        "thickness_ang": depth,
    }
    n_hat = np.array([0.3, 0.0, -0.95])
    n_hat /= np.linalg.norm(n_hat)
    many = len(layers) > 1
    owner, fraction, start, end = segment_escape_paths(
        segments, xp.arange(n), n_hat, layers=layers if many else None, xp=xp
    )
    table = characteristic.load_characteristic_cross_sections(element)
    composition = [(el, d) for _a, _b, comp in layers for el, d in comp if el == element]
    edges, widths = characteristic._energy_bin_edges_and_widths(np.linspace(100.0, 1.2e4, 4000))
    prepared = characteristic._prepare_lines(
        table, composition, layers if many else None, edges, widths, table.recommended_cutoff_eV
    )
    seg_E = xp.asarray(segments["E_keV"], dtype=REAL)[owner]
    weight = None
    if weighted:
        shells = len(table.ionization_shell_labels)
        weight = xp.asarray(rng.uniform(0.2, 3.0, (int(seg_E.size), shells)), dtype=REAL)
    return {
        "table": table,
        "prepared": prepared,
        "seg_E": seg_E,
        "seg_L": xp.asarray(segments["L_ang"], dtype=REAL)[owner] * fraction,
        "path_start": start,
        "path_end": end,
        "density": composition[0][1],
        "weight": weight,
        "case": {"composition": composition, "abs_layers": layers},
    }


def _score(inputs, chunk, arm, rows=None):
    from pyrite.montecarlo.spectrum import _characteristic_chunk_kernel as fused
    from pyrite.montecarlo.spectrum import characteristic

    os.environ[fused.KERNEL_ENV] = "fused" if arm == "B" else "grouped"
    p = inputs["prepared"]
    cut = slice(None) if rows is None else slice(0, rows)

    def take(value):
        return None if value is None else value[cut]

    out = characteristic._accumulate_line_yields(
        inputs["table"],
        p.response,
        p.mu_by_line,
        take(inputs["seg_E"]),
        take(inputs["seg_L"]),
        take(inputs["path_start"]),
        take(inputs["path_end"]),
        chunk=chunk,
        number_density_ang3=inputs["density"],
        shell_tables=p.shell_tables,
        packed_shells=p.packed_shells,
        shell_weight=take(inputs["weight"]),
    )
    return characteristic._to_cpu(out)


def _twin(inputs, chunk, rows):
    from pyrite.montecarlo.spectrum import _characteristic_chunk_kernel as fused
    from pyrite.montecarlo.spectrum import characteristic

    p, cpu = inputs["prepared"], characteristic._to_cpu

    def shell_sigma(energy_keV):
        import cupy

        return cpu(
            characteristic._interpolate_shell_cross_sections(
                inputs["table"], cupy.asarray(energy_keV), p.shell_tables, p.packed_shells
            )
        )

    weight = inputs["weight"]
    return fused.host_chunk_line_yields(
        cpu(inputs["seg_E"][:rows]),
        cpu(inputs["seg_L"][:rows]),
        cpu(inputs["path_start"][:rows]),
        cpu(inputs["path_end"][:rows]),
        chunk=chunk,
        number_density_ang3=inputs["density"],
        shell_sigma=shell_sigma,
        response=cpu(p.response),
        mu_by_line=cpu(p.mu_by_line),
        shell_weight=None if weight is None else cpu(weight[:rows]),
    )


def _rel(a, b):
    scale = np.maximum(np.abs(b), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / scale)) if a.size else 0.0


def _timed(inputs, chunk, arm, repeats, label):
    import cupy

    _score(inputs, chunk, arm)  # warm-up: compilation and pool growth
    walls = []
    for repeat in range(repeats):
        cupy.cuda.Device().synchronize()
        if repeat == 0:
            cupy.cuda.nvtx.RangePush(label)
        t0 = time.perf_counter()
        _score(inputs, chunk, arm)
        cupy.cuda.Device().synchronize()
        walls.append(time.perf_counter() - t0)
        if repeat == 0:
            cupy.cuda.nvtx.RangePop()
    return float(np.median(walls))


def check(args):
    import pyrite.montecarlo.runner as runner
    from pyrite._backend import REAL, xp

    if getattr(xp, "__name__", "") != "cupy" or np.dtype(REAL) != np.float32:
        sys.exit("check needs the CuPy float32 backend (PYRITE_MC_BACKEND=cuda or rocm)")
    records, failed = [], False
    for name, element, layers, weighted in CASES:
        inputs = _inputs(element, layers, weighted, args.segments)
        rows = int(inputs["seg_E"].size)
        automatic = runner._characteristic_chunk(inputs["case"], 4000)
        for chunk in sorted({997, automatic}):
            a, b = _score(inputs, chunk, "A"), _score(inputs, chunk, "B")
            twin_rows = min(rows, args.twin_rows)
            record = {
                "case": name,
                "chunk": chunk,
                "rows": rows,
                "chunks": -(-rows // chunk),
                "lines": int(a.size),
                "max_rel_B_vs_A": _rel(b, a),
                "B_rerun_bitwise": bool(np.array_equal(_score(inputs, chunk, "B"), b)),
                "max_rel_B_vs_twin": _rel(
                    _score(inputs, chunk, "B", rows=twin_rows), _twin(inputs, chunk, twin_rows)
                ),
                "wall_A_s": _timed(inputs, chunk, "A", args.repeats, f"char1455.{name}.A.{chunk}"),
                "wall_B_s": _timed(inputs, chunk, "B", args.repeats, f"char1455.{name}.B.{chunk}"),
            }
            record["pass"] = (
                record["max_rel_B_vs_A"] <= RTOL
                and record["max_rel_B_vs_twin"] <= RTOL
                and record["B_rerun_bitwise"]
            )
            failed |= not record["pass"]
            records.append(record)
            print(json.dumps(record), flush=True)
    with open(args.out, "w") as handle:
        json.dump({"rtol": RTOL, "segments": args.segments, "records": records}, handle, indent=1)
    sys.exit(1 if failed else 0)


_LAUNCH_QUERY = """
SELECT COALESCE(n.text, s.value) AS label, COUNT(k.correlationId)
FROM NVTX_EVENTS n
LEFT JOIN StringIds s ON n.textId = s.id
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.start >= n.start AND r.end <= n.end AND r.globalTid = n.globalTid
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
WHERE COALESCE(n.text, s.value) LIKE 'char1455.%'
GROUP BY label
"""


def launches(args):
    with sqlite3.connect(args.sqlite) as db:
        rows = db.execute(_LAUNCH_QUERY).fetchall()
    print(json.dumps({label: count for label, count in rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("check")
    run.add_argument("out")
    run.add_argument("--segments", type=int, default=2_000_000)
    run.add_argument("--twin-rows", type=int, default=50_000)
    run.add_argument("--repeats", type=int, default=3)
    run.set_defaults(func=check)
    count = commands.add_parser("launches")
    count.add_argument("sqlite")
    count.set_defaults(func=launches)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
