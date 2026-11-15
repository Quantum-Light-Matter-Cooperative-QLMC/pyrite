"""Phase profile of measured CXR reflection selection on frozen transports (#1200).

Replays a trusted saved transport (the #642 Si 300 keV slab or MoS2-on-Si
layered scene) through :func:`select_case_reflections`, the production
incremental selector, on a declared band axis or the case's automatic axis.
Every call runs under synchronized phase timers that split shell enumeration,
the per-call spectrum prologue (``_prepare_spectrum``), reflection tables,
segment-block kernels, reductions, the node far-tail and host transfer.

Arms differ only in reflections per line call (``--max-reflections``). Arms
run round-robin after one untimed warm-up selection, so no arm is the only
cold one. Each arm records its selected-set digest, shell table and spectrum
for parity against the first arm.

Remote only (frozen transports, multi-minute GPU selections). Diagnostic:
no accuracy verdict; spectra are compared, not qualified, here.
"""

import argparse
import cProfile
import hashlib
import json
import pickle
import pstats
import resource
import runpy
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np

GATE = Path(__file__).with_name("defaults_spectral_gate_642.py")
CRYSTAL = {"mos2-on-si": "mos2"}

#: (phase, module, attribute). Missing attributes are reported, not fatal.
PHASES = (
    ("enumeration", "pyrite.montecarlo.runner.reflection_cutoff", "_reflection_families"),
    ("line_call", "pyrite.montecarlo.runner", "_cxr_for_segments"),
    ("layer_split", "pyrite.montecarlo.runner.case_driver", "_segments_in_layer"),
    ("prologue", "pyrite.montecarlo.spectrum.cxr._spectrum", "_prepare_spectrum"),
    ("tables", "pyrite.montecarlo.spectrum.cxr._batched", "_batched_reflection_tables"),
    ("block_geometry", "pyrite.montecarlo.spectrum.cxr._batched", "_batched_block"),
    ("block_lines", "pyrite.montecarlo.spectrum.cxr._batched", "_batched_incoherent_block"),
    ("reduce_window", "pyrite.montecarlo.spectrum.cxr._batched", "accumulate_sincsq_windowed"),
    ("reduce_node", "pyrite.montecarlo.spectrum.cxr._batched", "accumulate_sincsq"),
    ("reduce_bin", "pyrite.montecarlo.spectrum.cxr._batched", "sincsq_bin_lineshape"),
    ("reduce_flush", "pyrite.montecarlo.spectrum.cxr._batched", "_CxrBatch.flush"),
    ("per_hkl", "pyrite.montecarlo.spectrum.cxr._spectrum", "_accumulate_per_hkl"),
    ("far_tail", "pyrite.montecarlo.spectrum.cxr._node_window", "NodeTail.apply"),
    ("finalize", "pyrite.montecarlo.spectrum.cxr._spectrum", "_finalize_spectrum"),
)


class PhaseTimer:
    """Inclusive and exclusive wall time per phase, synchronized at each boundary."""

    def __init__(self, sync):
        self.sync = sync
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self._stack = []

    def wrap(self, phase, fn):
        def timed(*args, **kwargs):
            self.sync()
            begin = time.perf_counter()
            self._stack.append(0.0)
            try:
                return fn(*args, **kwargs)
            finally:
                self.sync()
                elapsed = time.perf_counter() - begin
                children = self._stack.pop()
                if self._stack:
                    self._stack[-1] += elapsed
                self.inclusive[phase] += elapsed
                self.exclusive[phase] += elapsed - children
                self.calls[phase] += 1

        return timed

    def record(self):
        return {
            phase: dict(
                calls=self.calls[phase],
                inclusive_s=self.inclusive[phase],
                exclusive_s=self.exclusive[phase],
            )
            for phase in sorted(self.calls)
        }


@contextmanager
def instrumented(timer):
    """Patch every :data:`PHASES` target with ``timer``; restore on exit."""
    import importlib

    patched, missing = [], []
    for phase, module_name, attribute in PHASES:
        owner = importlib.import_module(module_name)
        *path, name = attribute.split(".")
        for part in path:
            owner = getattr(owner, part)
        original = getattr(owner, name, None)
        if original is None:
            missing.append(f"{module_name}.{attribute}")
            continue
        setattr(owner, name, timer.wrap(phase, original))
        patched.append((owner, name, original))
    try:
        yield missing
    finally:
        for owner, name, original in reversed(patched):
            setattr(owner, name, original)


def hkl_digest(case):
    radiators = case.get("layer_radiators") or [case]
    payload = [None if r is None else [list(h) for h in r["hkl_list"]] for r in radiators]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def automatic_axis(case, material, transport, gate):
    """The production automatic axis for this frozen transport (as in #1173 job 12679)."""
    from pyrite.campaign.sweep import build_cases
    from pyrite.montecarlo.runner.case_driver import case_spectrum_inputs

    energy, thickness = float(case["E0_keV"]), case.get("thickness_ang")
    sweep = gate["scene_sweep"](material, energy, None if thickness is None else float(thickness))
    built = build_cases(sweep, n_electrons=case["Ne"], n_electrons_brem=case["Ne"])[0].to_dict()
    case = dict(case)
    for key in ("E_grid", "E_grid_line", "line_grid_policy", "line_quadrature"):
        case.pop(key, None)
        if key in built:
            case[key] = built[key]
    case.pop("reflection_selection", None)
    if "reflection_selection" in built:
        case["reflection_selection"] = built["reflection_selection"]
    begin = time.perf_counter()
    inputs = case_spectrum_inputs(case, transport["segs"], case["Ne"], case["Ne"])
    grid = np.asarray(inputs["E_grid"], dtype=float)
    return case, grid, time.perf_counter() - begin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("material", choices=["silicon", "mos2-on-si"])
    parser.add_argument("replay", type=Path, help="trusted same-source transport pickle")
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", choices=["band", "automatic"], default="band")
    parser.add_argument("--band", type=float, nargs=2, default=(500.0, 10000.0))
    parser.add_argument("--spacing", type=float, default=0.09375, help="band node spacing (eV)")
    parser.add_argument("--quadrature", choices=["node", "bin-mean"], default="node")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--max-reflections",
        type=lambda text: [int(x) for x in text.split(",") if x],
        default=[512],
        help="comma-separated reflections per line call; first is the parity baseline",
    )
    parser.add_argument("--repeats", type=int, default=1, help="round-robin passes per arm")
    parser.add_argument(
        "--cprofile", action="store_true", help="also dump cProfile stats of the first pass"
    )
    args = parser.parse_args()

    from pyrite._backend import BACKEND
    from pyrite.energy_grid.convergence_case import CaseLadder
    from pyrite.montecarlo.runner.reflection_cutoff import (
        axis_yield_weights,
        select_case_reflections,
    )

    gate = runpy.run_path(str(GATE))
    sync, device, pool_bytes, _ = gate["runtime"]()
    release = BACKEND.release_memory if BACKEND.name != "cpu" else (lambda: None)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.replay.read_bytes()
    saved = pickle.loads(raw)  # noqa: S301 - explicit trusted study artifact
    case, transport = saved["case"], saved["transport"]
    if case["crystal"] != CRYSTAL.get(args.material, args.material):
        raise ValueError("replay material mismatch")
    report = dict(
        schema="pyrite.cxr-selection-profile.v1",
        material=args.material,
        source_revision=Path("SOURCE_SHA").read_text().strip()
        if Path("SOURCE_SHA").exists()
        else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replay_sha256=hashlib.sha256(raw).hexdigest(),
        seed=case["seed"],
        histories=case["Ne"],
        backend=BACKEND.name,
        device=device,
        axis=args.axis,
        tolerance=args.tolerance,
        max_reflections=args.max_reflections,
        repeats=args.repeats,
        rows=int(len(transport["segs"]["L_ang"])),
        arms=[],
    )
    del raw

    def save():
        report["host_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.axis == "automatic":
        case, grid, resolve_s = automatic_axis(case, args.material, transport, gate)
        quadrature = case.get("line_quadrature", "node")
        report["axis_resolution_s"] = resolve_s
    else:
        quadrature = args.quadrature
        start, stop = args.band
        grid = gate["axis"](start, stop, args.spacing, quadrature)
        case = dict(case, line_quadrature=quadrature)
    report.update(
        quadrature=quadrature,
        nodes=int(grid.size),
        support_eV=[float(grid[0]), float(grid[-1])],
    )
    save()
    scorer = CaseLadder(case, transport=transport)
    weights = axis_yield_weights(grid, quadrature)

    def select(max_reflections):
        def line_density(pilot, axis_grid):
            scorer.case = pilot
            return scorer.lines(axis_grid)

        return select_case_reflections(
            case,
            line_density,
            args.tolerance,
            grid,
            weights,
            max_reflections=max_reflections,
        )

    # Untimed warm-up: compiles kernels and fills module caches for every arm.
    sync()
    begin = time.perf_counter()
    warm = select(args.max_reflections[0])
    sync()
    report["warmup_s"] = time.perf_counter() - begin
    baseline = warm[2]
    report["warmup_hkl_digest"] = hkl_digest(warm[0])
    save()
    for repeat in range(args.repeats):
        for max_reflections in args.max_reflections:
            timer = PhaseTimer(sync)
            profiler = cProfile.Profile() if args.cprofile and repeat == 0 else None
            release()
            pool_before = pool_bytes()
            with ExitStack() as stack:
                missing = stack.enter_context(instrumented(timer))
                sync()
                begin = time.perf_counter()
                if profiler is not None:
                    profiler.enable()
                pruned, outcomes, density = select(max_reflections)
                if profiler is not None:
                    profiler.disable()
                sync()
                elapsed = time.perf_counter() - begin
            if profiler is not None:
                stats_path = args.output / f"cprofile-{max_reflections}.txt"
                with stats_path.open("w") as stream:
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
            density = np.asarray(density, dtype=float)
            if repeat == 0:
                np.savez_compressed(
                    args.output / f"spectrum-{max_reflections}.npz", grid_eV=grid, density=density
                )
            scale = float(np.abs(baseline).max()) or 1.0
            arm = dict(
                repeat=repeat,
                max_reflections=max_reflections,
                wall_s=elapsed,
                phases=timer.record(),
                missing_phases=missing,
                hkl_digest=hkl_digest(pruned),
                same_set_as_warmup=hkl_digest(pruned) == report["warmup_hkl_digest"],
                max_abs_diff_rel_peak=float(np.abs(density - baseline).max()) / scale,
                yield_rel_diff=float((density - baseline) @ weights) / float(baseline @ weights),
                device_pool_reserved_after_emptied_bytes=pool_bytes(),
                device_pool_reserved_before_bytes=pool_before,
                radiators=[
                    dict(
                        cutoff_invang=o.cutoff_invang,
                        reflections=len(o.hkl_list),
                        tail_fraction=o.tail_fraction if np.isfinite(o.tail_fraction) else None,
                        converged=o.converged,
                        shells=[
                            dict(upper=u, n=k, yield_=y, tail=t if np.isfinite(t) else None)
                            for u, k, y, t in o.shells
                        ],
                    )
                    for o in outcomes
                ],
            )
            report["arms"].append(arm)
            save()
            print(
                json.dumps(
                    dict(
                        max_reflections=max_reflections,
                        repeat=repeat,
                        wall_s=round(elapsed, 2),
                        exclusive={k: round(v["exclusive_s"], 2) for k, v in arm["phases"].items()},
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
