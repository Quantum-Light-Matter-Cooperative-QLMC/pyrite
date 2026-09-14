"""Catalog-case adapter and resumable driver for the line-grid refinement ladder.

:mod:`pyrite.energy_grid.convergence` is grid- and producer-agnostic. This
module binds it to one production case: transport runs exactly once through the
runner's own transport phase, and every rung re-evaluates only the production
spectrum reductions (lines, characteristic, bremsstrahlung under the lines) on
those segments. ``CaseLadder.evaluate_components`` at the case's own grid is
pinned bit-for-bit against the runner's spectrum phase by a regression test, so
the ladder measures the production spectrum rather than a re-implementation.

Gated observables use the CXR line density alone. Characteristic lines are
deposited as exact Lorentzian bin masses, a different, grid-exact
discretization whose sampled FWHM is the bin width by construction; it is
recorded per rung as an ungated informational yield.

Heavy ladders are remote work, submitted and pulled through the light
:mod:`pyrite.energy_grid.convergence_job`. ``run`` below is the slice body that
job executes. It checkpoints after every
rung; a resumed configuration re-runs its fixed-seed transport and refuses to
continue unless the segment fingerprint is identical to the stored one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .._backend import BACKEND, _to_cpu
from .._grid_semantics import resolution_num
from .._line_grid_policy import DEFAULT_RTOL
from ..campaign.config import material_sweep
from ..campaign.sweep import build_cases
from ..materials import CATALOG
from ..montecarlo import runner
from ..montecarlo.spectrum.diagnostics import sinc_feature_spacing
from .convergence import (
    Rung,
    SpectrumSample,
    evaluate_ladder,
    nested_uniform_ladder,
    require_identical_segments,
    richardson_acceptance,
    segment_fingerprint,
)
from .convergence_job import DEFAULT_NE, DEFAULT_THICKNESS_ANG, add_ladder_arguments

CHECKPOINT_SCHEMA = 1


def build_ladder_case(
    material: str,
    energy_keV: float,
    tilt_deg: float,
    tilt_azim_deg: float,
    *,
    thickness_ang: float = DEFAULT_THICKNESS_ANG,
    n_electrons: int = DEFAULT_NE,
    seed: int | None = None,
) -> dict[str, Any]:
    """One production case on the catalog line grid for a fixed geometry."""
    sweep = material_sweep(
        material,
        thickness_ang=thickness_ang,
        energy_keV=energy_keV,
        tilt_deg=tilt_deg,
        tilt_azim_deg=tilt_azim_deg,
    )
    case = dict(build_cases(sweep, n_electrons=n_electrons, n_electrons_brem=n_electrons)[0])
    if seed is not None:
        case["seed"] = int(seed)
    return case


class CaseLadder:
    """One transport of ``case``; any number of spectrum-only evaluations."""

    def __init__(
        self,
        case: Mapping[str, Any],
        *,
        transport_core: str = "auto",
        transport: Mapping[str, Any] | None = None,
    ):
        """Transport ``case`` once, or adopt an already-computed ``transport``.

        An injected transport (for example one pickled by another process with a
        different backend precision) is used as-is; no Monte Carlo runs here.
        """
        self.case = case
        started = time.perf_counter()
        if transport is None:
            self.transport = runner._transport_case(case, transport_core=transport_core)
            self.transport_wall_s = time.perf_counter() - started
        else:
            self.transport = dict(transport)
            self.transport_wall_s = 0.0
        self.segments = self.transport["segs"]
        self.fingerprint = segment_fingerprint(self.segments)
        self._segments_device = runner._segments_on_device(self.segments)
        self._brem_wide = None

    @property
    def bandwidth_eV(self) -> tuple[float, float]:
        """``(start, stop)`` of the case's resolved line grid."""
        grid = np.asarray(self.transport["E_grid"], dtype=float)
        return float(grid[0]), float(grid[-1])

    def _brem_wide_density(self):
        if self._brem_wide is None:
            tp = self.transport
            self._brem_wide = np.asarray(
                _to_cpu(
                    runner._brem_wide_from_segments(
                        self._segments_device,
                        tp["E_brem"],
                        self.case,
                        tp["n_hat"],
                        self.case.get("abs_layers"),
                        groove=tp.get("groove"),
                        Ne=tp["Ne_brem"],
                    )
                ),
                dtype=float,
            )
        return self._brem_wide

    def lines(self, E_grid_eV: object) -> np.ndarray:
        """Production incoherent CXR line density alone, on host."""
        tp = self.transport
        lines = runner._lines_for_segments(
            self._segments_device,
            np.asarray(E_grid_eV, dtype=float),
            self.case,
            tp["n_hat"],
            self.case.get("abs_layers"),
            tp.get("groove"),
            coherent=False,
            Ne=tp["Ne_lines"],
            table_cache={},
        )
        return np.asarray(_to_cpu(lines), dtype=float)

    def evaluate_components(self, E_grid_eV: object) -> dict[str, np.ndarray]:
        """Production line, characteristic, and under-line brem densities."""
        tp = self.transport
        E = np.asarray(E_grid_eV, dtype=float)
        lines = self.lines(E)
        characteristic = runner._characteristic_from_segments(
            self._segments_device,
            E,
            self.case,
            tp["n_hat"],
            self.case.get("abs_layers"),
            groove=tp.get("groove"),
            Ne=tp["Ne_brem"],
        )
        brem = np.interp(E, tp["E_brem"], self._brem_wide_density())
        return {
            "lines": lines,
            "characteristic": np.asarray(_to_cpu(characteristic), dtype=float),
            "brem": brem,
        }

    def evaluate(self, E_grid_eV: object) -> SpectrumSample:
        """Harness evaluator: CXR lines gated; characteristic yield informational."""
        E = np.asarray(E_grid_eV, dtype=float)
        parts = self.evaluate_components(E)
        return SpectrumSample(
            line=parts["lines"],
            background=parts["brem"],
            informational={
                "characteristic_yield": float(np.trapezoid(parts["characteristic"], E)),
            },
        )

    def sinc_estimate(self, aliased_weight_limit: float) -> dict[str, float]:
        """The #109 estimator on these exact segments, for comparison with the ladder."""
        step, aliased, count = sinc_feature_spacing(
            self.segments,
            self.transport["n_hat"],
            electron_limit=self.transport["Ne_lines"],
            aliased_weight_limit=aliased_weight_limit,
        )
        return {
            "aliased_weight_limit": float(aliased_weight_limit),
            "target_spacing_eV": float(step),
            "aliased_weight_fraction": float(aliased),
            "n_spacing_segments": int(count),
        }


def ladder_grids(start_eV: float, stop_eV: float, spacings_eV: Sequence[float]) -> list[np.ndarray]:
    """Nested exact-halving grids when the spacings halve; independent otherwise."""
    spacings = [float(value) for value in spacings_eV]
    if any(b >= a for a, b in zip(spacings, spacings[1:], strict=False)):
        raise ValueError("ladder spacings must be strictly decreasing")
    if all(
        math.isclose(a, 2.0 * b, rel_tol=1e-12)
        for a, b in zip(spacings, spacings[1:], strict=False)
    ):
        return nested_uniform_ladder(start_eV, stop_eV, spacings[0], len(spacings))
    return [
        np.linspace(start_eV, stop_eV, resolution_num(start_eV, stop_eV, value))
        for value in spacings
    ]


def _device_peak_mib() -> float | None:
    peak = BACKEND.allocator_stats().get("peak_mib")
    return None if peak is None else float(peak)


def config_key(config: Mapping[str, Any]) -> str:
    return (
        f"{config['material']}|{config['energy_keV']:g}keV|tilt{config['tilt_deg']:g}"
        f"|azim{config['tilt_azim_deg']:g}|t{config['thickness_ang']:g}"
        f"|ne{config['n_electrons']}|seed{config['seed']}"
    )


def _atomic_write_json(path: str | os.PathLike[str], payload: object) -> None:
    target = Path(path)
    directory = target.parent if str(target.parent) else Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def run_config(
    config: Mapping[str, Any],
    spacings_eV: Sequence[float],
    *,
    state: dict[str, Any] | None = None,
    save=None,
    out_of_time=None,
    transport_core: str = "auto",
) -> tuple[dict[str, Any], bool]:
    """Run or resume one ladder configuration; return ``(state, complete)``.

    ``state`` is the configuration's checkpoint entry. Rungs already present are
    kept; transport re-runs at the stored seed and must reproduce the stored
    segment fingerprint before any further rung is evaluated.
    """
    state = dict(state or {})
    spacings = [float(value) for value in spacings_eV]
    if state.get("spacings_eV") not in (None, spacings):
        raise ValueError(
            f"checkpoint for {config_key(config)} used spacings {state['spacings_eV']}, "
            f"not {spacings}; use a new --json-out"
        )
    if state.get("report") is not None:
        return state, True
    case = build_ladder_case(
        config["material"],
        config["energy_keV"],
        config["tilt_deg"],
        config["tilt_azim_deg"],
        thickness_ang=config["thickness_ang"],
        n_electrons=config["n_electrons"],
        seed=config["seed"],
    )
    ladder = CaseLadder(case, transport_core=transport_core)
    if state.get("fingerprint") is not None:
        require_identical_segments(state["fingerprint"], ladder.fingerprint)
    start, stop = ladder.bandwidth_eV
    state.update(
        config=dict(config),
        spacings_eV=spacings,
        fingerprint=ladder.fingerprint,
        bandwidth_eV=[start, stop],
    )
    state.setdefault("transport_wall_s", []).append(ladder.transport_wall_s)
    state.setdefault(
        "sinc_estimate",
        {name: ladder.sinc_estimate(limit) for name, limit in DEFAULT_RTOL.items()},
    )
    rungs = [Rung(**value) for value in state.get("rungs", [])]
    grids = ladder_grids(start, stop, spacings)
    if save is not None:
        save(state)
    for grid in grids[len(rungs) :]:
        if out_of_time is not None and out_of_time():
            return state, False
        rungs.extend(
            evaluate_ladder(
                [grid],
                ladder.evaluate,
                segments=ladder.segments,
                device_peak_mib=_device_peak_mib,
            )
        )
        state["rungs"] = [asdict(rung) for rung in rungs]
        if save is not None:
            save(state)
    state["report"] = richardson_acceptance(rungs).to_dict()
    state["report"].pop("rungs", None)
    if save is not None:
        save(state)
    return state, True


def _floats(text: str) -> list[float]:
    return [float(value) for value in text.split(",") if value.strip()]


def _configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for material in [value for value in args.materials.split(",") if value]:
        azimuth = (
            float(args.azimuth)
            if args.azimuth is not None
            else float(CATALOG.material(material).scan.tilt_azim_deg[0])
        )
        for energy in _floats(args.energies):
            for tilt in _floats(args.tilts):
                configs.append(
                    {
                        "material": material,
                        "energy_keV": energy,
                        "tilt_deg": tilt,
                        "tilt_azim_deg": azimuth,
                        "thickness_ang": float(args.thickness),
                        "n_electrons": int(args.ne),
                        "seed": int(args.seed),
                    }
                )
    return configs


def _summary_line(state: Mapping[str, Any]) -> str:
    config = state["config"]
    report = state.get("report") or {}
    accepted = report.get("accepted_spacing_eV")
    estimate = state["sinc_estimate"]["intrinsic_source"]["target_spacing_eV"]
    rungs = state.get("rungs", [])
    return (
        f"{config['material']:>6} {config['energy_keV']:>5g} keV tilt {config['tilt_deg']:>4g}: "
        f"accepted h={'-' if accepted is None else f'{accepted:.4g}'} eV, "
        f"sinc estimate {estimate:.4g} eV, rungs {len(rungs)}, "
        f"n_segments {state['fingerprint']['n_segments']}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    spacings = _floats(args.spacings)
    path = Path(args.json_out)
    payload: dict[str, Any] = {"schema": CHECKPOINT_SCHEMA, "configs": {}}
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise SystemExit(f"{path} is not a schema-{CHECKPOINT_SCHEMA} ladder checkpoint")
    started = time.monotonic()
    budget = None if args.max_minutes is None else float(args.max_minutes) * 60.0

    def out_of_time() -> bool:
        return budget is not None and time.monotonic() - started >= budget

    complete = True
    for config in _configs(args):
        key = config_key(config)

        def save(state, key=key):
            payload["configs"][key] = state
            _atomic_write_json(path, payload)

        if out_of_time():
            complete = False
            break
        state, done = run_config(
            config,
            spacings,
            state=payload["configs"].get(key),
            save=save,
            out_of_time=out_of_time,
        )
        print(_summary_line(state), flush=True)
        if not done:
            complete = False
            break
    if not complete:
        print("[line-grid-convergence] slice budget exhausted; work remains", flush=True)
        return 75
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run or resume ladders in this process")
    add_ladder_arguments(run)
    run.add_argument("--max-minutes", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
