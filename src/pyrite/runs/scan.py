"""Headless CXR scan runner (the library twin of ``src/pyrite/apps/scan_app.py``).

Runs the Monte-Carlo CXR parameter sweep for one material and writes the
per-material component checkpoint (``checkpoints/<material>/``). Use this to run sweeps
non-interactively -- in particular over SSH on the GPU box; see pyrite.remote,
which drives this and pulls the checkpoint back so interactive analysis and
static-HTML export can stay on the laptop.

    pyrite run standard               # standard profile membership
    pyrite run sub_100keV             # named profile membership
    pyrite run standard -m mose2      # one standard-profile member
    pyrite run standard -m mose2 --quick
    pyrite run standard -m mose2 --workers 0

(equivalently ``python -m pyrite._entry.scan standard -m mose2`` via the
module shim).

The ``if __name__ == "__main__"`` guard on the entry point is REQUIRED, not
stylistic: run_cases farms the electron transport out to a process pool, and the
default start method is 'spawn' on Windows and -- as of Python 3.14 --
'forkserver' on Linux. BOTH re-import the entry module in every worker, so without
the guard the sweep relaunches itself recursively (a process-spawn cascade that
surfaces as a forkserver ConnectionResetError). Inside the guard the re-import is
a harmless no-op.
"""

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from .._catalog_keys import material_keys
from .._env import GENERATED_INVOCATION_ENV, set_canonical_env
from .._progress import _electron_progress_fields, _ProgressTimer, _write_progress_record
from ..console import dashboard as _dashboard
from ..console import json as cli_json
from ..console import output as _cli_core
from ..console.outputs import output_dir

# Lazy runtime bindings keep help fast while preserving monkeypatchable module
# seams used by focused driver tests.
default_settings = None
format_penetration_watchdog_summary = None
gate_cases_by_penetration = None
material_sweep = None
run_sweep = None
build_cases = None
case_cost = None


def _load_runtime() -> None:
    global default_settings
    global format_penetration_watchdog_summary
    global gate_cases_by_penetration
    global material_sweep
    global run_sweep
    global build_cases
    global case_cost

    from .. import api as api_module
    from ..campaign import config as config_module
    from ..campaign import sweep as sweep_module
    from . import run as run_module

    default_settings = default_settings or config_module.default_settings
    format_penetration_watchdog_summary = (
        format_penetration_watchdog_summary or config_module.format_penetration_watchdog_summary
    )
    gate_cases_by_penetration = gate_cases_by_penetration or config_module.gate_cases_by_penetration
    material_sweep = material_sweep or config_module.material_sweep
    run_sweep = run_sweep or run_module.run_sweep
    build_cases = build_cases or api_module.build_configured_cases
    case_cost = case_cost or sweep_module.case_cost


def _local_probe():
    fields = []
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                parts = out.split("\n")[0].split(",")
                if len(parts) == 3:
                    gpu = float(parts[0])
                    vram_used = float(parts[1])
                    vram_total = float(parts[2])
                    vram_pct = 100.0 * vram_used / vram_total if vram_total > 0 else 0.0
                    fields.extend(
                        [
                            f"gpu_percent={gpu:.1f}",
                            f"vram_used_mib={vram_used:.0f}",
                            f"vram_total_mib={vram_total:.0f}",
                            f"vram_percent={vram_pct:.1f}",
                        ]
                    )
        except Exception:
            pass
    return "|".join(fields)


def _build_sections(args, materials, job_records, detail):
    meta = [
        f"kind: {'quick' if getattr(args, 'quick', False) else 'material-sweep'}",
        f"materials: {' '.join(materials)}",
        "job: local",
    ]
    if getattr(args, "workers", None) is not None:
        meta.append(f"workers: {args.workers}")
    if getattr(args, "max_minutes", None) is not None:
        meta.append(f"slice_minutes: {args.max_minutes}")
    if getattr(args, "catalog_profile", None) is not None:
        meta.append(f"profile: {args.catalog_profile}")

    slurm_job = os.environ.get("SLURM_JOB_ID")
    if slurm_job:
        meta.append(f"slurm_job_id: {slurm_job}")

    sections = {
        "META": "\n".join(meta),
        "STATE": "running",
        "RESOURCES": _local_probe(),
    }

    if slurm_job and shutil.which("squeue"):
        try:
            sq = subprocess.check_output(
                ["squeue", "-j", slurm_job, "-o", "%i|%T|%P|%M|%L|%D|%r", "--noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if sq:
                parts = sq.split("|")
                if len(parts) >= 7:
                    sections["SQUEUE"] = (
                        f"job_id={parts[0]}|state={parts[1]}|partition={parts[2]}|elapsed={parts[3]}|left={parts[4]}|nodes={parts[5]}|reason={parts[6]}"
                    )
        except Exception:
            pass

    progress_lines = []
    for _mat, rec in job_records.items():
        progress_lines.append(json.dumps(rec))

    if progress_lines:
        sections["PROGRESS"] = "\n".join(progress_lines)

    return sections


_dashboard_stop = threading.Event()


def _dashboard_loop(args, materials, job_records, detail):
    keys = _dashboard.KeyListener()
    try:
        while not _dashboard_stop.is_set():
            for key in keys.poll():
                if key.lower() == "v":
                    detail = (detail + 1) % 3
            sections = _build_sections(args, materials, job_records, detail)
            frame = _dashboard.style_states(_dashboard.format_job_status(sections, detail))
            _dashboard.render_frame(frame, tty=True)
            _dashboard_stop.wait(1.0)
    finally:
        keys.stop()


def validate_materials(materials: list[str]) -> None:
    """Reject runnable selections that are absent from the material catalog."""
    valid_materials = material_keys()
    unknown = [material for material in materials if material not in valid_materials]
    if unknown:
        raise SystemExit(f"unknown material(s): {', '.join(unknown)}")


def _profiled_child_command(
    *,
    catalog_profile,
    material,
    performance_profile,
    performance_dir,
    performance_interval,
    workers,
    quick,
    n_families,
    max_minutes=None,
    no_cache=False,
    recompute=False,
    checkpoint_subdir="nsys-checkpoints",
):
    """The ``python -m pyrite._dev perf`` child a local profiler wraps, and its
    output stem.

    The child re-runs this same command WITHOUT the profiler flag (so it cannot
    recurse), against an isolated, always-uncached checkpoint dir so the capture
    covers real work rather than a fast checkpoint resume. ``material`` is
    optional: when omitted the capture covers the profile's full membership and
    the output/checkpoint stem falls back to the profile name."""
    perf_root = Path(performance_dir) if performance_dir is not None else output_dir("performance")
    stem = material if material is not None else performance_profile
    trace_base = perf_root / performance_profile / stem
    checkpoint_root = (
        perf_root if performance_dir is not None else output_dir("checkpoints") / "performance"
    )
    checkpoint_dir = checkpoint_root / performance_profile / checkpoint_subdir / stem
    child = [
        sys.executable,
        "-m",
        "pyrite._dev",
        "perf",
        catalog_profile,
    ]
    if material is not None:
        child += ["-m", material]
    if performance_dir is not None:
        child += ["--performance-dir", str(performance_dir)]
    if performance_interval != 5.0:
        child += ["--perf-interval", f"{performance_interval:g}"]
    if workers is not None:
        child += ["--workers", str(workers)]
    if quick:
        child += ["--quick"]
    if n_families is not None:
        child += ["--n-families", str(n_families)]
    if max_minutes is not None:
        child += ["--max-minutes", f"{max_minutes:g}"]
    if no_cache:
        child += ["--no-cache"]
    elif recompute:
        child += ["--recompute"]
    child += ["--checkpoint-dir", str(checkpoint_dir)]
    return child, trace_base


def _nsys_reexec_command(
    *,
    catalog_profile,
    material,
    performance_profile,
    performance_dir,
    performance_interval,
    workers,
    quick,
    n_families,
    max_minutes=None,
    no_cache=False,
    recompute=False,
):
    """Build the ``nsys profile ... python -m pyrite._dev perf`` argv and the
    trace-output stem for a local ``--nsys`` capture.

    Pure (no side effects) so the argv/trace contract is unit-testable. The
    child is :func:`_profiled_child_command` under PYRITE_MC_NSYS=1, mirroring
    the remote job script's nsys launcher (:mod:`pyrite.remote.scripts`)."""
    child, trace_base = _profiled_child_command(
        catalog_profile=catalog_profile,
        material=material,
        performance_profile=performance_profile,
        performance_dir=performance_dir,
        performance_interval=performance_interval,
        workers=workers,
        quick=quick,
        n_families=n_families,
        max_minutes=max_minutes,
        no_cache=no_cache,
        recompute=recompute,
    )
    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--wait=all",
        "--force-overwrite=true",
        f"--output={trace_base}",
    ]
    return [*nsys_cmd, *child], trace_base


def _reexec_under_nsys(**kwargs):
    """Replace this process with the :func:`_nsys_reexec_command` launcher so
    Nsight Systems captures one uncached CUDA/NVTX trace of the run. Sets
    PYRITE_MC_NSYS=1 so the runner emits NVTX ranges in the child. Does not
    return on success (``os.execvp``)."""
    if shutil.which("nsys") is None:
        raise click.UsageError("--nsys requested but the nsys executable is not on PATH")
    argv, trace_base = _nsys_reexec_command(**kwargs)
    trace_base.parent.mkdir(parents=True, exist_ok=True)
    set_canonical_env("PYRITE_MC_NSYS", "1")
    # The parent already warned about any deprecated option it forwards.
    set_canonical_env(GENERATED_INVOCATION_ENV, "1")
    os.execvp(argv[0], argv)


def _py_spy_reexec_command(launcher, **kwargs):
    """Build the ``py-spy record ... -- python -m pyrite._dev perf`` argv, the
    profile output path, and the exit-status file for a local ``--py-spy``
    capture.

    Pure (no side effects) so the argv contract is unit-testable; ``launcher``
    is the py-spy executable argv (:func:`pyrite.perf.py_spy.py_spy_launcher`).
    The child runs through :func:`pyrite.perf.py_spy.main`, which records the
    exit status ``py-spy record --subprocesses`` would otherwise discard."""
    from ..perf.py_spy import (
        PY_SPY_STATUS_SUFFIX,
        PY_SPY_SUFFIX,
        py_spy_record_args,
        status_wrapped,
    )

    child, trace_base = _profiled_child_command(**kwargs, checkpoint_subdir="py-spy-checkpoints")
    output = Path(f"{trace_base}{PY_SPY_SUFFIX}")
    status = Path(f"{trace_base}{PY_SPY_STATUS_SUFFIX}")
    argv = [*launcher, *py_spy_record_args(str(output)), *status_wrapped(str(status), child)]
    return argv, output, status


def _reexec_under_py_spy(**kwargs):
    """Run py-spy sampling a :func:`_profiled_child_command` run, writing a
    speedscope profile next to the perf log, and exit with the child's status."""
    from ..perf.py_spy import py_spy_launcher, read_status

    launcher = py_spy_launcher()
    if launcher is None:
        raise click.UsageError("--py-spy requested but neither py-spy nor uv is on PATH")
    argv, output, status = _py_spy_reexec_command(launcher, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    status.unlink(missing_ok=True)
    # The parent already warned about any deprecated option it forwards.
    set_canonical_env(GENERATED_INVOCATION_ENV, "1")
    subprocess.run(argv, check=False)  # noqa: S603 -- argv built above
    raise SystemExit(read_status(status))


def _resolve_catalog_profile(catalog_profile: str, performance_profile: str | None) -> str:
    """Reconcile ``--profile``/``--performance-profile`` into one catalog
    profile name -- the same rule the command's early validation, material
    selection (``_selected``), and per-material settings resolution
    (``_resolved_run``) all need to agree on."""
    if performance_profile is not None:
        if catalog_profile not in ("standard", performance_profile):
            raise click.UsageError(
                "--performance-profile and --profile must name the same catalog profile"
            )
        catalog_profile = performance_profile
    return catalog_profile


def _effective_catalog_profile(args) -> str:
    """``args``-based wrapper around :func:`_resolve_catalog_profile`."""
    return _resolve_catalog_profile(
        getattr(args, "catalog_profile", "standard"),
        getattr(args, "performance_profile", None),
    )


def validate_catalog_profile(
    catalog_profile: str, materials: list[str], *, intersect: bool
) -> list[str]:
    """Validate ``--profile`` (catalog campaign name) against the material
    catalog and reconcile it with an already-selected material list.

    Unknown profile names are always a usage error listing what's available.
    With ``intersect=True`` (``--all``/``-A``), a profile's explicit
    ``materials`` membership list silently narrows the selection -- absent
    membership means every candidate stays. With ``intersect=False`` (an
    explicit MATERIAL/material list), any member outside the profile is a
    hard usage error instead: the user named it, so silently dropping it
    would be a surprise, not a convenience.
    """
    from ..materials import CATALOG

    if catalog_profile not in CATALOG.profile_names:
        available = ", ".join(sorted(CATALOG.profile_names)) or "(none defined)"
        raise click.UsageError(f"unknown profile {catalog_profile!r}; available: {available}")
    membership = CATALOG.profile_materials(catalog_profile)
    if membership is None:
        return materials
    allowed = set(membership)
    if intersect:
        narrowed = [m for m in materials if m in allowed]
        if not narrowed:
            raise click.UsageError(
                f"profile {catalog_profile!r} shares no materials with this selection "
                f"(members: {', '.join(membership)})"
            )
        return narrowed
    non_members = [m for m in materials if m not in allowed]
    if non_members:
        raise click.UsageError(
            f"profile {catalog_profile!r} does not include {', '.join(non_members)} "
            f"(members: {', '.join(membership)})"
        )
    return materials


def resolve_profile_materials(catalog_profile: str, material: str | None = None) -> list[str]:
    """Resolve one run selection from a profile and optional material override.

    Shared boundary contract for local and remote ``pyrite run``: ``-m``
    selects one profile member; omitting it selects the profile's explicit
    ``materials`` membership. A custom profile without an explicit membership
    selects the full catalog; every shipped profile is explicit.
    """
    from ..materials import CATALOG

    validate_catalog_profile(catalog_profile, [], intersect=False)
    if material is not None:
        validate_materials([material])
        materials = validate_catalog_profile(catalog_profile, [material], intersect=False)
    else:
        membership = CATALOG.profile_materials(catalog_profile)
        materials = list(membership) if membership is not None else list(CATALOG.material_keys)
    validate_materials(materials)
    return materials


def _selected(args):
    """Resolve profile-owned membership plus an optional one-material override."""
    catalog_profile = _effective_catalog_profile(args)
    args.high_energy_floor_map = {}
    return resolve_profile_materials(catalog_profile, getattr(args, "material", None))


def run(args):
    materials = _selected(args)
    deadline = None
    if getattr(args, "max_minutes", None) is not None:
        deadline = time.monotonic() + args.max_minutes * 60.0

    job_records = {}
    use_dashboard = (
        not getattr(args, "no_progress", False)
        and sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
    )

    if use_dashboard:
        args._job_records = job_records
        set_canonical_env("PYRITE_LOCAL_DASHBOARD", "1")
        detail = getattr(args, "verbose", 0)
        t = threading.Thread(
            target=_dashboard_loop, args=(args, materials, job_records, detail), daemon=True
        )
        t.start()

    try:
        incomplete = False
        from ..materials import CATALOG, load_material_catalog

        profile = _effective_catalog_profile(args)
        catalog = CATALOG if profile == "standard" else load_material_catalog(profile=profile)
        detector_ids = tuple(catalog.profile_detector_set(profile))
        for material in materials:
            for detector_id in detector_ids:
                args.detector_id = detector_id
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    complete = _run_material(args, material, max_seconds=remaining)
                except Exception as error:
                    from ..montecarlo.trajectories import TrajectoryArtifactError

                    if not isinstance(error, TrajectoryArtifactError):
                        raise
                    raise SystemExit(str(error)) from None
                if not complete:
                    incomplete = True
        if incomplete:
            raise SystemExit(75)  # EX_TEMPFAIL: budget hit, work remains
    finally:
        if hasattr(args, "detector_id"):
            del args.detector_id
        if use_dashboard:
            _dashboard_stop.set()
            t.join(timeout=2.0)
            sections = _build_sections(args, materials, job_records, getattr(args, "verbose", 0))
            sections["STATE"] = "done" if not incomplete else "paused"
            frame = _dashboard.style_states(
                _dashboard.format_job_status(sections, getattr(args, "verbose", 0))
            )
            _dashboard.render_frame(frame, tty=True)
            if "PYRITE_LOCAL_DASHBOARD" in os.environ:
                del os.environ["PYRITE_LOCAL_DASHBOARD"]


def _run_json(args):
    """Run batch with machine-only output and retain partial material results."""
    materials = _selected(args)
    started = time.monotonic()
    deadline = (
        None if getattr(args, "max_minutes", None) is None else started + args.max_minutes * 60.0
    )
    completed = []
    failed = []
    errors = {}
    resumable = False
    args.no_progress = True
    from ..materials import CATALOG, load_material_catalog

    profile = _effective_catalog_profile(args)
    catalog = CATALOG if profile == "standard" else load_material_catalog(profile=profile)
    detector_ids = tuple(catalog.profile_detector_set(profile))
    checkpoints = []
    try:
        for material in materials:
            material_errors = []
            for detector_id in detector_ids:
                args.detector_id = detector_id
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        complete = _run_material(args, material, max_seconds=remaining)
                    checkpoints.append(
                        os.path.join(args.checkpoint_dir, _checkpoint_stem(args, material))
                    )
                except (Exception, SystemExit) as exc:
                    material_errors.append(f"{detector_id}: {str(exc) or type(exc).__name__}")
                    continue
                if not complete:
                    material_errors.append(f"{detector_id}: resumable work remains")
                    resumable = True
            if material_errors:
                failed.append(material)
                errors[material] = "; ".join(material_errors)
            else:
                completed.append(material)
    finally:
        if hasattr(args, "detector_id"):
            del args.detector_id
    result = cli_json.operation_summary(
        "run",
        materials,
        completed,
        failed_materials=failed,
        checkpoints=checkpoints,
        elapsed_seconds=time.monotonic() - started,
        resumable=resumable,
        material_errors=errors,
    )
    _cli_core.emit_json_result(
        result,
        failure_exit=75
        if resumable and all("resumable work remains" in message for message in errors.values())
        else 1,
    )


def _select_quick_energies(by_energy, e_grid_line, *, nominal=(30, 50), count=2):
    """Bounded, deterministic quick beam energies with valid line grids.

    ``--quick`` needs a small representative beam-energy subset whose line grids
    the effective profile/material actually configures, so ``build_cases`` never
    requests an absent per-energy grid. Policy:

    * A fixed ``E_grid_line`` or an absent ``E_grid_line_by_energy`` mapping means
      every beam energy resolves the same grid; keep the historical ``[30, 50]``
      (int literals -- preserves the standard profile's checkpoint identity
      bit-for-bit).
    * Otherwise the valid energies are the mapping's keys. Preserve the nominal
      ``30``/``50`` literals where they are valid, then fill any remaining slots
      with the valid energies nearest the unmet nominal targets (ties break low)
      to keep energy diversity. A profile exposing fewer than ``count`` valid
      energies runs exactly those; an empty mapping falls through to the nominal
      pair so the strict missing-grid error still fires downstream.
    """
    if by_energy is None or e_grid_line is not None:
        return list(nominal)
    valid = sorted(float(key) for key in by_energy)
    if not valid:
        return list(nominal)
    valid_set = set(valid)
    chosen = [energy for energy in nominal if float(energy) in valid_set]
    chosen_values = {float(energy) for energy in chosen}
    remaining = [energy for energy in valid if energy not in chosen_values]
    for target in (energy for energy in nominal if float(energy) not in valid_set):
        if len(chosen) >= count or not remaining:
            break
        nearest = min(remaining, key=lambda energy: (abs(energy - target), energy))
        chosen.append(nearest)
        remaining.remove(nearest)
    for energy in remaining:
        if len(chosen) >= count:
            break
        chosen.append(energy)
    return chosen[:count]


def _resolved_run(args, material):
    """Resolve settings, sweep, identity, and collision-free checkpoint stem."""
    import numpy as np

    _load_runtime()
    assert default_settings is not None
    assert material_sweep is not None

    fidelity = getattr(args, "fidelity", "full")
    catalog_profile = _effective_catalog_profile(args)
    settings = default_settings() if fidelity == "full" else default_settings(fidelity)
    # Emission (incoherent/coherent/both) is PROFILE-owned -- there is no CLI
    # override flag. A catalog_profile emission key (set via `pyrite profile
    # set/add/remove --emission/--coherent/--incoherent`) overrides the
    # fidelity preset's emission; absent means the fidelity's own emission
    # stands. settings.emission drives the dataset_identity divergence key and
    # the canonical_full collision guard below, so a coherent/both run never
    # shares the plain incoherent <material> stem.
    from ..materials import CATALOG

    detector_id = getattr(args, "detector_id", None)

    from ..campaign.profiles import profile_run_settings

    settings = profile_run_settings(settings, catalog_profile, fidelity, CATALOG)
    overrides = {}
    if getattr(args, "quick", False):
        # Resolve quick beam energies from the effective profile/material line
        # grids instead of forcing a fixed [30, 50]: a named profile may own an
        # E_grid_line_by_energy mapping without 50 keV, and build_cases rejects a
        # beam energy with no configured line grid before compute. The probe
        # sweep (same catalog_profile/fidelity, no beam/tilt overrides) exposes
        # the valid line-grid energies; _select_quick_energies keeps the standard
        # profile's [30, 50] bit-for-bit where both are valid.
        probe = material_sweep(
            material, fidelity=fidelity, catalog_profile=catalog_profile, detector_id=detector_id
        )
        overrides.update(
            # Start at 5 deg, not 0: tilt=0 is a banned emission geometry
            # (issue_notes.md #1), and build_cases rejects it.
            tilt_deg=np.linspace(5.0, 85.0, 5),
            tilt_azim_deg=np.array([10.0, 30.0]),
            energy_keV=_select_quick_energies(
                probe.detector.energy_bins.line_by_energy,
                probe.detector.energy_bins.line,
            ),
        )
    if getattr(args, "n_families", None) is not None:
        overrides["n_families"] = args.n_families
    if getattr(args, "beam_uvw", None) is not None:
        overrides["beam_uvw"] = tuple(args.beam_uvw)
    # Beam source and distribution settings are profile-owned.
    sweep = material_sweep(
        material,
        fidelity=fidelity,
        catalog_profile=catalog_profile,
        detector_id=detector_id,
        **overrides,
    )

    # Legacy remote records may carry a high-energy floor. Preserve that private
    # compatibility path while new runs express their energy range through the
    # catalog profile, where it participates directly in dataset identity.
    floor = getattr(args, "high_energy_floor_map", None) or {}
    floor = floor.get(material)
    if floor is not None and not getattr(args, "quick", False):
        from ..campaign.sweep import beam_replace

        energies = np.asarray(sweep.beam.energy_keV, dtype=float)
        kept = energies[energies >= floor]
        if kept.size == 0:
            raise SystemExit(
                f"{material}: no energies >= {floor} keV in its grid (high-energy "
                "floor); lower --high-energy-min-kev or drop this material"
            )
        # Also record in `overrides` (not just apply via `replace` below) so the
        # canonical_full check further down treats a floored grid as a variant,
        # not the plain canonical stem -- otherwise a floored tise2 checkpoint
        # would collide with an unfiltered tise2 full-grid checkpoint.
        overrides["energy_keV"] = kept
        sweep = replace(sweep, beam=beam_replace(sweep.beam, energy_keV=kept))

    from ..campaign.profiles import profile_precision_source

    # The electron-count policy also depends on the material's sweep (count
    # overrides, a GDF beam, a grooved face), so it settles once the sweep does.
    settings = replace(
        settings,
        precision=profile_precision_source(settings, catalog_profile, CATALOG, sweep=sweep)[0],
    )
    from ..campaign.geometry import Stack
    from ..campaign.profiles import (
        canonical_settings,
        dataset_identity,
        detector_variant,
        variant_stem,
    )
    from ..xsgen.sbethe import resolve_catalog_table
    from ..xsgen.store import identity_markers

    target = sweep.target
    assert target is not None
    table_keys = (
        [layer.material for layer in target.layers]
        if isinstance(target, Stack)
        else [target.material]
    )
    tables = [resolve_catalog_table(key) for key in table_keys]
    if settings.elastic_model == "elsepa":
        from ..xsgen.elsepa.catalog import resolve_catalog_tables

        tables.extend(table for key in table_keys for table in resolve_catalog_tables(key))
    from ..xsgen.bremslib.tables import load_bremsstrahlung_tables, resolve_bremsstrahlung_model
    from ..xsgen.elsepa.catalog import catalog_composition

    elements = [element for key in table_keys for element, _ in catalog_composition(key)]
    # Resolve "auto" once, here, so the settings a run executes with, its
    # dataset identity, and its table markers all name the same source.
    settings = replace(
        settings,
        bremsstrahlung_model=resolve_bremsstrahlung_model(settings.bremsstrahlung_model, elements),
    )
    xsgen_tables = identity_markers(tables)
    if settings.bremsstrahlung_model == "bremslib":
        xsgen_tables |= {
            table.key: table.digest for table in load_bremsstrahlung_tables(elements).values()
        }

    if detector_id is None:
        detector_id = next(iter(CATALOG.profile_detector_set(catalog_profile)))
    variant = detector_variant(
        catalog_profile, detector_id, quick=bool(getattr(args, "quick", False))
    )
    named_variant = variant is not None and variant != "quick"
    identity = dataset_identity(
        material,
        fidelity,
        settings,
        sweep,
        variant=variant,
        catalog_profile=catalog_profile,
        xsgen_tables=xsgen_tables,
    )
    if named_variant:
        identity["detector_id"] = detector_id
    canonical_full = (
        fidelity == "full"
        and not overrides
        and not getattr(args, "quick", False)
        and not named_variant
        and catalog_profile == "standard"
        and canonical_settings(settings)
    )
    stem = variant_stem(identity, canonical_full=canonical_full)
    if getattr(args, "quick", False) and named_variant:
        stem = f"{material}_quick_{detector_id}"
    return settings, sweep, identity, stem


def _sweep_observation(args, identity, settings, stem, content_key_fn):
    """The profile's counting observation for this sweep, or ``None``.

    Observations for ``<checkpoint_dir>/<stem>`` live in the sibling
    ``observations/<stem>`` store, so a custom checkpoint root keeps both
    together and neither participates in the other's lifecycle.
    """
    from ..campaign.observation import resolve_profile_observation
    from ..materials import CATALOG, load_material_catalog

    profile = str(identity.get("catalog_profile", "standard"))
    catalog = CATALOG if profile == "standard" else load_material_catalog(profile=profile)
    detector_id = getattr(args, "detector_id", None)
    observation = resolve_profile_observation(catalog, profile, detector_id=detector_id)
    if observation is None:
        return None
    from ..api import run_provenance
    from ..montecarlo.runner import cached_case_table_markers
    from ..observations import ObservationStore, SweepObservation

    table_markers = cached_case_table_markers()
    return SweepObservation(
        observation=observation,
        store=ObservationStore(stem, Path(args.checkpoint_dir).resolve().parent / "observations"),
        content_key_fn=content_key_fn,
        emission=settings.emission,
        provenance_fn=lambda case: {
            **run_provenance(case, table_markers(case)),
            **({"detector_id": detector_id} if detector_id is not None else {}),
        },
    )


def _checkpoint_stem(args, material):
    return _resolved_run(args, material)[3]


def _run_material(args, material, max_seconds=None):
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    _load_runtime()
    assert format_penetration_watchdog_summary is not None
    assert gate_cases_by_penetration is not None
    assert run_sweep is not None
    assert build_cases is not None

    settings, sweep, identity, stem = _resolved_run(args, material)
    fidelity = identity["fidelity"]

    cases = build_cases(sweep, settings)
    cases, dropped = gate_cases_by_penetration(cases)
    summary = format_penetration_watchdog_summary(dropped, material=material)
    if summary is not None:
        print(summary)
    print(
        f"{material} [{getattr(args, 'detector_id', 'default')}]: {len(cases)} cases across "
        f"{len({c['name'] for c in cases})} configs "
        f"[profile={fidelity}, parameters={identity['parameter_sha256'][:12]}]"
        + (" (quick grid)" if args.quick else "")
        + ("" if settings.emission == "incoherent" else f" ({settings.emission})")
    )
    # read the RESOLVED orientation off the first case, not the Sweep request:
    # HOPG/h-BN hand-pin hkl_list and bypass dominant_reflections entirely, so
    # echoing sweep.n_families would claim a knob that's actually inert for them.
    print(
        f"crystal orientation: beam_uvw={cases[0]['beam_uvw']} "
        f"hkl_list ({len(cases[0]['hkl_list'])} reflections)={cases[0]['hkl_list']}"
    )

    # Always pass an explicit path named for the catalog material key: run_sweep's
    # default derives the name from the film crystal, which would make a named
    # stack (e.g. mos2-on-sio2-si) clobber/resume the plain film's checkpoint.
    # A --quick smoke test writes to its OWN checkpoint (<material>_quick/), so
    # its coarse off-grid points never contaminate the real per-material sweep.
    ckpt = os.path.join(args.checkpoint_dir, stem)
    results = {}
    progress_file = getattr(args, "progress_file", None)
    progress_phase = getattr(args, "progress_phase", None)
    has_dashboard = hasattr(args, "_job_records")
    latest_progress = {
        "total_cases": len(cases),
        "cached_cases": 0,
        "completed_new_cases": 0,
    }
    if progress_file is not None and Path(progress_file).is_file():
        try:
            previous = json.loads(Path(progress_file).read_text(encoding="utf-8"))
            if (
                previous.get("material") == material
                and previous.get("total_cases") == len(cases)
                and all(
                    isinstance(previous.get(key), int)
                    for key in ("cached_cases", "completed_new_cases")
                )
                and 0 <= previous["cached_cases"] + previous["completed_new_cases"] <= len(cases)
            ):
                latest_progress.update(
                    cached_cases=previous["cached_cases"],
                    completed_new_cases=previous["completed_new_cases"],
                )
        except OSError, ValueError, TypeError:
            pass
    latest_case = {}
    last_completed_case = {}
    # Cost weighting uses sweep.case_cost and run_sweep's cached/done split to
    # track relative matmul work; scan_app.py uses the same meter locally.
    latest_cost = {}
    initial_done_cost = None
    progress_timer = _ProgressTimer(progress_file) if progress_file is not None else None
    runtime_info = {}
    timing_info: dict[str, Any] = {
        "checkpoint_count": 0,
        "checkpoint_seconds_total": 0.0,
        "gpu_oom_retry_count_total": 0,
        "gpu_feed_wait_fraction": None,
    }
    activity_info = {
        "phase": "setup",
        "active_case": None,
        "in_flight_case_count": 0,
    }
    performance_state = {"state": "running"}
    performance_lock = threading.Lock()

    def _case_summary(case):
        from .._backend import BackendResourceError
        from ..montecarlo import runner as montecarlo_runner

        # The runtime plan is informational here. A grid the device budget
        # cannot admit is the scheduler's to handle (CPU fallback or refusal);
        # progress reporting must not raise it first.
        try:
            plan = montecarlo_runner.case_runtime_plan(case)
        except BackendResourceError as error:
            plan = {"runtime_plan_error": str(error)}
        return {
            "configuration": str(case["name"]),
            "energy_keV": round(float(case["E0_keV"]), 3),
            "tilt_deg": round(float(case["tilt_deg"]), 2),
            "azimuth_deg": round(float(case["tilt_azim_deg"]), 2),
            "thickness_um": round(float(case["thickness_ang"]) / 1e4, 4),
            **plan,
        }

    def _note_case(case):
        # Completion callbacks are not active-case signals. Keep this separately
        # so the dashboard never presents a finished frontier as NOW TESTING.
        with performance_lock:
            last_completed_case.clear()
            last_completed_case.update(_case_summary(case))

    def _record_cost(done_cost, total_cost):
        nonlocal initial_done_cost
        with performance_lock:
            if initial_done_cost is None:
                initial_done_cost = done_cost
            latest_cost.update(done_cost=done_cost, total_cost=total_cost)

    def _record_progress(completed_new_cases, total_cases, cached_cases):
        with performance_lock:
            latest_progress.update(
                total_cases=total_cases,
                cached_cases=cached_cases,
                completed_new_cases=completed_new_cases,
            )
            progress_snapshot = dict(latest_progress)
            cost_snapshot = dict(latest_cost)
            case_snapshot = dict(latest_case) or None
            electron_snapshot = _electron_progress_fields(activity_info) if case_snapshot else {}

        computed_cost = (
            None
            if initial_done_cost is None or "done_cost" not in cost_snapshot
            else max(0.0, cost_snapshot["done_cost"] - initial_done_cost)
        )
        if hasattr(args, "_job_records"):
            rec = {
                "material": material,
                "state": "running",
                "total_cases": progress_snapshot["total_cases"],
                "cached_cases": progress_snapshot["cached_cases"],
                "completed_new_cases": progress_snapshot["completed_new_cases"],
            }
            if case_snapshot:
                rec["current"] = case_snapshot
            if activity_info.get("phase"):
                rec["activity"] = activity_info["phase"]
            if last_completed_case:
                rec["last_completed"] = dict(last_completed_case)
            if "done_cost" in cost_snapshot and "total_cost" in cost_snapshot:
                rec["done_cost"] = cost_snapshot["done_cost"]
                rec["total_cost"] = cost_snapshot["total_cost"]
            rec.update(electron_snapshot)
            if progress_timer:
                rec.update(
                    progress_timer.snapshot(
                        completed_new_cases=progress_snapshot["completed_new_cases"],
                        computed_cost=computed_cost,
                    )
                )
            args._job_records[material] = rec

        if progress_file is not None:
            assert progress_timer is not None
            _write_progress_record(
                progress_file,
                material=material,
                phase=progress_phase,
                state="running",
                current=case_snapshot,
                activity=activity_info.get("phase"),
                last_completed=dict(last_completed_case) or None,
                **progress_snapshot,
                **cost_snapshot,
                **electron_snapshot,
                **progress_timer.snapshot(
                    completed_new_cases=progress_snapshot["completed_new_cases"],
                    computed_cost=computed_cost,
                ),
            )

    def _record_runtime(info):
        with performance_lock:
            runtime_info.clear()
            runtime_info.update(info)

    def _record_timing(info):
        with performance_lock:
            checkpoint_seconds = info.get("checkpoint_seconds")
            if checkpoint_seconds is not None:
                timing_info["checkpoint_count"] += 1
                timing_info["checkpoint_seconds"] = checkpoint_seconds
                timing_info["checkpoint_seconds_total"] += checkpoint_seconds
            retries = info.get("gpu_oom_retry_count", 0)
            timing_info["gpu_oom_retry_count_total"] += retries
            for key, value in info.items():
                if key == "gpu_oom_retry_count":
                    continue
                if key in {
                    "transport_seconds",
                    "spectrum_seconds",
                    "driver_wait_seconds",
                }:
                    timing_info[f"last_{key}"] = value
                else:
                    timing_info[key] = value
            wait = timing_info.get("driver_wait_seconds_total", 0.0)
            spectrum = timing_info.get("spectrum_seconds_total", 0.0)
            denominator = wait + spectrum
            timing_info["gpu_feed_wait_fraction"] = wait / denominator if denominator > 0 else None

    def _record_activity(info):
        info = dict(info)
        active = info.pop("case", None)
        active_case = _case_summary(active) if active is not None else None
        raw_phase = info.get("phase")
        phase = raw_phase if raw_phase in {"loading", "saving", "handoff"} else "computing"
        with performance_lock:
            activity_info.clear()
            activity_info.update(info, phase=phase, active_case=active_case)
            latest_case.clear()
            if active_case is not None and phase == "computing":
                latest_case.update(active_case)
        _record_progress(
            latest_progress["completed_new_cases"],
            latest_progress["total_cases"],
            latest_progress["cached_cases"],
        )

    def _performance_context():
        with performance_lock:
            return {
                **runtime_info,
                **latest_progress,
                **latest_cost,
                **timing_info,
                **activity_info,
                "current": dict(latest_case) or None,
                **performance_state,
            }

    if progress_file is not None:
        assert progress_timer is not None
        _write_progress_record(
            progress_file,
            material=material,
            phase=progress_phase,
            state="running",
            **latest_progress,
            **progress_timer.snapshot(),
        )
    performance_logger = None
    performance_profile = getattr(args, "performance_profile", None)
    if performance_profile is not None:
        from ..campaign.profiles import _jsonable
        from ..perf.performance_profile import PerformanceLogger

        performance_dir = getattr(args, "performance_dir", None)
        performance_root = Path(performance_dir) if performance_dir else output_dir("performance")
        profile_dir = performance_root / performance_profile
        performance_logger = PerformanceLogger(
            profile_dir / f"{material}.ndjson",
            profile=performance_profile,
            material=material,
            static={
                "fidelity": fidelity,
                "catalog_profile": identity.get("catalog_profile", "standard"),
                "parameter_sha256": identity["parameter_sha256"],
                "checkpoint_stem": stem,
                "total_case_count": len(cases),
                # Beam parameters now resolve through Sweep.beam, including
                # profile TOML values plus CLI overrides. Log that resolved
                # dataclass instead of reconstructing legacy top-level fields.
                "beam_parameters": _jsonable(sweep.beam),
            },
            context=_performance_context,
            latest_path=profile_dir / f"{material}.latest.json",
            interval_seconds=getattr(args, "performance_interval", 5.0),
        )
        performance_logger.start()
    # Shared per-case cache gates. Default reads + writes; --recompute / --no-cache
    # / a -p perf run narrow this (see the `run` command). cache_read also drives
    # the per-stem resume, so a --no-cache/--recompute/perf run recomputes rather
    # than resume-skipping (the perf-run "measures near-nothing" bug).
    cache_read = getattr(args, "cache_read", True)
    cache_write = getattr(args, "cache_write", True)
    # Opt-in transport artifacts (issue #159): absent unless --trajectories.
    capture_kw = {}
    trajectory_dir = getattr(args, "trajectories", None)
    if trajectory_dir is not None:
        from ._trajectory_capture import scan_trajectory_capture

        capture_kw["trajectory_capture"] = scan_trajectory_capture(
            args, identity, stem, material, fidelity
        )
    from ..api import source_content_key_fn

    content_key_fn = source_content_key_fn()
    try:
        if progress_timer is not None:
            progress_timer.start()
        result = run_sweep(
            cases,
            results,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_path=ckpt,
            max_workers=args.workers,
            resume=cache_read,
            content_key_fn=content_key_fn,
            cache_read=cache_read,
            cache_write=cache_write,
            progress=not getattr(args, "no_progress", False),
            on_progress=(
                _record_progress
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_case=(
                _note_case
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_runtime=_record_runtime if performance_logger is not None else None,
            on_timing=_record_timing if performance_logger is not None else None,
            on_activity=(
                _record_activity
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            deadline=deadline,
            max_seconds=(None if deadline is None else max(0.0, deadline - time.monotonic())),
            dataset_identity=identity,
            case_cost_fn=(
                case_cost
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            on_cost=(
                _record_cost
                if progress_file is not None or performance_logger is not None or has_dashboard
                else None
            ),
            metadata_only_complete=True,
            **capture_kw,
            observation=_sweep_observation(args, identity, settings, stem, content_key_fn),
        )
        # run_sweep returns a bool (complete?). Only a bare None -- test doubles
        # that predate the budget feature and don't bother returning anything --
        # is read as complete; every other falsy return fails loud as incomplete.
        complete = True if result is None else bool(result)
        if complete:
            from ..checkpoints.campaign_lock import write_lock
            from ..materials import CATALOG

            catalog_profile = str(identity.get("catalog_profile", "standard"))
            write_lock(
                ckpt,
                profile=catalog_profile,
                material=material,
                dataset_identity=identity,
                energy_grid_digest=CATALOG.profile_energy_grid_ref(catalog_profile, material),
            )
    except BaseException:
        with performance_lock:
            performance_state["state"] = "failed"
        if progress_file is not None:
            assert progress_timer is not None
            _write_progress_record(
                progress_file,
                material=material,
                phase=progress_phase,
                state="failed",
                **latest_progress,
                **latest_cost,
                **progress_timer.snapshot(
                    completed_new_cases=latest_progress["completed_new_cases"],
                    computed_cost=(
                        None
                        if initial_done_cost is None or "done_cost" not in latest_cost
                        else max(0.0, latest_cost["done_cost"] - initial_done_cost)
                    ),
                ),
            )
        if performance_logger is not None:
            performance_logger.close("failed")
        raise
    with performance_lock:
        performance_state["state"] = "done" if complete else "paused"
    if performance_logger is not None:
        performance_logger.close("done" if complete else "paused")

    if hasattr(args, "_job_records") and material in args._job_records:
        args._job_records[material]["state"] = "done" if complete else "paused"

    if progress_file is not None:
        assert progress_timer is not None
        _write_progress_record(
            progress_file,
            material=material,
            phase=progress_phase,
            state="done" if complete else "paused",
            activity="handoff",
            last_completed=dict(last_completed_case) or None,
            **latest_progress,
            **latest_cost,
            **progress_timer.snapshot(
                completed_new_cases=latest_progress["completed_new_cases"],
                computed_cost=(
                    None
                    if initial_done_cost is None or "done_cost" not in latest_cost
                    else max(0.0, latest_cost["done_cost"] - initial_done_cost)
                ),
            ),
        )
    n = max(
        sum(len(v) for v in results.values()),
        latest_progress["cached_cases"] + latest_progress["completed_new_cases"],
    )
    if complete:
        print(f"{_cli_core.paint('done', 'done')} -> {args.checkpoint_dir}/{stem}/ ({n} records)")
    else:
        print(
            f"{_cli_core.paint('paused', 'warning')} (budget) -> "
            f"{args.checkpoint_dir}/{stem}/ ({n} records)"
        )
    return complete
