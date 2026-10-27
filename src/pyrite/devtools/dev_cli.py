"""Small cross-platform developer command runner for pyrite.

This keeps agents and humans out of shell one-liner hell on Windows.
Exposed as the ``pyrite-dev`` console script (see ``[project.scripts]``); run via:

    uv run pyrite-dev <command>

Commands:
    acp-up     start the Claude and Codex ACP WebSocket bridges
    acp-down   stop bridges started by acp-up
    repo-map   print a compact repo tree and the canonical commands
    api        check governed Python API root exports (and optional alias identity)
    docs       clean, strict Sphinx build (optional external link check)
    lint       run Ruff over the repository
    format     run Ruff formatter
    typecheck  run ty
    precommit  run all pre-commit hooks
    nbqa       lint notebooks with nbQA + Ruff
    nbstrip    strip notebook outputs in-place
    test       run pytest, forwarding selectors and arguments
               (--numba disables JIT via NUMBA_DISABLE_JIT=1, must precede
               other forwarded args, to measure @njit bodies under --cov;
               runs without test paths use memory-bounded xdist workers,
               PYRITE_TEST_WORKERS=N overrides, 1 = serial;
               --slow also runs tests marked slow via PYRITE_SLOW_TESTS=1)
    test-suite run one focused domain suite (integration overlaps domains)
               with the same memory-bounded xdist worker default and --slow
               flag as test
    package-smoke build and install clean wheel/editable environments
    smoke      exercise checkpoint loading and plotting
    perf       measure MC sweeps with compute-performance telemetry
    startup    benchmark fresh-process CPU imports and the README simulation
    performance list, analyze, or delete compute-performance artifacts
    energy-grid maintain derived detector energy-grid artifacts
    release    bump version, gate on due deprecation removals, print release notes
    release-notes  render notes for a pushed vX.Y.Z tag (release workflow)
    sync-skills mirror .agents/skills into .claude/skills
    check-skills validate the canonical skills and exact mirror
    check-lines check source size (1,250 lines; marimo notebooks 4,000)
    complexity report source complexity; --check runs the high-CC ratchet,
               --write rewrites its baseline
    verify     check types, lint, formatting, source size, complexity ratchet,
               docs, and imports (no tests)
    cli-reference     write or --check docs/repo-design/cli/cli-reference.md
    cli-deprecations  write or --check docs/repo-design/cli/cli-deprecations.md
    validation-ledger write or --check generated validation-ledger views
    validation-records run one standalone check and write its evidence records
"""

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pyrite._acp import ACP_SERVERS, start_acp_servers, stop_acp_servers
from pyrite.devtools import repo_root

ROOT = repo_root()
AGENT_SKILLS_DIR = ROOT / ".agents" / "skills"
CLAUDE_SKILLS_DIR = ROOT / ".claude" / "skills"
LEGACY_NOTEBOOK = ROOT / "checks" / "feranchuk_analysis.ipynb"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

TEST_SUITE_PATTERNS = {
    "packaging": (
        "dev/test_agent_tooling.py",
        "cli/test_contract.py",
        "cli/test_reference.py",
        "detectors/test_package.py",
        "dev/test_dev.py",
        "dev/test_doc_blocks.py",
        "materials/test_materials_package.py",
        "montecarlo/test_exports.py",
        "plots/test_exports.py",
        "results/test_exports.py",
    ),
    "apps": (
        "notebooks/altair/test_*.py",
        "notebooks/analysis_app/test_app.py",
        "notebooks/analysis_app/test_analyze.py",
        "notebooks/analysis_app/test_export.py",
        "check/test_app.py",
        "plots/test_material_comparison.py",
        "notebooks/test_design.py",
        "plots/test_plotly_trajectories.py",
        "plots/test_render_*.py",
        "notebooks/test_scan_app.py",
        "notebooks/test_split_apps.py",
        "notebooks/test_trace_app.py",
        "plots/test_trajectories.py",
        "materials/test_validation_background.py",
        "notebooks/test_viewer.py",
    ),
    "cli": (
        "checkpoint/test_archive.py",
        "cli/test_blaze.py",
        "materials/test_catalog_startup_errors.py",
        "check/test_config.py",
        "checkpoint/test_cli.py",
        "checkpoint/test_gc.py",
        "cli/test_*.py",
        "energy-grid/test_cli.py",
        "scan/test_local_dashboard.py",
        "montecarlo/test_output_noise.py",
        "remote/test_remote.py",
        "remote/test_click.py",
        "scan/test_scan_*.py",
        "checkpoint/test_slim.py",
    ),
    # First match owns a module. Keep public-boundary suites above these
    # domains; keep specialized physics before each family catch-all.
    "quantum": (
        "materials/test_channeling_profile.py",
        "montecarlo/test_axial_*.py",
        "montecarlo/test_channeling_axial*.py",
        "montecarlo/test_channeling_density.py",
        "montecarlo/test_channeling_emission.py",
        "montecarlo/test_channeling_metrics.py",
        "montecarlo/test_channeling_mcwf.py",
        "montecarlo/test_channeling_operators.py",
        "montecarlo/test_channeling_quantum_regression.py",
        "montecarlo/test_channeling_states.py",
        "montecarlo/test_channeling_vortex.py",
    ),
    "channeling": (
        "montecarlo/test_channeling_*.py",
        "materials/test_channeling_*.py",
    ),
    "spectrum-coherent": (
        "montecarlo/test_coherent_*.py",
        "montecarlo/test_temporal_profile.py",
    ),
    "spectrum-brem": (
        "montecarlo/test_brem_*.py",
        "montecarlo/test_bremsstrahlung_*.py",
        "montecarlo/test_positron_brems.py",
    ),
    "spectrum": (
        "montecarlo/test_beam_energy_spread_grid.py",
        "montecarlo/test_characteristic.py",
        "montecarlo/test_cherenkov*.py",
        "montecarlo/test_chunk_invariance.py",
        "montecarlo/test_electron_blocks.py",
        "montecarlo/test_emission_adapters.py",
        "montecarlo/test_line_*.py",
        "montecarlo/test_mosaic.py",
        "montecarlo/test_mosaic_mc.py",
        "montecarlo/test_radiation_error_estimators.py",
        "montecarlo/test_segment_escape.py",
        "montecarlo/test_segment_staging.py",
        "montecarlo/test_sinc_*.py",
        "montecarlo/test_spectrum_*.py",
        "montecarlo/test_substep_invariance.py",
        "montecarlo/test_transition_radiation*.py",
        "montecarlo/test_xray_dispersion*.py",
    ),
    "transport-cascades": (
        "montecarlo/test_annihilation_cuda.py",
        "montecarlo/test_coupled_radiative_runner.py",
        "montecarlo/test_hard_radiative*.py",
        "montecarlo/test_pair_production*.py",
        "montecarlo/test_positron_*.py",
        "montecarlo/test_shell_secondary*.py",
    ),
    "transport-collisions": (
        "montecarlo/test_atomic_electron_deflection.py",
        "montecarlo/test_dielectric_loss.py",
        "montecarlo/test_eadl_relaxation.py",
        "montecarlo/test_eedl_ionization_consistency.py",
        "montecarlo/test_elastic_model_case.py",
        "montecarlo/test_elsepa_*.py",
        "montecarlo/test_energy_loss_straggling.py",
        "montecarlo/test_inelastic_partition.py",
        "montecarlo/test_mott_coverage.py",
        "montecarlo/test_sbethe_*.py",
        "montecarlo/test_shell_*.py",
        "montecarlo/test_stopping_*.py",
        "montecarlo/test_straggling_*.py",
    ),
    "transport-runner": (
        "montecarlo/test_adaptive_stopping.py",
        "montecarlo/test_block_transport.py",
        "montecarlo/test_case.py",
        "montecarlo/test_gpu_oom_retry.py",
        "montecarlo/test_imports.py",
        "montecarlo/test_paraview_preset.py",
        "montecarlo/test_physics_baseline.py",
        "montecarlo/test_pool_cadence.py",
        "montecarlo/test_stage_counters.py",
        "montecarlo/test_trajectory_*.py",
        "montecarlo/test_transport_batch_progress.py",
        "montecarlo/test_transport_core_default.py",
    ),
    # Flight integration, geometry, beam entry and event contracts remain here.
    "transport": ("montecarlo/test_*.py",),
    "energy-grid": ("energy-grid/test_*.py",),
    "materials": ("materials/test_*.py",),
    "xsgen": ("xsgen/test_*.py",),
}

INTEGRATION_TESTS = (
    "notebooks/analysis_app/test_app.py",
    "cli/test_contract.py",
    "materials/test_materials_package.py",
    "montecarlo/test_exports.py",
    "remote/test_remote.py",
    "results/test_exports.py",
    "scan/test_run.py",
    "scan/test_sweep.py",
)
PRIMARY_TEST_SUITE_NAMES = ("core", *TEST_SUITE_PATTERNS)
TEST_SUITE_NAMES = (*PRIMARY_TEST_SUITE_NAMES, "integration")


class AgentToolingError(RuntimeError):
    """Raised when the portable skill tree is malformed or out of sync."""


def run(*args: str, cwd: Path = ROOT, extra_env: dict[str, str] | None = None) -> None:
    env = {**os.environ, **extra_env} if extra_env else None
    subprocess.run([sys.executable, *args], cwd=cwd, check=True, env=env)


def run_uv(*args: str, cwd: Path = ROOT, extra_env: dict[str, str] | None = None) -> None:
    """Run a locked project tool, provisioning its explicit dependency group."""
    env = {**os.environ, **extra_env} if extra_env else None
    subprocess.run(["uv", "run", *args], cwd=cwd, check=True, env=env)


def cmd_acp_up(_: argparse.Namespace) -> None:
    processes = start_acp_servers()
    print("ACP bridges running:")
    for name, (_host, port) in ACP_SERVERS.items():
        print(f"  {name}: ws://localhost:{port}")
    print("Press Ctrl-C to stop both bridges, or run acp-down from another terminal.")
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_acp_servers()


def cmd_acp_down(_: argparse.Namespace) -> None:
    stop_acp_servers()


def iter_notebooks() -> list[Path]:
    return [LEGACY_NOTEBOOK] if LEGACY_NOTEBOOK.is_file() else []


def cmd_repo_map(args: argparse.Namespace) -> None:
    from pyrite.devtools import repo_map

    if getattr(args, "json", False):
        import json

        print(json.dumps(repo_map.dependency_report(ROOT / "src"), indent=2, sort_keys=True))
        return

    write = getattr(args, "write", False)
    check = getattr(args, "check", False)
    if write or check:
        current = repo_map.write_or_check(root=ROOT, check=check)
        if not current:
            print(
                "repository map dependency graph changed; "
                "regenerate with `pyrite-dev repo-map --write`",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    interesting = [
        "src/pyrite",
        "tests",
        "checks",
        "docs",
        "dev",
        "scripts",
    ]
    print("pyrite repo map")
    for rel in interesting:
        path = ROOT / rel
        if not path.exists():
            continue
        if path.is_dir():
            print(f"{rel}/")
            entries = []
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                if child.is_dir():
                    entries.append(f"  {child.name}/")
                else:
                    entries.append(f"  {child.name}")
            for line in entries[:40]:
                print(line)
            if len(entries) > 40:
                print("  ...")
            print()
        else:
            print(rel)
    print("Agent tooling:")
    for rel in [".agents", ".claude", "agentdocs"]:
        if (ROOT / rel).exists():
            print(f"  {rel}/")
    print()
    print("Canonical commands:")
    for line in [
        "uv run pyrite-dev acp-up",
        "uv run pyrite-dev acp-down",
        "uv run pyrite-dev lint",
        "uv run pyrite-dev format",
        "uv run pyrite-dev typecheck",
        "uv run pyrite-dev check-lines",
        "uv run pyrite-dev complexity --check",
        "uv run pyrite-dev precommit",
        "uv run pyrite-dev test path/to/test.py -k test_name",
        "uv run pyrite-dev test-suite transport",
        "uv run pyrite-dev test-suite spectrum",
        "uv run pyrite-dev package-smoke",
        "uv run pyrite-dev smoke --material hopg --output-dir /tmp/pyrite-smoke",
        "uv run pyrite-dev docs",
        "uv run pyrite-dev cli-reference --check",
        "uv run pyrite-dev cli-deprecations --check",
        "uv run pyrite-dev validation-ledger --check",
        "uv run pyrite-dev validation-records -- checks/dans_diffraction_oracle.py",
        "uv run pyrite-dev repo-map --check",
        "uv run pyrite-dev sync-skills",
        "uv run pyrite-dev check-skills",
        "uv run pyrite-dev verify",
        "uv run pyrite-dev nbqa",
        "uv run pyrite-dev nbstrip",
    ]:
        print(f"  {line}")


def cmd_lint(_: argparse.Namespace) -> None:
    run("-m", "ruff", "check", ".")


def cmd_format(_: argparse.Namespace) -> None:
    run("-m", "ruff", "format", ".")


def cmd_typecheck(_: argparse.Namespace) -> None:
    # Newer ty releases emit useful warnings for existing patterns. Keep those
    # visible without making the gate fail unless there is an actual type error.
    run("-m", "ty", "check", "--exit-zero-on-warning")


def cmd_precommit(_: argparse.Namespace) -> None:
    run("-m", "pre_commit", "run", "--all-files")


def cmd_nbqa(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run(
        "-m",
        "nbqa",
        "ruff check",
        *[str(p) for p in notebooks],
        "--nbqa-shell",
    )


def cmd_nbstrip(_: argparse.Namespace) -> None:
    notebooks = iter_notebooks()
    if not notebooks:
        print("No notebooks found.")
        return
    run("-m", "nbstripout", *[str(p) for p in notebooks])


# Measured per-worker peak RSS over the full suite is ~1.2-1.8 GiB; budget 2 GiB
# so a laptop with other work running does not swap or OOM.
TEST_WORKER_MEMORY_BYTES = 2 * 1024**3
MAX_TEST_WORKERS = 6


def default_test_workers() -> int:
    """xdist worker count for an untargeted run: ``PYRITE_TEST_WORKERS`` or a
    CPU- and available-memory-bounded default (1 means serial)."""
    override = os.environ.get("PYRITE_TEST_WORKERS")
    if override:
        return max(1, int(override))
    import psutil

    by_memory = psutil.virtual_memory().available // TEST_WORKER_MEMORY_BYTES
    return max(1, min(os.cpu_count() or 1, by_memory, MAX_TEST_WORKERS))


def _selects_test_paths(pytest_args: list[str]) -> bool:
    return any(
        not arg.startswith("-") and ("::" in arg or (ROOT / arg).exists() or Path(arg).exists())
        for arg in pytest_args
    )


def parallel_pytest_args(pytest_args: list[str]) -> list[str]:
    """Prepend xdist workers unless the caller supplies a targeted selection.

    Call with forwarded arguments only; ``test-suite`` adds its owned paths
    afterwards. Runs that name test paths, choose ``-n``/``--numprocesses``,
    or disable xdist are forwarded unchanged.
    """
    explicit = any(
        arg in {"-n", "--numprocesses", "no:xdist"}
        or arg.startswith(("-n=", "--numprocesses="))
        or (arg.startswith("-n") and arg[2:].isdigit())
        for arg in pytest_args
    )
    if explicit or _selects_test_paths(pytest_args):
        return pytest_args
    workers = default_test_workers()
    if workers < 2:
        return pytest_args
    print(
        f"pyrite-dev: {workers} xdist workers (PYRITE_TEST_WORKERS=1 for serial)",
        file=sys.stderr,
    )
    return ["-n", str(workers), "--dist", "worksteal", *pytest_args]


def _pytest_env(args: argparse.Namespace) -> dict[str, str]:
    """Environment overrides selected by ``test``/``test-suite`` flags."""
    env = {}
    if getattr(args, "numba", False):
        env["NUMBA_DISABLE_JIT"] = "1"
    if getattr(args, "slow", False):
        env["PYRITE_SLOW_TESTS"] = "1"
    return env


def _run_pytest(args: argparse.Namespace, *pytest_args: str) -> None:
    env = _pytest_env(args)
    if env:
        run("-m", "pytest", *pytest_args, extra_env=env)
    else:
        run("-m", "pytest", *pytest_args)


def cmd_test(args: argparse.Namespace) -> None:
    _run_pytest(args, *parallel_pytest_args(getattr(args, "pytest_args", [])))


def test_files_for_suite(name: str, root: Path = ROOT) -> list[Path]:
    """Return deterministic repository test paths for one documented suite.

    Primary domain suites partition every ``tests/**/test_*.py`` module once.
    ``core`` owns shared infrastructure and modules outside the named domains.
    ``integration`` intentionally samples public boundaries across domains; the
    full ``pyrite-dev test`` gate remains unchanged.
    """
    tests_dir = root / "tests"
    files = sorted(tests_dir.rglob("test_*.py"))
    if name == "integration":
        selected = [tests_dir / filename for filename in INTEGRATION_TESTS]
        missing = [path.name for path in selected if not path.is_file()]
        if missing:
            raise AgentToolingError("missing integration tests: " + ", ".join(missing))
        return selected
    if name not in {"core", *TEST_SUITE_PATTERNS}:
        raise AgentToolingError(f"unknown test suite {name!r}")

    selected = []
    for path in files:
        owner = "core"
        for candidate, patterns in TEST_SUITE_PATTERNS.items():
            relative = path.relative_to(tests_dir).as_posix()
            if any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
                owner = candidate
                break
        if owner == name:
            selected.append(path)
    return selected


def cmd_test_suite(args: argparse.Namespace) -> None:
    paths = [str(path.relative_to(ROOT)) for path in test_files_for_suite(args.suite)]
    if not paths:
        raise AgentToolingError(f"test suite {args.suite!r} contains no test modules")
    pytest_args = list(getattr(args, "pytest_args", []))
    # argparse's REMAINDER swallows a flag that follows the suite name.
    if pytest_args[:1] == ["--slow"]:
        args.slow = True
        pytest_args = pytest_args[1:]
    _run_pytest(args, *parallel_pytest_args(pytest_args), *paths)


def cmd_smoke(args: argparse.Namespace) -> None:
    from pyrite.devtools.smoke import main

    status = main(["--material", args.material, "--output-dir", args.output_dir])
    if status:
        raise SystemExit(status)


def cmd_package_smoke(_: argparse.Namespace) -> None:
    from pyrite.devtools.package_smoke import main

    main()


def _frontmatter_fields(path: Path) -> dict[str, str]:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentToolingError(f"{path}: cannot read UTF-8 text: {exc}") from exc

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise AgentToolingError(f"{path}: missing or unterminated YAML frontmatter")

    frontmatter_text = parts[1]
    try:
        fields = yaml.safe_load(frontmatter_text)
        if not isinstance(fields, dict):
            raise ValueError("Frontmatter is not a dictionary")
        return {str(k): str(v) for k, v in fields.items()}
    except Exception as exc:
        raise AgentToolingError(f"{path}: malformed YAML frontmatter: {exc}") from exc


def validate_skill_file(path: Path) -> None:
    """Validate the required portable frontmatter for one skill."""
    fields = _frontmatter_fields(path)
    for field in ("name", "description"):
        if field not in fields:
            raise AgentToolingError(f"{path}: missing {field} in YAML frontmatter")
    name = fields["name"]
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise AgentToolingError(f"{path}: invalid skill name {name!r}")
    if name != path.parent.name:
        raise AgentToolingError(
            f"{path}: skill name {name!r} must match directory {path.parent.name!r}"
        )
    if not fields["description"].startswith("Use when"):
        raise AgentToolingError(f"{path}: description must start with 'Use when'")


def _relative_files(root: Path) -> dict[Path, Path]:
    if not root.is_dir():
        raise AgentToolingError(f"missing skill directory: {root}")
    return {path.relative_to(root): path for path in sorted(root.rglob("*")) if path.is_file()}


def _validated_canonical_files(root: Path) -> dict[Path, Path]:
    files = _relative_files(root)
    skill_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not skill_dirs:
        raise AgentToolingError(f"no skill directories found under {root}")
    for skill_dir in skill_dirs:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            raise AgentToolingError(f"{skill_dir}: missing SKILL.md")
        validate_skill_file(skill_file)
    return files


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def check_skill_trees(canonical: Path = AGENT_SKILLS_DIR, mirror: Path = CLAUDE_SKILLS_DIR) -> None:
    """Validate canonical skills and require an exact byte-for-byte mirror."""
    canonical_files = _validated_canonical_files(canonical)
    mirror_files = _relative_files(mirror)

    canonical_names = set(canonical_files)
    mirror_names = set(mirror_files)
    missing = sorted(canonical_names - mirror_names)
    extra = sorted(mirror_names - canonical_names)
    different = sorted(
        rel
        for rel in canonical_names & mirror_names
        if canonical_files[rel].read_bytes() != mirror_files[rel].read_bytes()
        or mirror_files[rel].is_symlink()
    )
    errors = []
    if missing:
        errors.append("missing from mirror: " + ", ".join(map(str, missing)))
    if extra:
        errors.append("unexpected mirror files: " + ", ".join(map(str, extra)))
    if different:
        errors.append("byte-different mirror files: " + ", ".join(map(str, different)))
    if errors:
        raise AgentToolingError("; ".join(errors))


def sync_skill_trees(canonical: Path = AGENT_SKILLS_DIR, mirror: Path = CLAUDE_SKILLS_DIR) -> None:
    """Replace the generated mirror with ordinary copies of canonical files."""
    _validated_canonical_files(canonical)
    staged = mirror.with_name(f".{mirror.name}.tmp")
    backup = mirror.with_name(f".{mirror.name}.backup")

    for path in (staged, backup):
        _remove_path(path)

    shutil.copytree(canonical, staged, symlinks=False)
    check_skill_trees(canonical, staged)

    had_mirror = mirror.exists() or mirror.is_symlink()
    if had_mirror:
        mirror.replace(backup)
    try:
        staged.replace(mirror)
    except OSError:
        if had_mirror and not mirror.exists():
            backup.replace(mirror)
        raise
    finally:
        _remove_path(staged)
    _remove_path(backup)


def cmd_sync_skills(_: argparse.Namespace) -> None:
    sync_skill_trees()
    check_skill_trees()
    print(
        f"Synchronized {AGENT_SKILLS_DIR.relative_to(ROOT)} -> {CLAUDE_SKILLS_DIR.relative_to(ROOT)}"
    )


def cmd_check_skills(_: argparse.Namespace) -> None:
    check_skill_trees()
    print("Skill mirror is valid and synchronized.")


def cmd_imports(_: argparse.Namespace) -> None:
    subprocess.run(["lint-imports", "--no-cache"], cwd=ROOT, check=True)


def cmd_docs(args: argparse.Namespace) -> None:
    """Build all maintained and generated documentation from clean state."""
    from pyrite.devtools.docs_paths import check_doc_paths
    from pyrite.devtools.validation_ledger import write_or_check

    check_doc_paths(ROOT)
    cmd_cli_reference(argparse.Namespace(check=True))
    ledger = ROOT / "docs" / "validation" / "physics-validation-ledger.md"
    if not write_or_check(ledger, check=True):
        raise RuntimeError(
            "generated validation views are stale; run `pyrite-dev validation-ledger --write`"
        )
    docs_dir = ROOT / "docs"
    _remove_path(docs_dir / "_autosummary")
    _remove_path(docs_dir / "_build")
    builder = "linkcheck" if getattr(args, "linkcheck", False) else "html"
    run_uv(
        "--group",
        "docs",
        "sphinx-build",
        "-E",
        "-a",
        "-W",
        "--keep-going",
        "-b",
        builder,
        str(docs_dir),
        str(docs_dir / "_build" / builder),
        extra_env={"PYRITE_MC_BACKEND": "cpu"},
    )


def cmd_check_lines(_: argparse.Namespace) -> None:
    from pyrite.devtools.source_checks import oversized_source_modules, source_line_limit

    package = ROOT / "src" / "pyrite"
    oversized = oversized_source_modules(package)
    if oversized:
        details = ", ".join(
            f"{path}: {count} (limit {source_line_limit(path, (package / path).read_text())})"
            for path, count in oversized.items()
        )
        raise AgentToolingError(f"source modules exceed their line budgets: {details}")


def cmd_complexity(args: argparse.Namespace) -> None:
    from pyrite.devtools.complexity import ComplexityToolError, run

    try:
        status = run(
            ROOT,
            scope=getattr(args, "scope", "src"),
            check=getattr(args, "check", False),
            write=getattr(args, "write", False),
            as_json=getattr(args, "json", False),
            top=getattr(args, "top", 15),
            churn=getattr(args, "churn", None),
        )
    except ComplexityToolError as exc:
        raise SystemExit(f"complexity: {exc}") from exc
    if status:
        raise SystemExit(status)


def cmd_api(args: argparse.Namespace) -> None:
    """Check the governed Python API inventory without changing exports."""
    from pyrite.devtools.api_surface import check

    errors = check(ROOT, imports=getattr(args, "imports", False))
    if errors:
        for error in errors:
            print(f"api: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("API root namespace inventory OK")


def cmd_verify(args: argparse.Namespace) -> None:
    """Run static repository gates; test selection is always explicit."""
    cmd_typecheck(args)
    cmd_lint(args)
    run("-m", "ruff", "format", "--check", ".")
    cmd_check_lines(args)
    cmd_complexity(argparse.Namespace(check=True))
    cmd_api(argparse.Namespace(imports=False))
    cmd_docs(argparse.Namespace(linkcheck=False))
    cmd_imports(args)


def cmd_regen_golden(args: argparse.Namespace) -> None:
    from pyrite.energy_grid.golden import regen

    raise SystemExit(regen(check=getattr(args, "check", False)))


def cmd_cli_deprecations(args: argparse.Namespace) -> None:
    from pyrite.devtools.cli_deprecations import main

    target = ROOT / "docs" / "repo-design" / "cli" / "cli-deprecations.md"
    mode = "--check" if getattr(args, "check", False) else "--write"
    status = main([mode, str(target)])
    if status:
        raise SystemExit(status)


def cmd_release(args: argparse.Namespace) -> None:
    from pyrite import __version__ as old
    from pyrite.devtools import release

    try:
        if release.parse_version(args.version) <= release.parse_version(old):
            raise release.ReleaseError(f"target {args.version} must be greater than {old}")
        due = release.due_removals(args.version)
        if due:
            rows = "\n".join(f"  {label}: remove in {target}" for label, target in due.items())
            raise release.ReleaseError(
                f"{len(due)} deprecation removal(s) due at or before {args.version} remain:\n{rows}"
            )
        base = release.notes_base(ROOT, args.base)
        notes = release.render_notes(args.version, base, release.collect_changes(ROOT, base))
        if not args.check:
            release.bump_versions(ROOT, args.version)
            # uv.lock records the project version; CI syncs with --locked.
            subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
            # Generated docs embed __version__, so regenerate in fresh processes.
            for command in ("cli-reference", "cli-deprecations"):
                run("-m", "pyrite._dev", command, "--write")
    except release.ReleaseError as exc:
        raise SystemExit(f"release: {exc}") from exc
    if args.notes_file:
        Path(args.notes_file).write_text(notes, encoding="utf-8")
    print(notes, end="")


def cmd_release_notes(args: argparse.Namespace) -> None:
    from pyrite.devtools import release

    try:
        notes = release.write_tag_notes(ROOT, args.tag, args.notes_file)
    except release.ReleaseError as exc:
        raise SystemExit(f"release-notes: {exc}") from exc
    print(notes, end="")


def cmd_cli_reference(args: argparse.Namespace) -> None:
    from pyrite.devtools.cli_reference import main

    target = ROOT / "docs" / "repo-design" / "cli" / "cli-reference.md"
    mode = "--check" if getattr(args, "check", False) else "--write"
    status = main([mode, str(target)])
    if status:
        raise SystemExit(status)


def cmd_validation_ledger(args: argparse.Namespace) -> None:
    from pyrite.devtools.validation_ledger import write_or_check

    ledger = ROOT / "docs" / "validation" / "physics-validation-ledger.md"
    check = getattr(args, "check", False)
    if not write_or_check(ledger, check=check):
        print(
            "generated validation views are stale; regenerate with "
            "`pyrite-dev validation-ledger --write`",
            file=sys.stderr,
        )
        raise SystemExit(1)


def cmd_validation_records(args: argparse.Namespace) -> None:
    """Run a mapped standalone check and persist one result per ledger claim."""
    from pyrite.validation.check_records import current_revision, records_for_exit, write_records

    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        raise SystemExit("validation-records requires a check command after `--`")
    check = Path(command[0])
    check_path = check if check.is_absolute() else ROOT / check
    try:
        check_name = check_path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise SystemExit(f"check must be beneath {ROOT}: {check}") from exc
    try:
        records_for_exit(check_name, 0, "pending")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = args.output_dir or ROOT / "docs" / "validation" / "check-records"
    result = subprocess.run(
        [sys.executable, str(check_path), *command[1:]],
        cwd=ROOT,
        check=False,
    )
    records = records_for_exit(check_name, result.returncode, current_revision())
    write_records(output_dir / f"{check_path.stem}.jsonl", records)
    raise SystemExit(result.returncode)


def _run_relocated_click(command, argv: list[str], *, prog_name: str) -> None:
    """Run a relocated Click tree while preserving stream and exit contracts."""
    from pyrite.console.output import run as run_cli
    from pyrite.console.outputs import enable_legacy_migration

    enable_legacy_migration()
    status = run_cli(command, argv, prog_name=prog_name)
    if isinstance(status, int) and status:
        raise SystemExit(status)


def cmd_performance(args: argparse.Namespace) -> None:
    from pyrite.devtools.cli_commands import performance_command

    _run_relocated_click(
        performance_command,
        args.command_args,
        prog_name="pyrite-dev performance",
    )


def cmd_perf(args: argparse.Namespace) -> None:
    from pyrite.cli.commands.scan import performance_command

    _run_relocated_click(
        performance_command,
        args.command_args,
        prog_name="pyrite-dev perf",
    )


def cmd_startup(args: argparse.Namespace) -> None:
    from pyrite.devtools.startup import run

    run(args)


def _positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return count


def _positive_repeats(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--repeats must be a positive integer")
    return count


def cmd_energy_grid(args: argparse.Namespace) -> None:
    from pyrite.devtools.cli_commands import energy_grid_command

    _run_relocated_click(
        energy_grid_command,
        args.command_args,
        prog_name="pyrite-dev energy-grid",
    )


def _cmd_bundled_object(args: argparse.Namespace, kind: str) -> None:
    """Run an object command tree against packaged data, ignoring user state."""
    from importlib import import_module

    from pyrite._catalog_layout import bundled_catalog, bundled_profile_writes

    command = import_module(f"pyrite.cli.commands.{kind}").command
    previous = os.environ.get("PYRITE_CATALOG")
    os.environ["PYRITE_CATALOG"] = str(bundled_catalog())
    try:
        with bundled_profile_writes():
            _run_relocated_click(command, args.command_args, prog_name=f"pyrite-dev {kind}")
    finally:
        if previous is None:
            os.environ.pop("PYRITE_CATALOG", None)
        else:
            os.environ["PYRITE_CATALOG"] = previous


def cmd_profile(args: argparse.Namespace) -> None:
    """Maintain bundled demo profiles without the user layer."""
    _cmd_bundled_object(args, "profile")


def cmd_beam(args: argparse.Namespace) -> None:
    """Maintain bundled example beams without the user layer."""
    _cmd_bundled_object(args, "beam")


def cmd_detector(args: argparse.Namespace) -> None:
    """Maintain bundled example detectors without the user layer."""
    _cmd_bundled_object(args, "detector")


def cmd_material(args: argparse.Namespace) -> None:
    """Explicit maintainer authoring against the checkout's bundled catalog."""
    from pyrite._catalog_layout import bundled_catalog, bundled_profile_writes
    from pyrite.cli.commands.material import command

    checkout = ROOT / "src" / "pyrite" / "data" / "catalog"
    if not (ROOT / "pyproject.toml").is_file() or checkout.resolve() != bundled_catalog().resolve():
        raise SystemExit("pyrite-dev material requires a source checkout with its editable install")
    previous = os.environ.get("PYRITE_CATALOG")
    os.environ["PYRITE_CATALOG"] = str(checkout)
    try:
        with bundled_profile_writes():
            _run_relocated_click(command, args.command_args, prog_name="pyrite-dev material")
    finally:
        if previous is None:
            os.environ.pop("PYRITE_CATALOG", None)
        else:
            os.environ["PYRITE_CATALOG"] = previous


_SLOW_HELP = "also run tests marked slow (sets PYRITE_SLOW_TESTS=1); must precede pytest_args"


def build_parser(prog_name: str = "pyrite-dev") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog_name)
    sub = ap.add_subparsers(dest="command", required=True)

    for name, fn in [
        ("acp-up", cmd_acp_up),
        ("acp-down", cmd_acp_down),
        ("lint", cmd_lint),
        ("format", cmd_format),
        ("typecheck", cmd_typecheck),
        ("precommit", cmd_precommit),
        ("nbqa", cmd_nbqa),
        ("nbstrip", cmd_nbstrip),
        ("sync-skills", cmd_sync_skills),
        ("check-skills", cmd_check_skills),
        ("check-lines", cmd_check_lines),
        ("package-smoke", cmd_package_smoke),
    ]:
        sp = sub.add_parser(name)
        sp.set_defaults(func=fn)
    repo_map = sub.add_parser(
        "repo-map",
        help="print repository inventory or update its generated dependency graph",
    )
    repo_map_mode = repo_map.add_mutually_exclusive_group()
    repo_map_mode.add_argument(
        "--write",
        action="store_true",
        help="replace the generated dependency region in docs/repo_map.md",
    )
    repo_map_mode.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated dependency region is stale",
    )
    repo_map_mode.add_argument(
        "--json",
        action="store_true",
        help="print deterministic module/package edges and SCCs as JSON",
    )
    repo_map.set_defaults(func=cmd_repo_map)
    docs = sub.add_parser(
        "docs",
        help="clean and strictly build the Sphinx documentation",
    )
    docs.add_argument(
        "--linkcheck",
        action="store_true",
        help="check external links (requires network access; excluded from offline gates)",
    )
    docs.set_defaults(func=cmd_docs)
    test = sub.add_parser("test")
    test.add_argument(
        "--numba",
        action="store_true",
        help=(
            "set NUMBA_DISABLE_JIT=1 so @njit bodies (transport.py, geometry.py, "
            "groove.py) are traced by coverage instead of running compiled; "
            "must precede pytest_args, e.g. `pyrite-dev test --numba --cov`"
        ),
    )
    test.add_argument("--slow", action="store_true", help=_SLOW_HELP)
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test.set_defaults(func=cmd_test)
    test_suite = sub.add_parser(
        "test-suite",
        help="run one focused domain suite",
        description=(
            "Run one domain partition; core contains shared/residual tests. "
            "integration is an overlapping public-boundary sample. "
            "Use test for the full suite."
        ),
    )
    test_suite.add_argument("suite", choices=TEST_SUITE_NAMES)
    test_suite.add_argument("--slow", action="store_true", help=_SLOW_HELP)
    test_suite.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test_suite.set_defaults(func=cmd_test_suite)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--material", default="hopg")
    smoke.add_argument("--output-dir", default="smoke_out")
    smoke.set_defaults(func=cmd_smoke)
    complexity = sub.add_parser(
        "complexity",
        help="report source complexity or run the high-CC ratchet",
        description=(
            "Report Radon CC/MI and optional scc inventory and git churn; "
            "--check fails on new or worsened routines above CC 20."
        ),
    )
    complexity_mode = complexity.add_mutually_exclusive_group()
    complexity_mode.add_argument(
        "--check", action="store_true", help="fail on new or worsened high-CC routines"
    )
    complexity_mode.add_argument(
        "--write", action="store_true", help="rewrite the baseline and print its changes"
    )
    complexity_mode.add_argument("--json", action="store_true", help="emit the report as JSON")
    complexity.add_argument(
        "--scope", choices=("src", "tests"), default="src", help="tests are report-only"
    )
    complexity.add_argument("--top", type=_positive_int, default=15, help="list length")
    complexity.add_argument(
        "--churn", type=_positive_int, metavar="N", help="add git churn over N commits"
    )
    complexity.set_defaults(func=cmd_complexity)
    api = sub.add_parser("api", help="check the governed Python API namespace")
    api_sub = api.add_subparsers(dest="api_command", required=True)
    api_check = api_sub.add_parser("check", help="check the root export inventory")
    api_check.add_argument(
        "--imports",
        action="store_true",
        help="also import aliases and check object identity (may initialize engines)",
    )
    api_check.set_defaults(func=cmd_api)
    verify = sub.add_parser(
        "verify",
        help="run static repository checks (no tests)",
        description=(
            "Check types, lint, formatting, source size, complexity ratchet, docs, and imports; "
            "no tests run."
        ),
    )
    verify.add_argument(
        "--skip-tests",
        action="store_true",
        help=argparse.SUPPRESS,  # Compatible no-op for older automation.
    )
    verify.set_defaults(func=cmd_verify)
    regen_golden = sub.add_parser("regen-golden")
    regen_golden.add_argument("--check", action="store_true")
    regen_golden.set_defaults(func=cmd_regen_golden)
    performance = sub.add_parser(
        "performance",
        add_help=False,
        help="list, analyze, or delete compute-performance artifacts",
    )
    performance.add_argument("command_args", nargs=argparse.REMAINDER)
    performance.set_defaults(func=cmd_performance)
    perf = sub.add_parser(
        "perf",
        add_help=False,
        help="measure MC sweeps with compute-performance telemetry",
    )
    perf.add_argument("command_args", nargs=argparse.REMAINDER)
    perf.set_defaults(func=cmd_perf)
    startup = sub.add_parser(
        "startup", help="benchmark fresh-process CPU imports and the README simulation"
    )
    startup.add_argument(
        "--repeats",
        type=_positive_repeats,
        default=3,
        help="fresh processes (positive integer; default: 3)",
    )
    startup.add_argument(
        "--cache",
        choices=("warm", "cold"),
        default="warm",
        help="Numba cache: prime first, or use an empty temporary cache per sample (default: warm)",
    )
    startup.add_argument("--json", action="store_true", help="emit one JSON benchmark report")
    startup.add_argument(
        "--profile",
        type=Path,
        help="write cProfile data from a separate fresh-process first call",
    )
    startup.set_defaults(func=cmd_startup)
    energy_grid = sub.add_parser(
        "energy-grid",
        add_help=False,
        help="maintain derived detector energy-grid artifacts",
    )
    energy_grid.add_argument("command_args", nargs=argparse.REMAINDER)
    energy_grid.set_defaults(func=cmd_energy_grid)
    for kind, handler in (("profile", cmd_profile), ("beam", cmd_beam), ("detector", cmd_detector)):
        objects = sub.add_parser(
            kind,
            add_help=False,
            help=f"create and edit bundled {kind} examples (maintainers only)",
        )
        objects.add_argument("command_args", nargs=argparse.REMAINDER)
        objects.set_defaults(func=handler)
    material = sub.add_parser(
        "material",
        add_help=False,
        help="author bundled materials (checkout maintainers only)",
    )
    material.add_argument("command_args", nargs=argparse.REMAINDER)
    material.set_defaults(func=cmd_material)
    cli_deprecations = sub.add_parser("cli-deprecations")
    cli_deprecation_mode = cli_deprecations.add_mutually_exclusive_group()
    cli_deprecation_mode.add_argument("--write", action="store_true")
    cli_deprecation_mode.add_argument("--check", action="store_true")
    cli_deprecations.set_defaults(func=cmd_cli_deprecations)
    release_cmd = sub.add_parser(
        "release", help="bump version, regenerate docs, print release notes (see releasing.md)"
    )
    release_cmd.add_argument("version", help="target version X.Y.Z")
    release_cmd.add_argument("--base", help="notes base rev (default: last v* tag/release commit)")
    release_cmd.add_argument("--notes-file", help="also write the release notes to this path")
    release_cmd.add_argument(
        "--check", action="store_true", help="validate and print notes without editing files"
    )
    release_cmd.set_defaults(func=cmd_release)
    release_notes = sub.add_parser(
        "release-notes", help="render notes for a pushed vX.Y.Z tag (release workflow)"
    )
    release_notes.add_argument("tag", help="the pushed tag, vX.Y.Z; must match __version__")
    release_notes.add_argument("--notes-file", help="also write the release notes to this path")
    release_notes.set_defaults(func=cmd_release_notes)
    cli_reference = sub.add_parser("cli-reference")
    cli_reference_mode = cli_reference.add_mutually_exclusive_group()
    cli_reference_mode.add_argument("--write", action="store_true")
    cli_reference_mode.add_argument("--check", action="store_true")
    cli_reference.set_defaults(func=cmd_cli_reference)
    validation_ledger = sub.add_parser("validation-ledger")
    validation_mode = validation_ledger.add_mutually_exclusive_group(required=True)
    validation_mode.add_argument("--write", action="store_true")
    validation_mode.add_argument("--check", action="store_true")
    validation_ledger.set_defaults(func=cmd_validation_ledger)
    validation_records = sub.add_parser(
        "validation-records",
        help="run a mapped standalone check and write its machine-readable evidence",
    )
    validation_records.add_argument(
        "--output-dir",
        type=Path,
        help="write records here instead of docs/validation/check-records",
    )
    validation_records.add_argument("command", nargs=argparse.REMAINDER)
    validation_records.set_defaults(func=cmd_validation_records)
    return ap


def main(argv: list[str] | None = None, *, prog_name: str = "pyrite-dev") -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in {
        "perf",
        "performance",
        "energy-grid",
        "profile",
        "beam",
        "detector",
        "material",
    }:
        command = raw_args[0]
        func = {
            "perf": cmd_perf,
            "performance": cmd_performance,
            "energy-grid": cmd_energy_grid,
            "profile": cmd_profile,
            "beam": cmd_beam,
            "detector": cmd_detector,
            "material": cmd_material,
        }[command]
        func(argparse.Namespace(command=command, command_args=raw_args[1:]))
        return
    if raw_args and raw_args[0] == "test":
        rest = raw_args[1:]
        # pyrite-dev's own flags lead, in any order; the rest goes to pytest.
        flags = set()
        while rest and rest[0] in {"--numba", "--slow"} - flags:
            flags.add(rest.pop(0))
        cmd_test(
            argparse.Namespace(
                command="test",
                numba="--numba" in flags,
                slow="--slow" in flags,
                pytest_args=rest,
            )
        )
        return
    args = build_parser(prog_name).parse_args(raw_args)
    args.func(args)


if __name__ == "__main__":
    main()
