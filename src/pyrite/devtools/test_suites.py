"""Authoritative test-suite inventory and path ownership, independent of the CLI."""

import fnmatch
from pathlib import Path, PurePosixPath

from pyrite.devtools import repo_root
from pyrite.devtools.errors import AgentToolingError

ROOT = repo_root()

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
        "notebooks/analysis_app/test_app.py",
        "notebooks/analysis_app/test_analyze.py",
        "notebooks/analysis_app/test_export.py",
        "check/test_app.py",
        "plots/test_material_comparison.py",
        "plots/test_penetration_views.py",
        "plots/test_plotly_detectors.py",
        "plots/test_plotly_material_comparison.py",
        "plots/test_plotly_spectra.py",
        "plots/test_plotly_spectra_compare.py",
        "plots/test_plotly_sweeps.py",
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
        "remote/test_remote_command_*.py",
        "remote/test_click.py",
        "scan/test_scan_*.py",
        "checkpoint/test_slim.py",
    ),
    # Genuine cross-directory exceptions to TEST_SUITE_DIRECTORIES.
    "transport": ("montecarlo/test_cone_split.py", "montecarlo/test_cone_split_weights.py"),
    "quantum": ("montecarlo/channeling/test_channeling_bloch.py",),
    "channeling": ("materials/test_channeling_*.py",),
}

# Directory-owned suites (#854): every module below one of these directories
# belongs to its suite, so a new module needs no edit here. TEST_SUITE_PATTERNS
# entries (first match) win over directories; ``core`` owns the rest.
TEST_SUITE_DIRECTORIES = {
    "montecarlo/transport": "transport",
    "montecarlo/collisions": "transport-collisions",
    "montecarlo/cascades": "transport-cascades",
    "montecarlo/runner": "transport-runner",
    "montecarlo/spectrum": "spectrum",
    "montecarlo/coherent": "spectrum-coherent",
    "montecarlo/brem": "spectrum-brem",
    "montecarlo/channeling": "channeling",
    "montecarlo/quantum": "quantum",
    "energy-grid": "energy-grid",
    "materials": "materials",
    "xsgen": "xsgen",
}

INTEGRATION_TESTS = (
    "notebooks/analysis_app/test_app.py",
    "cli/test_contract.py",
    "materials/test_materials_package.py",
    "montecarlo/test_exports.py",
    "remote/test_remote_command_successful.py",
    "results/test_exports.py",
    "scan/test_run.py",
    "scan/test_sweep.py",
)
PRIMARY_TEST_SUITE_NAMES = (
    "core",
    "packaging",
    "apps",
    "cli",
    "quantum",
    "channeling",
    "spectrum-coherent",
    "spectrum-brem",
    "spectrum",
    "transport-cascades",
    "transport-collisions",
    "transport-runner",
    "transport",
    "energy-grid",
    "materials",
    "xsgen",
)
# Modules holding ``identity_pin`` tests (#1316); the ``identity-pins`` suite runs
# only those tests (``-m identity_pin``). ``--update-pins`` re-mints every pin.
IDENTITY_PIN_TESTS = (
    "campaign/test_channeling_section.py",
    "detectors/test_acceptance.py",
    "materials/test_profiles.py",
    "montecarlo/runner/test_case.py",
    "scan/test_adaptive_api.py",
)
#: Cross-domain samples: they overlap the primary partition.
SAMPLED_TEST_SUITES = {"integration": INTEGRATION_TESTS, "identity-pins": IDENTITY_PIN_TESTS}
#: Marker expressions that narrow a sampled suite to its owned tests.
SUITE_MARKERS = {"identity-pins": "identity_pin"}
TEST_SUITE_NAMES = (*PRIMARY_TEST_SUITE_NAMES, *SAMPLED_TEST_SUITES)


def test_files_for_suite(name: str, root: Path = ROOT) -> list[Path]:
    """Return deterministic repository test paths for one documented suite.

    Primary domain suites partition every ``tests/**/test_*.py`` module once.
    ``core`` owns shared infrastructure and modules outside the named domains.
    ``integration`` intentionally samples public boundaries across domains, and
    ``identity-pins`` the modules holding identity pins; the full
    ``pyrite-dev test`` gate remains unchanged.
    """
    tests_dir = root / "tests"
    files = sorted(tests_dir.rglob("test_*.py"))
    if name in SAMPLED_TEST_SUITES:
        selected = [tests_dir / filename for filename in SAMPLED_TEST_SUITES[name]]
        missing = [path.name for path in selected if not path.is_file()]
        if missing:
            raise AgentToolingError(f"missing {name} tests: " + ", ".join(missing))
        return selected
    if name not in PRIMARY_TEST_SUITE_NAMES:
        raise AgentToolingError(f"unknown test suite {name!r}")

    return [path for path in files if test_suite_owner(path.relative_to(tests_dir)) == name]


def test_suite_owner(relative: str | Path) -> str:
    """Return the primary suite owning one ``tests/``-relative module path.

    Explicit patterns win (first match), then the deepest owning directory,
    then ``core``.
    """
    posix = Path(relative).as_posix()
    for candidate, patterns in TEST_SUITE_PATTERNS.items():
        if any(fnmatch.fnmatchcase(posix, pattern) for pattern in patterns):
            return candidate
    for parent in PurePosixPath(posix).parents:
        if (owner := TEST_SUITE_DIRECTORIES.get(parent.as_posix())) is not None:
            return owner
    return "core"
