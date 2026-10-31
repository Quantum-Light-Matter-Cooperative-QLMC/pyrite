"""Versioned JSONL evidence records emitted by standalone validation checks."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

SCHEMA_REVISION = 1
Verdict = Literal["pass", "fail", "skip"]

CHECK_LEDGER_IDS: Final[dict[str, tuple[str, ...]]] = {
    # Optimization comparison is a study diagnostic, not an independent physics claim.
    "checks/pxr_optimization.py": (),
    # #278 / #286 measurement diagnostics: they record grid error, accuracy vs electron
    # count and per-flush cost as study evidence, not as an independent physics claim.
    "checks/continuum_grid_convergence.py": (),
    "checks/electron_count_accuracy.py": (),
    "checks/line_cost_286.py": (),
    "checks/brem_source_comparison.py": ("brem-source-comparison",),
    "checks/sb85_eta_small_kappa_fit.py": ("bremslib-electron-electron",),
    "checks/coherent_flat_omission.py": ("coherent-flat-term-omission",),
    "checks/coherent_physical_population_remote.py": ("coherent-physical-bunch-population",),
    "checks/coherent_window_excluded_dispersion.py": ("coherent-line-grid-windowed-resolution",),
    "checks/coherent_transverse_coherence.py": ("transverse-bunch-form-factor",),
    "checks/collision_statistics_refinement.py": ("energy-controlled-propagation",),
    "checks/cross_reflection_coherence.py": ("cross-reflection-coherence",),
    "checks/abtem_channeling_oracle.py": ("cr-density-matrix-split-operator",),
    "checks/dans_diffraction_oracle.py": ("dans-diffraction-oracle",),
    "checks/kephart_occupation.py": ("cr-dechanneling-collapse",),
    "checks/channeling_handoff.py": ("cr-zone-handoff-accounting",),
    "checks/channeling_overflow.py": ("cr-planar-overflow-spectrum",),
    # Axial handoff research diagnostics are recorded in the ledger notes, not
    # as physical validation verdicts.
    "checks/axial_escape_accounting.py": (),
    "checks/axial_matching_study.py": (),
    # Hardware performance evidence accompanies the CUDA backend validation;
    # it does not add a separate pass/fail ledger claim.
    "checks/axial_mcwf_cuda.py": (),
    # First-flight CUDA parity/timing is backend evidence, not a physics verdict.
    "checks/photon_first_flight_cuda.py": (),
    "checks/axial_mcwf_precision_cuda.py": (),
    "checks/axial_recording_performance.py": (),
    "checks/axial_split_step_precision_cuda.py": (),
    "checks/axial_mcwf_return_flux.py": (),
    "checks/quantum_decoherence_budget.py": ("quantum-decoherence-budget",),
    "checks/channeling_material_survey.py": (),
    "checks/channeling_submev_screen.py": (),
    "checks/srd164_inner_shell_comparison.py": (),
    "checks/channeling_modulated_emission.py": ("channeling-modulated-incoherent-emission",),
    "checks/xraylib_cascade_oracle.py": ("characteristic-radiation",),
    "checks/detector_solid_angle_check.py": (
        "detector-eaglexo",
        "detector-line-broadening",
    ),
    "checks/elsepa_line_sensitivity.py": ("elsepa-elastic-sampling",),
    "checks/energy_loss_straggling_observables.py": ("energy-loss-straggling",),
    "checks/energy_step_convergence_matrix.py": ("energy-step-convergence",),
    "checks/feranchuk_check_script.py": ("closed-form-flux", "pxr-amplitude", "cbs-amplitude"),
    "checks/feranchuk_vs_zhai_check.py": ("closed-form-flux", "coherent-line-spectrum"),
    "checks/kinematic_validity_check.py": ("coherent-line-spectrum", "line-energy-dispersion"),
    "checks/line_window_backend_agreement.py": ("line-window-seeding",),
    "checks/mosaic_mc_check.py": ("mosaic-analytic", "mosaic-mc"),
    "checks/multilayer_check.py": ("multilayer-stack", "self-absorption"),
    "checks/multilayer_slice3_check.py": ("multilayer-stack", "self-absorption"),
    "checks/multilayer_validation_check.py": ("multilayer-stack", "self-absorption"),
    "checks/transition_radiation_omission.py": ("tr-detected-omission-bound",),
    "checks/optical_response_cuda.py": ("optical-material-response",),
    "checks/optical_transition_cuda.py": ("optical-planar-tr", "optical-ctr-population"),
    # Decision evidence for the pixel angular-reconstruction policy (#23); v1
    # makes no ledgered reconstruction-accuracy claim, so it emits no records.
    "checks/pixel_reconstruction_oracle.py": (),
    # Sizing evidence for pair conversion (#275): magnitudes, no pass/fail
    # anchor, so it emits no records.
    "checks/pair_conversion_yield.py": (),
    # Native-output/parser inventory (#276), not a transport-physics anchor.
    "checks/positron_table_inventory.py": (),
    # Dispatch/runtime evidence (#485), not a transport-physics anchor.
    "checks/physics_adapter_overhead.py": (),
    # Unit-boundary conversion timing, without a physics-validation claim.
    "checks/unit_conversion_overhead.py": (),
    # Distributed orchestration evidence (#681), not a physics-validation claim.
    "checks/dask_execution_smoke.py": (),
    # Checkpoint storage performance (#612), without a physics-validation claim.
    "checks/checkpoint_selective_memory.py": (),
    "checks/checkpoint_sweep_memory.py": (),
    # Viewer workload evidence for the trajectory backend decision (#205):
    # resource costs, not transport physics, so they emit no records.
    "checks/trajectory_paraview_probe.py": (),
    "checks/trajectory_trame_probe.py": (),
    "checks/trajectory_viewer_benchmark.py": (),
    "checks/cxr_reduction_benchmark.py": (),
    # Exact CUDA transport kernel build/variant timing (#796, #827): bitwise-
    # matched performance evidence, no physics-validation claim.
    "checks/transport_kernel_builds.py": (),
    "checks/transport_kernel_variants.py": (),
    # Experimental model comparison is descriptive, with no pass/fail verdict.
    "checks/inner_shell_experiment_comparison.py": (),
    "checks/shell_ionization_comparison.py": ("eedl-shell-ionization-comparison",),
    "checks/brem_angular_comparison.py": ("bremslib-angular-schiff",),
    "checks/radiation_error_estimator_calibration.py": ("radiation-error-estimators",),
    "checks/shell_secondary_transport_observables.py": ("shell-secondary-transport",),
    "checks/segment_escape_split_ladder.py": ("segment-escape-average",),
    "checks/shell_soft_hard_transport_observables.py": ("shell-soft-hard-transport",),
    "checks/soft_deflection_line_sensitivity.py": ("shell-soft-hard-transport",),
    "checks/soft_inelastic_deflection.py": ("shell-soft-hard-transport",),
    "checks/sinc_bin_integration.py": ("sinc-bin-integration",),
    "checks/substep_invariance.py": ("substep-radiation-invariance",),
    "checks/transport_core_goldens.py": ("electron-transport",),
    # Model sensitivity evidence (#339), without a measured-data pass/fail gate.
    "checks/zhai_hbn_energy_trend.py": (),
}
"""Maintained standalone-check to detailed-ledger claim mappings."""


@dataclass(frozen=True)
class ValidationRecord:
    """One measured, reproducible automated-evidence result for a ledger claim."""

    ledger_id: str
    measured_value: float | str | None
    reference_value: float | str | None
    tolerance: float | str | None
    verdict: Verdict
    revision: str
    recorded_at: str
    check: str
    schema_revision: int = SCHEMA_REVISION

    @classmethod
    def create(
        cls,
        *,
        ledger_id: str,
        measured_value: float | str | None,
        reference_value: float | str | None,
        tolerance: float | str | None,
        verdict: Verdict,
        revision: str,
        check: str,
        recorded_at: str | None = None,
    ) -> ValidationRecord:
        """Build a record, using an explicit UTC timestamp when needed."""
        return cls(
            ledger_id,
            measured_value,
            reference_value,
            tolerance,
            verdict,
            revision,
            recorded_at or datetime.now(UTC).isoformat(),
            check,
        )

    @classmethod
    def from_json(cls, value: dict[str, object]) -> ValidationRecord:
        """Validate and decode one persisted JSONL object."""
        expected = {
            "ledger_id",
            "measured_value",
            "reference_value",
            "tolerance",
            "verdict",
            "revision",
            "recorded_at",
            "check",
            "schema_revision",
        }
        if set(value) != expected or value["schema_revision"] != SCHEMA_REVISION:
            raise ValueError("unsupported validation record schema")
        if value["verdict"] not in {"pass", "fail", "skip"}:
            raise ValueError("invalid validation record verdict")
        if not all(
            isinstance(value[field], str) and value[field]
            for field in ("ledger_id", "revision", "recorded_at", "check")
        ):
            raise ValueError("validation record requires non-empty identity fields")
        scalar_fields = ("measured_value", "reference_value", "tolerance")
        if not all(
            value[field] is None or isinstance(value[field], (float, int, str))
            for field in scalar_fields
        ):
            raise ValueError("validation record has invalid measured fields")
        return cls(
            ledger_id=cast(str, value["ledger_id"]),
            measured_value=cast(float | str | None, value["measured_value"]),
            reference_value=cast(float | str | None, value["reference_value"]),
            tolerance=cast(float | str | None, value["tolerance"]),
            verdict=cast(Verdict, value["verdict"]),
            revision=cast(str, value["revision"]),
            recorded_at=cast(str, value["recorded_at"]),
            check=cast(str, value["check"]),
            schema_revision=cast(int, value["schema_revision"]),
        )


def write_records(path: Path, records: list[ValidationRecord]) -> None:
    """Replace a check's prior evidence atomically with deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(
        "".join(json.dumps(asdict(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    staged.replace(path)


def records_for_exit(check: str, exit_code: int, revision: str) -> list[ValidationRecord]:
    """Convert a standalone check process result into records for its claims."""
    try:
        ledger_ids = CHECK_LEDGER_IDS[check]
    except KeyError as exc:
        raise ValueError(f"no ledger mapping registered for {check}") from exc
    # Exit 2 is the conventional explicit skip used by standalone checks. Exit
    # 75 (EX_TEMPFAIL) means required remote work or cached evidence is not yet
    # available, so no claim was evaluated and a failure would be misleading.
    verdict: Verdict = "pass" if exit_code == 0 else "skip" if exit_code in {2, 75} else "fail"
    return [
        ValidationRecord.create(
            ledger_id=ledger_id,
            measured_value=None,
            reference_value=None,
            tolerance=None,
            verdict=verdict,
            revision=revision,
            check=check,
        )
        for ledger_id in ledger_ids
    ]


SYNC_STAMP_NAME: Final = ".pyrite-sync"
"""Code-identity stamp ``pyrite remote sync`` leaves in a synced checkout root."""


def _stamped_revision(root: Path) -> str | None:
    """Return the revision recorded by the last code sync into ``root``.

    ``pyrite remote sync`` unpacks an exported tree, not a repository, so on the
    box this stamp is the only revision evidence there is. A dirty payload is
    reported as ``<revision>-dirty``: the working tree it came from is not
    reproducible from that revision alone, and evidence must say so. A stamp
    whose own revision is unknown yields the payload digest instead, which at
    least identifies the exact code that ran.
    """
    try:
        lines = (root / SYNC_STAMP_NAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition(": ")
        if separator:
            fields[key.strip()] = value.strip()
    revision = fields.get("code_revision", "")
    digest = fields.get("code_digest", "")
    if revision and revision != "unknown":
        return f"{revision}-dirty" if fields.get("code_dirty") == "True" else revision
    return f"synced-{digest}" if digest else None


def _revision_for(root: Path) -> str:
    """Return ``root``'s git revision, or the revision its last code sync stamped.

    The git revision wins whenever there is one: a developer checkout can carry
    a stale stamp from a sync it made earlier.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stamped = _stamped_revision(root)
        if stamped is None:
            raise RuntimeError(
                f"cannot determine source revision: {root} is not a git checkout and "
                f"carries no {SYNC_STAMP_NAME} stamp from pyrite remote sync"
            ) from exc
        return stamped
    return result.stdout.strip()


def current_revision() -> str:
    """Return the checked-out source revision used to produce an evidence record.

    Falls back to the synced code stamp when the tree is not a git checkout, so
    evidence collected on the remote box carries a revision instead of crashing
    on ``git rev-parse``.
    """
    return _revision_for(Path(__file__).parents[3])


def read_records(root: Path) -> tuple[ValidationRecord, ...]:
    """Read every JSONL check-evidence file in deterministic path order."""
    records: list[ValidationRecord] = []
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(ValidationRecord.from_json(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"malformed validation record {path}:{line_number}") from exc
    return tuple(records)
