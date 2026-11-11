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
    # #610 capture-policy benchmark: engineering evidence, not a physics claim.
    "checks/pyrite_issue610_overlap_benchmark.py": (),
    # #278 / #286 measurement diagnostics: they record grid error, accuracy vs electron
    # count and per-flush cost as study evidence, not as an independent physics claim.
    # Continuum benchmark diagnostics: engineering evidence, no physics verdict.
    "checks/continuum_backend_parity.py": (),
    "checks/continuum_benchmark_report.py": (),
    "checks/continuum_geant4_precision.py": (),
    "checks/continuum_primary_tail.py": (),
    "checks/continuum_pyrite_precision.py": (),
    "checks/continuum_spectral_statistics.py": (),
    "checks/continuum_grid_convergence.py": (),
    "checks/electron_count_accuracy.py": (),
    "checks/line_cost_286.py": (),
    "checks/line_grid_candidates_286.py": (),
    "checks/line_policy_286.py": (),
    "checks/line_setup_support_286.py": ("line-tabulation-nan-ceiling",),
    "checks/line_crossover_286.py": (),
    "checks/continuum_grid_end_to_end.py": (),
    "checks/detector_cone_variance_reduction.py": (),
    # #879 fixed-population spectral/timing diagnostics, not a physics claim.
    "checks/line_roulette_spectral_gate_879.py": (),
    "checks/line_roulette_independent_879.py": (),
    "checks/roulette879_analysis.py": (),
    "checks/line_roulette_fom_879.py": (),
    "checks/line_roulette_blocks_879.py": (),
    "checks/line_roulette_core_bench_879.py": (),
    "checks/line_roulette_mass_jitter_879.py": (),
    "checks/line_roulette_off_hashes_879.py": (),
    "checks/line_roulette_tail_879.py": (),
    # #875 launch-shape timing diagnostic: bitwise-identity and timing, no physics claim.
    "checks/launch_retune_875.py": (),
    "checks/brem_source_comparison.py": ("brem-source-comparison",),
    # Fused band scoring diagnostics (#938): performance/parity evidence only.
    "checks/brem_primary_band_benchmark.py": (),
    "checks/brem_band_request_confirmation.py": (),
    "checks/brem_band_kernel_diagnostic.py": (),
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
    # #1055 ownership measurement: records ratios, without a pass/fail gate.
    "checks/planar_zone_row_cbs.py": (),
    # Axial handoff research diagnostics are recorded in the ledger notes, not
    # as physical validation verdicts.
    "checks/axial_escape_accounting.py": (),
    "checks/axial_matching_study.py": (),
    "checks/axial_fiber_andersen.py": ("cr-axial-fiber-regression",),
    "checks/axial_fiber_angular_map.py": ("cr-axial-fiber-angular",),
    "checks/axial_fiber_source.py": ("cr-axial-fiber-regression",),
    "checks/axial_outgoing_convergence.py": ("cr-axial-fiber-return-deficit",),
    "checks/axial_outgoing_maps.py": ("cr-axial-zone-handoff-accounting",),
    "checks/axial_depth_calibration.py": ("cr-axial-depth-calibration",),
    "checks/axial_fiber_fates.py": ("cr-axial-zone-handoff-physical",),
    # #585 static screen: a screening record, not a support verdict.
    "checks/channeling_submev_axial_screen.py": (),
    "checks/channeling_submev_w001_entry.py": (),
    "checks/channeling_submev_w001_kinematics.py": (),
    # Hardware performance evidence accompanies the CUDA backend validation;
    # it does not add a separate pass/fail ledger claim.
    "checks/axial_mcwf_cuda.py": (),
    "checks/axial_energy_mcwf_cuda.py": (),
    # #344: scheduled orchestration and CPU/CUDA parity are engineering checks,
    # not independent physics evidence eligible for claim promotion.
    "checks/refresh_validation_evidence.py": (),
    "checks/scheduled_cuda_parity.py": (),
    # First-flight CUDA parity/timing is backend evidence, not a physics verdict.
    "checks/photon_first_flight_cuda.py": (),
    # Live streamed scoring memory/wall benchmark (#611): engineering evidence.
    "checks/streamed_scoring_memory.py": (),
    # Bin-mean reducer profile and A/B replay (#1003, #286): engineering evidence.
    "checks/cxr_bin_mean_profile_1003.py": (),
    "checks/line_sparse_near_286.py": (),
    # Characteristic vs continuum budget evaluation (#1006): engineering evidence.
    "checks/characteristic_budget_1006.py": (),
    # Streamed-block sizing sweep (#907): engineering evidence.
    "checks/streamed_block_sizing.py": (),
    # Characteristic scoring setup/accumulate split (#1005): engineering evidence.
    "checks/characteristic_scoring_1005.py": (),
    "checks/characteristic_nsys_1005.py": (),
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
    "checks/longitudinal_coherence_regimes.py": (
        "longitudinal-quantum-bunching",
        "cr-recoil-line-energy",
    ),
    "checks/mosaic_mc_check.py": ("mosaic-analytic", "mosaic-mc"),
    "checks/multilayer_check.py": ("multilayer-stack", "self-absorption"),
    "checks/multilayer_slice3_check.py": ("multilayer-stack", "self-absorption"),
    "checks/multilayer_validation_check.py": ("multilayer-stack", "self-absorption"),
    "checks/transition_radiation_omission.py": ("tr-detected-omission-bound",),
    "checks/optical_response_cuda.py": ("optical-material-response",),
    "checks/optical_stack_cuda.py": ("optical-stack-fast-solver",),
    "checks/optical_transition_cuda.py": ("optical-planar-tr", "optical-ctr-population"),
    # #616 measured slit-DR screen: detector geometry is fitted, so no claim.
    "checks/castellano_slit_dr.py": (),
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
