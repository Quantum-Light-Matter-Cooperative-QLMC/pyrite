# Developer physics checks

Standalone, potentially slow validation anchors. These are not application
entry points and are not part of the fast unit-test suites. The validation
ledger remains authoritative; a successful check is evidence, not human
sign-off.

The canonical check-to-ledger mapping is `pyrite.validation.check_records.CHECK_LEDGER_IDS`.
Use the collector to preserve normal output and write dated JSONL evidence:

```bash
uv run pyrite-dev validation-records -- checks/dans_diffraction_oracle.py
```

On remote workers, use the collector's `--output-dir`; copy reviewed records into
`docs/validation/check-records/` before regenerating the status summary. The collector
marks exit code 0 as pass, 2 (explicit skip) or 75 (missing remote/cache
prerequisite) as skip, and any other nonzero exit as fail.

Records carry the git revision but not the compute backend. Several anchors are
host-only -- flight-grouped incoherent CXR has no device port -- so run the
collector with `PYRITE_MC_BACKEND=cpu` unless a check documents otherwise; a
CUDA-pinned shell records a spurious `fail` for those.

`transport_core_goldens.py` has no committed baseline record. It compares
against a golden directory captured from an earlier revision (`--dir`), which is
a local working artifact, so capture-then-check on one revision is circular and
its `electron-transport` claim stays `missing` until a reviewed golden set is
run through the collector.

| Artifact | Purpose |
|---|---|
| `coherent_transverse_coherence.py` | Tests whether a single coherent transverse-direction draw is representative of the observable spectrum. |
| `collision_statistics_refinement.py` | Measures collision-statistics changes under transport substep refinement. |
| `cross_reflection_coherence.py` | Bounds the coherent-spectrum effect of omitted cross-reflection terms. |
| `dans_diffraction_oracle.py` | Pinned external lattice, reciprocal-geometry, and structure-factor comparison. |
| `detector_solid_angle_check.py` | Solid-angle integration and analytic aperture-width comparison; the integrated-spectrum route is not separately ledgered. |
| `energy_loss_straggling_observables.py` | Paired-seed phase, terminal-fraction, stopped-range, bremsstrahlung, and coherent-line response to Urban straggling. |
| `energy_step_convergence_matrix.py` | Measures energy-controlled transport convergence across the maintained refinement matrix. |
| `feranchuk_check_script.py` | Legacy Feranchuk–Spence LiF absolute-flux anchor. |
| `feranchuk_vs_zhai_check.py` | Analytic-versus-transport comparison through the maintained Zhai anchor pipeline. |
| `kinematic_validity_check.py` | Kinematic-approximation audit; its validity parameters are supporting diagnostics, not separate ledger claims. |
| `line_window_backend_agreement.py` | CUDA versus CPU line-route agreement on a production window plan, float32 and FP64, on one pickled transport (#101). |
| `mosaic_mc_check.py` | Perfect-crystal limit, quadrature convergence, broadening, and yield checks. |
| `multilayer_check.py` | First-slice stack transport and front/back escape checks. |
| `multilayer_slice3_check.py` | Per-layer crystalline-radiation and incoherent-sum checks. |
| `multilayer_validation_check.py` | Closed-form stack attenuation and depth-range scaling anchors. |
| `radiation_error_estimator_calibration.py` | Calibrates warning thresholds for the radiation error estimator. |
| `shell_soft_hard_transport_observables.py` | Opt-in shell soft/hard inelastic transport against continuous stopping on Si/SiO2/MoS2 at 5/20/100 keV: stopping closure along trajectories, straggling, transmission, backscatter, range, `W_c` convergence, energy conservation, event contract. |
| `sinc_bin_integration.py` | Bin-mean line quadrature on one transport: yield against an exact node reference, and CPU/CuPy/fused-CUDA agreement (remote only at 300 keV). |
| `substep_invariance.py` | Measures emitted-radiation invariance under numerical transport substepping. |
| `transport_core_goldens.py` | Verifies bit-for-bit CPU transport-core golden outputs. |
| `cxr_analysis_feranchuk.ipynb`, `cxr_analysis_feranchuk.md` | Legacy paired notebook retained for historical Feranchuk analysis; it emits no records. Keep output-free. |

Run commands are documented in each check. Heavy Monte Carlo or GPU work must
use `pyrite remote`; do not launch it locally from this directory.
