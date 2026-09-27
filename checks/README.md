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
| `full_track_bremslib/` | Pinned Geant4 TestEm5 and exact PyRITE CPU full-track comparison for W and Si at 300 keV, including cutoff sweep, raw outputs, and unresolved W transport mismatch (#182). |
| `brem_source_comparison.py` | EEDL and BremsLib bremsstrahlung `chi`, hard cross section, and radiative moment against the pinned Seltzer–Berger tables for every catalogue element, 1 keV–30 MeV; gates BremsLib, reports the EEDL interpolation defect (#174). |
| `coherent_transverse_coherence.py` | Tests whether a single coherent transverse-direction draw is representative of the observable spectrum. |
| `collision_statistics_refinement.py` | Measures collision-statistics changes under transport substep refinement. |
| `cross_reflection_coherence.py` | Bounds the coherent-spectrum effect of omitted cross-reflection terms. |
| `dans_diffraction_oracle.py` | Pinned external lattice, reciprocal-geometry, and structure-factor comparison. |
| `detector_solid_angle_check.py` | Solid-angle integration and analytic aperture-width comparison; the integrated-spectrum route is not separately ledgered. |
| `elsepa_line_sensitivity.py` | PXR/CBS line, bremsstrahlung, characteristic and endpoint response of Si, HOPG and MoS2 cases to the default ELSEPA elastic model against the previous Mott model, unpaired over seeds (GPU; remote only). |
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
| `shell_ionization_comparison.py` | Pinned Bote–Salvat `xion.f` transcription, local EEDL shell interpolation, and characteristic-production ratios for the 24 catalogue elements. |
| `brem_angular_comparison.py` | BremsLib angular shape (enclosed-flux angles) against the Schiff formula, Koch–Motz 2BS, for the 24 catalogue elements at 5–30 MeV. |
| `shell_soft_hard_transport_observables.py` | Opt-in shell soft/hard inelastic transport against continuous stopping on Si/SiO2/MoS2 at 5/20/100 keV: stopping closure along trajectories, straggling, transmission, backscatter, range, `W_c` convergence, energy conservation, event contract. |
| `shell_secondary_transport_observables.py` | Secondary transport (#94) on Si/MoS2 slabs at 20/100 keV: threshold convergence of energy backscatter/transmission, depth dose, characteristic and bremsstrahlung yields; energy balance, event contract per track, and the generation-1 launch spectrum against the free-electron Moller DCS. |
| `segment_escape_split_ladder.py` | Split-segment ladder (k = 1, 8, 32 collinear pieces) for HOPG C K, bremsstrahlung and PXR/CBS yields: characteristic and bremsstrahlung escape must be invariant; the transparent-escape control isolates the line routes' midpoint escape bias (GPU; remote only). |
| `soft_deflection_line_sensitivity.py` | Paired line-spectrum response of a Si shell-mode transport to emulated soft inelastic direction wander (per row) and extra angular diffusion (per vertex), against seed-to-seed Monte Carlo error (GPU; remote only). |
| `soft_inelastic_deflection.py` | Soft and hard inelastic angular transport rates of the closed shell model against the elastic Mott transport rate, and per-row soft deflection, on Si/SiO2/MoS2 at 5–100 keV. |
| `sinc_bin_integration.py` | Bin-mean line quadrature on one transport: yield against an exact node reference, and CPU/CuPy/fused-CUDA agreement (remote only at 300 keV). |
| `substep_invariance.py` | Measures emitted-radiation invariance under numerical transport substepping. |
| `transport_core_goldens.py` | Verifies bit-for-bit CPU transport-core golden outputs. |
| `xraylib_cascade_oracle.py` | Pinned xraylib Kissel full-cascade comparison of EADL vacancy propagation and K/L/M line cross sections for Si, Cu and Au from shared photoionization primaries; gates vacancy enhancement and K-alpha/Au L3 lines (#91). |
| `cxr_analysis_feranchuk.ipynb`, `cxr_analysis_feranchuk.md` | Legacy paired notebook retained for historical Feranchuk analysis; it emits no records. Keep output-free. |

Run commands are documented in each check. Heavy Monte Carlo or GPU work must
use `pyrite remote`; do not launch it locally from this directory.
