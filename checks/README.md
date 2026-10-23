# Developer physics checks

Standalone, potentially slow validation anchors. These are not application
entry points and are not part of the fast unit-test suites. The validation
ledger remains authoritative; a successful check is evidence, not human
sign-off.

The canonical check-to-ledger mapping is `pyrite.validation.check_records.CHECK_LEDGER_IDS`.
Use the collector to preserve normal output and write dated JSONL evidence:

```bash
pyrite-dev validation-records -- checks/dans_diffraction_oracle.py
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
| `optical_transition_cuda.py` | Real CUDA parity of optical-band TR/CTR: per-history power and complex bunch field before squaring (fused kernel and CuPy array route) and the optical scorer's self/pair/per-electron yields for importance-weighted sampled and Gaussian laws, on a Johnson–Christy/Rakić Au multilayer. Stage a clean tree with `pyrite remote sync`, run in a bounded GPU allocation, collect JSON evidence. No CPU fallback. |
| `optical_response_cuda.py` | Real CUDA complex128 upload and planar optical TR parity for the pinned CC0 Au response; unsupported-band/component refusal. Stage with `pyrite remote sync`, run in a bounded GPU allocation, and collect JSON evidence. No CPU fallback. |
| `channeling_overflow.py` | Experimental full-depth planar momentum instrument (#522): nonwrapped differential gain, unresolved probability, in-band gain shift and closure-gated conditional angle samples. Fixed-energy research diagnostic; no production handoff or derived escape boundary. |
| `axial_escape_accounting.py` | Bounded deterministic Si TDS plane-wave comparison: a position-diagonal sink cannot identify outgoing Fourier-window escape for different incident momenta (#548). Research only; no physical escape model or validation verdict. |
| `axial_matching_study.py` | Research study of an energy-defined retained projector and return-flux loss, with Gamma-point and Bloch-grid TDS rates (#548). No dechanneling prediction or physical validation verdict. |
| `axial_mcwf_cuda.py` | Bounded remote-GPU benchmark of matched CPU/CUDA axial MCWF runs, recording cold and warm timing, setup, transfer and memory evidence; `--workload` adds heavier supercell cases, comparing CPU/CUDA throughput per electron-step when `--cpu-max-electrons` caps the CPU run. It is performance evidence for the CUDA backend, not a separate physics-validation verdict. |
| `axial_mcwf_precision_cuda.py` | Remote-GPU paired complex64 versus complex128 axial MCWF ensembles (same Philox streams; jump kernels float64) with a predeclared tolerance against Monte Carlo standard error, jump-sequence agreement and timing. Research measurement only; production axial MCWF stays complex128. |
| `axial_split_step_precision_cuda.py` | Remote-GPU complex64 versus complex128 tolerance and throughput of the deterministic axial no-jump split step (norm/jump-clock error, infidelity, density L1 versus depth). Research measurement only; production axial MCWF stays complex128. |
| `axial_mcwf_return_flux.py` | MCWF study of return-flux deficit under an energy-cut handoff, including TDS and electronic jumps (#548); optional `--loss-coupling` records cut/clearance and cross-cut loss histograms (#623). Research only; no native launch model or physical validation verdict. |
| `channeling_material_survey.py` | Catalog-wide thermal, shell-oscillator, plane-incidence and static-profile availability inventory (#525). JSON report only; no physics-validation verdict or ledger evidence record. |
| `trajectory_viewer_benchmark.py` | Tracks-only Plotly/PyVista loading, construction, camera screenshot, filtering and memory probe on identical saved captures. Replication diagnoses resource scaling, not transport physics; no ledger records. |
| `physics_adapter_overhead.py` | Bounded default-physics CPU phase benchmark across matched source revisions; separate startup/warm timings, source keys, output hashes and RSS. No transport-validity claim; see [overhead evidence](../docs/repo-design/compute/physics-adapter-overhead.md). |
| `full_track_bremslib/` | Pinned Geant4 TestEm5 and exact PyRITE CPU full-track comparison for W and Si at 300 keV, including cutoff sweep, raw outputs, and unresolved W transport mismatch (#182); production-stack primary T/R vs Geant4 DPWA single scattering (#183, #317). |
| `brem_source_comparison.py` | EEDL and BremsLib bremsstrahlung `chi`, hard cross section, and radiative moment against the pinned Seltzer–Berger tables for every catalogue element, 1 keV–30 MeV; gates BremsLib, reports the EEDL interpolation defect (#174). |
| `sb85_eta_small_kappa_fit.py` | Refits the small-`k/T` scale of the packaged Seltzer–Berger (1985) electron–electron η table against the pinned SB digital tables and prints per-element fits, spread and the stored Z-averaged value (#439). |
| `coherent_flat_omission.py` | Paired spectrum timing of the coherent Flat-term omission on one immutable transport realization; thin anchors run locally, thick/GPU cases through `pyrite remote` (#362). |
| `coherent_physical_population_remote.py` | Charge-weighted CUDA float64 limits, same-trajectory 30/60 keV grid ladders, and reduced short-bunch smoke; persists transport across scheduler slices. |
| `coherent_window_excluded_dispersion.py` | Per-row window-excluded production power against the dispersive window certificate on one transport; `--case anchor` on CPU, `--case thick` (1 mm) on `pyrite remote`. |
| `coherent_transverse_coherence.py` | Tests whether a single coherent transverse-direction draw is representative of the observable spectrum. |
| `transition_radiation_omission.py` | Exact planar-stack transition radiation of every boundary crossing, bounded over the detector acceptance, against the production bremsstrahlung density for tens-of-keV and MeV scenes; per-band omission bias against `target_rse / 3` (#102). Small CPU runs. |
| `collision_statistics_refinement.py` | Measures collision-statistics changes under transport substep refinement. |
| `cross_reflection_coherence.py` | Bounds the coherent-spectrum effect of omitted cross-reflection terms. |
| `abtem_channeling_oracle.py` | abTEM frozen-phonon multislice of oriented Si (beam `[1 −1 n]` in the (110) planes; Einstein displacements), projected on PyRITE Bloch states. Compares `P_n(z)` with the TDS-only density solver and gates on `--tolerance`. Runs on CPU or GPU (`--device gpu`, CuPy) (#504; `oracle` group). |
| `dans_diffraction_oracle.py` | Pinned external lattice, reciprocal-geometry, and structure-factor comparison. |
| `kephart_occupation.py` | Kephart et al. (1989) Si occupation lengths from the density solver, through the forward-detector observable with bound-weighted y-spreading. Sweeps cutoff, band, Bloch grid and depth step (remote). `--critical-angle-multiples` and `--gap-fractions` add exploratory boundaries with explicit spectral metadata; see [boundary investigation](../docs/research/physics/dechanneling-boundary.md) (#522). `--refit` re-fits the stored runs in `docs/validation/check-records/kephart_occupation_runs/` at the declared 300 eV cutoff and gates on `--tolerance` and `--outlier` (#504). |
| `channeling_handoff.py` | Planar channeling handoff (#506) at the production grid (Si(110), 5 MeV, `H = 16`, 16 cells, 20 um): total-variation distance of the `E_perp` free-flux marginal per 5 um chunk from the whole-depth marginal (the separable-sampling assumption, gated on `--tolerance`), overflow share and mean free handoff angle in units of `psi_c`. About 10 s on CPU. |
| `quantum_decoherence_budget.py` | Research (#501): cell-averaged transverse decoherence rate `Γ(Δx)` of the #504 jump operators for Si(110) and diamond(110) at 30 keV–30 MeV (diamond electronic through a graphite-oscillator surrogate), against CR and PXR/CBS formation lengths; reports HOPG/vdW cells as not computable. Gates `Γ(0) = 0`, TDS saturation and `1/v²` scaling. Under a second. |
| `channeling_modulated_emission.py` | Research prototype (#501): channeling-modulated local incoherent emission factor `F(z)` and depth-averaged yield `Y(L)` for aligned/tilted plane waves and d-periodic shaped combs in Si(110) at 0.3/3/17 MeV, from the density solver. Gates the plane-wave and converter limits. About 1.5 min on CPU. |
| `detector_solid_angle_check.py` | Solid-angle integration and analytic aperture-width comparison; the integrated-spectrum route is not separately ledgered. |
| `zhai_hbn_energy_trend.py` | Remote-only S5b h-BN sensitivity diagnostic: Friedel pairing, escape, basal reflections, Debye–Waller factors, bin quadrature, and detector response stages. Reports model factors; approximate paper peaks are not regression targets. |
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
| `pair_production_geant4/` | Pair-conversion e⁻/e⁺ energy share and polar angles from `sample_pair` against Geant4 11.4.2 TestEm5 (`empenelope`, `emstandard_opt0`, `emstandard_opt4`) for C and Pb at 2 and 5 MeV, from committed binned references (#275). |
| `pair_conversion_yield.py` | Expected pair conversions per primary for Si and MoS2 slabs at 3–5 MeV under coupled BremsLib transport: hard photons above 2mₑc², their expected first-interaction pair probability, and pairs per primary (#275); sizing evidence only, emits no records. |
| `positron_table_inventory.py` | Real ELSEPA positron free-atom Si/W and muffin-tin Si outputs, and SBETHE positron Si/W outputs, against the native parsers and stopping loader (#276). Generator-format inventory only; no Monte Carlo or transport-validation claim. `--out` retains decks, native outputs, source hashes, and compiler provenance. |
| `pixel_reconstruction_oracle.py` | Nearest-tile pixel angular reconstruction against direct per-pixel evaluation from one transport, across angular shapes and representative pixels (#23); decision evidence only, no ledger claim, emits no records. |
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
| `feranchuk_analysis.ipynb`, `feranchuk_analysis.md` | Legacy paired notebook retained for historical Feranchuk analysis; it emits no records. Keep output-free. |

Run commands are documented in each check. Heavy Monte Carlo or GPU work must
use `pyrite remote`; do not launch it locally from this directory.

Optional viewer evaluation probes: `trajectory_paraview_probe.py` (pvpython), `trajectory_trame_probe.py` (live server smoke without a client). Raw repeated results: `trajectory_viewer_results.json`; interpretation and limitations: [trajectory scenes](../docs/repo-design/storage/trajectory-scenes.md).

To diagnose an earlier physical-population run with updated code, pass
`--transport-record OLD.json --out NEW.json` to
`coherent_physical_population_remote.py`. The adjacent `OLD.transport.pkl`
must be a trusted artifact from that run. Replay checks the original code/table
provenance, inputs, segment fingerprint, population and axis against the JSON
record. Current tables must match the original digest. It records hashes of both
source files, preserves their contents, and evaluates the saved trajectories
without generating a new transport. Use a distinct output and retain the source
argument on every resumed slice. Run through the remote GPU workflow.
