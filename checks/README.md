# Developer physics checks

Standalone, potentially slow validation anchors. These are not application
entry points and are not part of the fast unit-test suites. The validation
ledger remains authoritative; a successful check is evidence, not human
sign-off.

The canonical check-to-ledger mapping is `pyrite.validation.check_records.CHECK_LEDGER_IDS`.
The `refresh_validation_evidence.py` driver runs a small scheduled CPU manifest
of explicit pass/fail checks (not report-only diagnostics),
collects isolated JSONL records, and writes a preview summary + manifest for review.
The `scheduled_cuda_parity.py` probe compares line, bremsstrahlung, characteristic,
and coherent spectra on three identical seeded segment records, across CPU FP64,
CUDA FP64 and CUDA FP32. Run it via `scheduled_cuda_parity.sbatch` on the lab box.
The `schedule_cuda_parity.sh` cron entry point guards recurring lab-host
submissions and records their job IDs. The orchestration and parity probes have
no independent ledger-claim mapping.
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
| `mixed_elastic/` | Paired equivalence, cost, and Geant4 comparison for mixed elastic transport (#894); see its README for the frozen protocol and run files. |
| `cherenkov_geant4/` | Geant4 11.4.2 `G4Cerenkov` reference for bulk Cherenkov yield (#587): minimal Geant4 app, `compare.py`, sbatch and pinned `reference/` outputs. Needs a Geant4 build; independent reference, does not sign off the `frank-tamm-homogeneous-dielectric` claim. |
| `equal_precision/` | Equal-precision PyRITE versus Geant4/PENRED comparator pilots, analyses and sbatch scripts (#510 and follow-ups); `manifest.json` freezes the scenes. Needs Geant4 and lab GPU; engineering evidence, no physics-validation claim. |
| `geant4_performance/` | Matched-bounds Geant4 single-scattering versus PyRITE CUDA benchmark (#821 and follow-ups). Needs Geant4 and a GPU; performance evidence, no physics-validation claim. |
| `mixed_elastic/` | Paired equivalence of `transport_mode="mixed"` against detailed ELSEPA on the #821 anchors (#894): planning pilot, frozen `protocol.json`, resumable held-out runner and #838 analysis, plus Geant4 WentzelVI and Goudsmit-Saunderson arms. Remote GPU and Geant4; results on the [equivalence page](../docs/validation/beam-transport/mixed-elastic-equivalence.md). |
| `photon_cuda/` | `benchmark.py` CUDA scaling/profiling/tuning workload and pinned RTX 5080 JSON records for the photon first-flight, production and charged re-entry stages (#341). Remote GPU only; cited by the photon re-entry validation page. |
| `rocm/` | Vendor-neutral GPU backend evaluation (#888): `evaluate.sbatch` (build, CuPy smoke, hardware suite, warm-transport bench), `cupy_smoke.py`, `bench_table.py`, runner scripts and `env.sh`. Remote AMD/NVIDIA GPU only; no physics-validation claim. |
| `reference/` | Pinned external reference data (NIST ESTAR stopping/range tables in `reference/estar/`) consumed by checks; not runnable. |

| `*.sbatch` (top level) | Remote SLURM wrappers for the same-stem or named check: `continuum_grid_convergence.sbatch`, `electron_count_accuracy.sbatch`, `line_cost_286.sbatch`, `line_crossover_286.sbatch` (#278/#286); `detector_cone_203_amd.sbatch` (`detector_cone_variance_reduction.py`); `cxr_launch825.sbatch` (`cxr_reduction_benchmark.py`, #825); `launch_retune_875.sbatch` (`launch_retune_875.py`); `streamed_completion_864.sbatch`, `streamed_split_872.sbatch`, `streamed_scoring_profile.sbatch` (`streamed_scoring_memory.py`, #864/#872/#611); `streamed_block_sizing.sbatch`, `streamed_block_sizing_ncu.sbatch` (`streamed_block_sizing.py`, #907); `characteristic_budget_1006.sbatch` (`characteristic_budget_1006.py` characteristic vs continuum convergence, cost and nested-prefix probes, #1006); `characteristic_scoring_1005.sbatch`, `characteristic_fine_grid_1005.sbatch` (`characteristic_scoring_1005.py` characteristic setup/scoring split on the #872 harness, `characteristic_nsys_1005.py` launches/copies per NVTX range, #1005); `tutorial_capture_cuda.sbatch` (regenerates one CUDA tutorial capture, #776). Submit from a synced checkout root; engineering evidence. |
| `pyrite_issue610_overlap_benchmark.py` | Trajectory-capture policy and write-overlap benchmark on a CUDA host (#610); results and reproduction commands live in `docs/repo-design/storage/trajectory-artifacts.md`. Engineering evidence, no physics-validation claim. |
| `slow_test_audit/` | Linux per-test/process-tree RSS sampling, duration summaries and axial eigensolve profiling for the remote CPU test audit (#646). See its README for environment isolation and reproduction; no physics-validation claim. |
| `dask_execution_smoke.py` | Fixed-seed local/Dask/native-SLURM CPU and isolated CUDA smoke (#681): numerical parity, shared artifacts, coordinator checkpoint recovery and resume. See the distributed-execution guide for site setup; engineering evidence, with no separate physics-validation claim. |
| `checkpoint_sweep_memory.py` | Campaign-shaped retained versus disk-backed sweep storage benchmark (#612), using synthetic spectra, fresh processes/directories, phase timing, peak RSS, and I/O counters. Run `uv run python checks/checkpoint_sweep_memory.py`; no physics-validation claim. |
| `streamed_block_sizing.py` | Streamed-block sizing (#907): one arm per process; continuation population, per-launch CUDA-event kernel time, host peak RSS, driver/pool VRAM peaks and spectra for joined, streamed and transport-only arms; optional synchronized scoring/transport split timers (`PROBE_SPLIT=1`) and cProfile (`PROBE_PROFILE=1`). Needs a CUDA device; engineering evidence, no physics-validation claim. |
| `streamed_scoring_memory.py` | Joined versus live streamed spectrum scoring (#611): one arm per process, host peak RSS, CuPy pool reserved/used peaks, synchronized phase timing, segment bytes moved, and spectrum agreement via `compare`. Run on the GPU box in a bounded allocation; engineering evidence, no physics-validation claim. |
| `line_roulette_spectral_gate_879.py` | Bounded remote-CUDA fixed-population CXR line roulette diagnostic (#879): freeze the full production line grid, capture per-primary spectra, and compare conditional full-bin/band errors with actual batched subset timings. Separate the continuum log grid and global-FP64 route; preserve original IDs/normalization and charge pilot cost once. Outputs stay in ignored scratch; no population-convergence or physics-validation claim. |
| `line_roulette_independent_879.py` | Frozen seed-0 predictor applied unchanged to independent seeds (#879): does it retain unseen rare line-dominant histories? Reduced per-primary masses stay in ignored scratch; no timing, deployment or physics-validation claim. |
| `line_roulette_fom_879.py`, `line_roulette_fom_879.sbatch` | Per-bin RSE of the resolved line spectrum at equal wall time against analog (#879), on one frozen grid, with roulette electron counts fitted from timed calibration runs and one `run_case` per seed and arm in rotated order. Remote CUDA only; measurements stay in ignored scratch. No physics-validation claim. |
| `line_roulette_blocks_879.py` | One large roulette run against one analog run at fitted equal wall time (#879), both streamed on one frozen grid; each block's line spectrum is saved, and per-bin RSE comes from batch means over blocks. Run through `line_roulette_fom_879.sbatch` with `CHECK=checks/line_roulette_blocks_879.py` (SLURM 4098, CUDA) or on the CPU partition with `--core per-electron` (SLURM 4111). Measurements stay in ignored scratch. Its same-primary agreement results are cited as remote evidence in the `line-scoring-roulette` ledger row; the automated record maps no ledger id. |
| `line_roulette_core_bench_879.py`, `line_roulette_core_bench_879.sbatch` | Transport throughput by core, block, schedule and precision on one MeV case, whole-request cost by core and precision, and roulette repeatability and streaming agreement in that precision (#879). Remote CUDA only; no physics-validation claim. |
| `line_roulette_mass_jitter_879.py` | Repeats the #879 line-mass pass, line spectrum and raw `bincount` on identical input to count distinct bit patterns. Remote CUDA only; engineering evidence, no physics-validation claim. |
| `line_roulette_off_hashes_879.py` | Prints hashes of analog output for one transport core, to compare a commit against `main` (#879). No physics-validation claim. |
| `line_roulette_tail_879.py`, `line_roulette_tail_879.sbatch` | Yield RSE, ESS and largest share against N for one seed (#879), from whole-electron blocks with per-primary line mass only and no spectrum. Remote CUDA only; no physics-validation claim. |
| `roulette879_analysis.py` | Numerical helper for the #879 conditional spectral errors, positive probabilities and pilot-frozen predictor; imported by the gate and focused tests. Post-hoc ranking is a reference heuristic, not an optimized cost/error bound. No physics-validation claim. |
| `line_roulette_gate_879.sbatch`, `line_roulette_spectral_gate_879.sbatch` | Remote wrappers for the #879 restricted-grid control and full-production CXR line gate respectively. Submit only from a pinned synced checkout in a bounded allocation; engineering evidence, with generated measurements under ignored scratch. |
| `checkpoint_selective_memory.py` | Synthetic full versus selected HDF5 record reads (#612), with matched warm file caches, fresh reader processes, wall time, and peak host RSS. Run `uv run python checks/checkpoint_selective_memory.py`; no simulation or physics-validation claim. |
| `optical_stack_cuda.py` | Real CUDA parity of the O(N) many-layer TR stack solve on complex amplitudes and responses, plus warm CPU/CUDA throughput and device memory against the dense solve (`optical-stack-fast-solver`). |
| `optical_transition_cuda.py` | Real CUDA parity of optical-band TR/CTR: per-history power and complex bunch field before squaring (fused kernel and CuPy array route) and the optical scorer's self/pair/per-electron yields for importance-weighted sampled and Gaussian laws, on a Johnson–Christy/Rakić Au multilayer. Stage a clean tree with `pyrite remote sync`, run in a bounded GPU allocation, collect JSON evidence. No CPU fallback. |
| `castellano_slit_dr.py` | Coherent slit diffraction radiation against Castellano et al., PRE 63, 056501 (2001), Fig. 8 (#616). Finite PEC slit-screen source; y-polarized Fresnel lens image scanned over the unpublished detector distance and aperture. Focal-plane detection is excluded; near-field geometries fit only when those parameters are fitted. The κ ≤ k source truncation of the paper's ref. [13] changes the ratios by < 1 %. It also runs an image-plane relay model (Leissner et al. topology) with the pupil from PRE's acceptance and one free screen-referred detector radius, and asserts that the infinite-detector limit equals the focal-plane result. With the relay geometry from Leissner's thesis (PITHA 99/4), it evaluates three equal-étendue horn readings with no fitted parameter. Only the disc reading fits (rms 0.015–0.086, against ≥ 0.19 for the others), so the agreement is model-conditional. Screening record; no ledger claim. Runs in about 10 s. |
| `photon_first_flight_cuda.py` | Real-CUDA first-flight parity and bounded end-to-end timing for analytic stacks and oriented boxes (#341). Run in a bounded remote GPU allocation; no CPU fallback or separate physics-validation verdict. |
| `optical_response_cuda.py` | Real CUDA complex128 upload and planar optical TR parity for the pinned CC0 Au response; unsupported-band/component refusal. Stage with `pyrite remote sync`, run in a bounded GPU allocation, and collect JSON evidence. No CPU fallback. |
| `channeling_overflow.py` | Experimental full-depth planar momentum instrument (#522): nonwrapped differential gain, unresolved probability, in-band gain shift and closure-gated conditional angle samples. Fixed-energy research diagnostic; no production handoff or derived escape boundary. |
| `axial_escape_accounting.py` | Bounded deterministic Si TDS plane-wave comparison: a position-diagonal sink cannot identify outgoing Fourier-window escape for different incident momenta (#548). Research only; no physical escape model or validation verdict. |
| `axial_matching_study.py` | Research study of an energy-defined retained projector and return-flux loss, with Gamma-point and Bloch-grid TDS rates (#548). No dechanneling prediction or physical validation verdict. |
| `axial_fiber_andersen.py` | Compares merged `axial_fiber_source.py` runs with Andersen et al. (1982) Si data: Fig. 8 line energies, Table 2 2p→1s widths (Eq. 35 finite-thickness Lorentzian fit) and Fig. 11 absolute 2p→1s yield vs tilt (#992). Research only. |
| `axial_fiber_angular_map.py` | Off-axis Stokes map of the Si fibred axial source on a (θγ, φ) grid via `fiber_angular_source`; chunks merge to a count-weighted mean with chunk-spread SEM; per-angle line peak, cos 2φ/cos 4φ contrasts and polarization degrees (#992). Research only. |
| `axial_fiber_source.py` | Si on-axis two-time photon source of the fibred absorbing axial model (continuum TDS strings plus electronic kicks) in mergeable incident chunks for `pyrite remote`/SLURM arrays, with optional tilted incidence and a whole-window (`reference`, P = R) arm; `merge` compares detector-smeared peaks with Watson & Koehler (1982) and `paired` gives chunk-paired yield changes between arms (#992). Research only; not a validated material source. |
| `axial_outgoing_convergence.py` | Paired convergence study of the absorbing fibre unraveling: retained/bound weight, exposure and return-flux deficit per numerical arm with shared seeds (#882). Research only; no physical escape fraction. |
| `axial_depth_calibration.py` | Compiles a Si axial depth budget per band from merged `axial_fiber_source.py` arms and `axial_fiber_fates.py` samples (#993), with interval components, audit numbers and a batch-doubling normality check (`cr-axial-depth-calibration`). |
| `axial_depth_calibration_993.sbatch`, `axial_depth_calibration_993.sh` | Remote CPU SLURM array wrapper running `axial_fiber_source.py` or `axial_fiber_fates.py` chunks, and the reduce script that merges each case's arms and fates and compiles the budget with `axial_depth_calibration.py` (#993). Submit from a synced `~/pyrite-993` checkout; engineering evidence. |
| `axial_fiber_fates.py` | Unit-weight terminal fates (incident, fate, depth) of the Si fibre model from the #883 physical outgoing instrument, in mergeable chunks, for axial depth budgets (#993). Research only (`cr-axial-zone-handoff-physical`). |
| `channeling_submev_axial_screen.py` | Static sub-MeV axial channeling-radiation screen (#585) for Si, diamond and W low-index axes at 30 keV–1 MeV: band structure, candidate bound→bound forward lines with dipole strengths, entry populations and kinematic/optical flags. Screening record, not a support verdict; no ledger claim. |
| `axial_outgoing_maps.py` | Research builder of material TDS jump maps (retained, represented-outgoing and unresolved effects) for the absorbing jump instrument (#623). Research diagnostic only; production handoff remains proposed. |
| `axial_mcwf_cuda.py` | Bounded remote-GPU benchmark of matched CPU/CUDA axial MCWF runs, recording cold and warm timing, setup, transfer and memory evidence; `--workload` adds heavier supercell cases, comparing CPU/CUDA throughput per electron-step when `--cpu-max-electrons` caps the CPU run. It is performance evidence for the CUDA backend, not a separate physics-validation verdict. |
| `axial_energy_mcwf_cuda.py` | Bounded remote-GPU batching benchmark of energy-resolved axial MCWF (#627): CPU reference (capped electrons) versus a CUDA admission `chunk_size` scan over monoenergetic and Gaussian-quadrature beams, with warm timing, per-electron-step throughput, CuPy pool footprint and statistical result summaries. Performance evidence only; no physics-validation verdict. |
| `axial_mcwf_precision_cuda.py` | Remote-GPU paired complex64 versus complex128 axial MCWF ensembles (same Philox streams; jump kernels float64) with a predeclared tolerance against Monte Carlo standard error, jump-sequence agreement and timing. Research measurement only; production axial MCWF stays complex128. |
| `axial_recording_performance.py` | Bounded remote-GPU axial diagnostic recording benchmark: cold/warm timings, GPU utilization, memory, source hashes and paired observable arrays. Performance/parity evidence only; no separate physics-validation verdict. |
| `axial_split_step_precision_cuda.py` | Remote-GPU complex64 versus complex128 tolerance and throughput of the deterministic axial no-jump split step (norm/jump-clock error, infidelity, density L1 versus depth). Research measurement only; production axial MCWF stays complex128. |
| `axial_mcwf_return_flux.py` | MCWF study of return-flux deficit under an energy-cut handoff, including TDS and electronic jumps (#548); optional `--loss-coupling` records cut/clearance and cross-cut loss histograms (#623). Research only; no native launch model or physical validation verdict. |
| `channeling_material_survey.py` | Catalog-wide thermal, shell-oscillator, plane-incidence and static-profile availability inventory (#525). JSON report only; no physics-validation verdict or ledger evidence record. |
| `channeling_submev_screen.py` | Static sub-MeV planar screen (#585): Si/diamond low-index bound bands over the half zone, bound→bound vacuum forward line ranges with velocity-form dipoles, aligned-entry bound populations, and harmonic/k-sampling refinement. JSON only; no support verdict or ledger evidence record. |
| `trajectory_viewer_benchmark.py` | Tracks-only Plotly/PyVista loading, construction, camera screenshot, filtering and memory probe on identical saved captures. Replication diagnoses resource scaling, not transport physics; no ledger records. |
| `physics_adapter_overhead.py` | Bounded default-physics CPU phase benchmark across matched source revisions; separate startup/warm timings, source keys, output hashes and RSS. No transport-validity claim; see [overhead evidence](../docs/repo-design/compute/physics-adapter-overhead.md). |
| `cxr_reduction_benchmark.py` | Remote-GPU CUDA CXR reducer launch benchmark (#200): captures one production batch, then times the unchanged reducers with hashes, RSS and device details. Performance evidence only; no physics-validation verdict or ledger records. |
| `transport_kernel_variants.py` | Remote-GPU exact CUDA transport kernel: #827 specialized variants against the monolithic kernel. Registers, occupancy, compile times and warm timing on the #821 anchors, with every arm hashed bitwise. Performance evidence only; no ledger records. |
| `transport_kernel_builds.py` | Remote-GPU exact CUDA transport kernel compile options (#796): register caps, launch bounds and `__ldg` read-only loads through the raw-module path. SASS load/spill counts and bitwise-matched warm timing. Performance evidence only; no ledger records. |
| `unit_conversion_overhead.py` | Scalar and NumPy-array unit-normalization microbenchmark with numeric and Pint inputs. Timing evidence only; no physics-validation verdict or ledger records. |
| `cxr_reduction_benchmark.py` | Captures one production CXR batch on a remote GPU, then times CUDA reduction launch configurations on it with matched outputs (#200); see [launch benchmarks](../docs/repo-design/compute/cxr-launch-benchmarks.md). Performance evidence only; no physics-validation verdict or ledger records. |
| `full_track_bremslib/` | Pinned Geant4 TestEm5 and exact PyRITE CPU full-track comparison for W and Si at 300 keV, including cutoff sweep, raw outputs, and unresolved W transport mismatch (#182); production-stack primary T/R vs Geant4 DPWA single scattering (#183, #317). |
| `mev_full_track/` | Geant4 11.4.2 TestEm5 (`empenelope`, primary only) against PyRITE production transport for W 100 µm and Si 1 mm at 3–100 MeV: transmission, continuous deposit, exit loss and angle, and primary bremsstrahlung; heavy runs on the lab box (#439). |
| `photon_full_track/` | Thick stopping W/Si targets at 0.3–10 MeV: Geant4 photon-only transport against weighted PyRITE aperture spectra. Compact independent-history blocks pin the historical 98/100 Auger-off result and the complete-cascade 100/100 corrected matrix that anchors the claim (#341). |
| `brem_primary_band_benchmark.py` | Captures bounded #821 transport populations and replays matched per-primary bremsstrahlung band scoring, precision variants and long-history accumulation stress (#938). Remote GPU; all outputs must be gitignored. Performance/parity evidence only; no physics-validation claim or ledger records. |
| `brem_band_request_confirmation.py` | Runs bounded #821 requests with portable or fused bremsstrahlung band scoring on the same seed and transport controls (#938); no Geant4 arm. Remote GPU; outputs must be gitignored. Performance/parity evidence only; no physics-validation claim or ledger records. |
| `brem_band_kernel_diagnostic.py` | Saves generated CUDA precision specializations and measures staged quadrature/primary kernels separately from warmed inclusive production scoring ranges (#938). Remote GPU; outputs and compiler cache must be gitignored. Performance/parity evidence only; no physics-validation claim or ledger records. |
| `brem_source_comparison.py` | EEDL and BremsLib bremsstrahlung `chi`, hard cross section, and radiative moment against the pinned Seltzer–Berger tables for every catalogue element, 1 keV–30 MeV; gates BremsLib, reports the EEDL interpolation defect (#174). |
| `sb85_eta_small_kappa_fit.py` | Refits the small-`k/T` scale of the packaged Seltzer–Berger (1985) electron–electron η table against the pinned SB digital tables and prints per-element fits, spread and the stored Z-averaged value (#439). |
| `srd164_inner_shell_comparison.py` | EEDL inner-shell ionization cross sections against NIST SRD 164 (Bote–Salvat) to 100 MeV for the shell-rate closure oscillators; reports the PENELOPE `rho` and outer-shell scale `N(E)` with either source. Report only; no gate, ledger evidence in the `penelope-shell-rate-closure` notes (#439). |
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
| `longitudinal_coherence_regimes.py` | Regime tables for the #629 research note: PINEM Talbot lengths, quantum versus classical harmonic-bunching caps, the ordinary-beam pair factor, and recoil-exact Si/diamond(110) CR lines. Exits 0 when the internal limits hold; under a second on CPU. |
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
| `inner_shell_experiment_comparison.py` | EEDL and tabulated Bote–Salvat against cited thin-film K and Pb/Th L-subshell measurements; descriptive overvoltage-band report, no agreement gate (`bote-salvat-ionization-backend`). |
| `shell_ionization_comparison.py` | Pinned Bote–Salvat `xion.f` transcription, local EEDL shell interpolation, and characteristic-production ratios for the 24 catalogue elements. |
| `brem_angular_comparison.py` | BremsLib angular shape (enclosed-flux angles) against the Schiff formula, Koch–Motz 2BS, at 5–30 MeV, gated for catalogue elements with Z ≥ 3; H reported. |
| `shell_soft_hard_transport_observables.py` | Opt-in shell soft/hard inelastic transport against continuous stopping on Si/SiO2/MoS2 at 5/20/100 keV: stopping closure along trajectories, straggling, transmission, backscatter, range, `W_c` convergence, energy conservation, event contract. |
| `shell_secondary_transport_observables.py` | Secondary transport (#94) on Si/MoS2 slabs at 20/100 keV: threshold convergence of energy backscatter/transmission, depth dose, characteristic and bremsstrahlung yields; energy balance, event contract per track, and the generation-1 launch spectrum against the free-electron Moller DCS. |
| `straggling_defaults/` | Remote-only current-model collision-straggling sensitivity pilot (#642): pinned shell/continuous on/off arms, raw per-primary energy/terminal scores, independent seeds, warm timing, separate step/cutoff probes and existing hardware anchors. No production default or accuracy qualification. |
| `segment_escape_split_ladder.py` | Split-segment ladder (k = 1, 8, 32 collinear pieces) for HOPG C K, bremsstrahlung and PXR/CBS yields: characteristic and bremsstrahlung escape must be invariant; the transparent-escape control isolates the line routes' midpoint escape bias (GPU; remote only). |
| `soft_deflection_line_sensitivity.py` | Paired line-spectrum response of a Si shell-mode transport to emulated soft inelastic direction wander (per row) and extra angular diffusion (per vertex), against seed-to-seed Monte Carlo error (GPU; remote only). |
| `soft_inelastic_deflection.py` | Soft and hard inelastic angular transport rates of the closed shell model against the elastic Mott transport rate, and per-row soft deflection, on Si/SiO2/MoS2 at 5–100 keV. |
| `pxr_optimization.py` | Bounded fixed-window PXR/CBS optimizer comparison and independent finalist checks (#682). Study diagnostic; no independent physics-validation verdict. Run on the lab worker. |
| `continuum_grid_convergence.py` | Continuum photon-grid candidates on fixed tracks (#278): intrinsic yield, centroid and detected counts against a reference grid. Study diagnostic; no independent physics-validation verdict. |
| `continuum_grid_end_to_end.py` | End-to-end continuum grid accuracy and runtime comparison (#278); study diagnostic, no independent physics-validation verdict. |
| `electron_count_accuracy.py` | Accuracy of line and continuum observables versus electron count `Ne` for one production case (#278). Study diagnostic; no independent physics-validation verdict. |
| `line_grid_candidates_286.py` | Fixed-track bin-mean line-grid coarsening study with two finer references, shared band edges, warm timing, cell-mass observables and fixed detector channels (#286). Submit `line_grid_candidates_286.sbatch` from a synced checkout; study evidence, no independent physics-validation verdict. |
| `line_setup_support_286.py` | Remote CUDA root-equivalence probe for the guarded shared table bound (#286); `line-tabulation-nan-ceiling` evidence. |
| `line_policy_286.py` | Same-track production uniform/adaptive policy pilot (#286): resolver/cache costs, refusals, audited scoring, common-band controls and refined references. Submit `line_policy_286.sbatch`; remote CUDA, measurement only, no adoption or ledger claim. |
| `cxr_bin_mean_profile_1003.py` | Capture and replay MeV bin-mean flushes through the reducer arms, with and without the midpoint series: time per flush, deviation from `auto`, repeat determinism, stage split (#1003). Submit `cxr_bin_mean_profile_1003.sbatch` (capture and profile), `cxr_bin_mean_replay_1003.sbatch` (replay and `ncu`) or `cxr_bin_mean_wall_1003.sbatch` (warm wall per primary and spectra, base commit against this checkout). Measurement only; no independent physics-validation verdict. |
| `line_sparse_near_286.py` | Tiled and sorted sparse-near prototype of the bin-mean reduction (#286), kept as a #1003 comparison arm. Measurement only; no independent physics-validation verdict. |
| `line_cost_286.py` | Per-flush attribution of the incoherent line phase: counts, reducer path, tree-build and kernel time (#286). Measurement only; no independent physics-validation verdict. |
| `line_crossover_286.py` | End-to-end line crossover measurement for issue #286; engineering evidence, no independent physics-validation verdict. |
| `detector_cone_variance_reduction.py` | Detector-cone variance-reduction baseline and comparison; engineering evidence, no separate physics-validation verdict. |
| `launch_retune_875.py` | Launch-shape and compile-option retune of the exact CUDA kernel: warm transport and CUDA-event kernel time per arm, with bitwise identity against the same-precision `t128` arm (#875). Measurement only; no independent physics-validation verdict. |
| `sinc_bin_integration.py` | Bin-mean line quadrature on one transport: yield against an exact node reference, and CPU/CuPy/fused-CUDA agreement (remote only at 300 keV). |
| `substep_invariance.py` | Measures emitted-radiation invariance under numerical transport substepping. |
| `fp32_speed_of_light/` | Timing-only FP32 and mixed-precision rewrites of the exact CUDA transport kernel's generated source, run through the #821 benchmark harness on the lab GPU (#831). `run.sbatch` runs the variants, `analyze.py` gives speedups and the observable deviation preview. Not a production option; no physics-validation claim. Results: "Mixed precision" in `docs/repo-design/compute/gpu-transport-rawkernel.md`. |
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

## Continuum benchmark support

These diagnostics record engineering and statistical evidence; they do not emit
independent physics-validation verdicts.

| Entry | Purpose |
| --- | --- |
| `continuum_backend_parity.py` | Compare CPU, CUDA and chunked scores on saved trajectories. |
| `continuum_benchmark_report.py` | Render benchmark JSON and spectral comparisons. |
| `continuum_geant4_precision.py` | Drive Geant4 precision measurements. |
| `continuum_primary_tail.py` | Report per-primary continuum moments and tail shares (#278); study diagnostic, no independent physics-validation verdict. |
| `continuum_pyrite_precision.py` | Drive PyRITE precision measurements. |
| `continuum_spectral_statistics.py` | Reduce per-primary Geant4 spectral sidecars. |
| `continuum_benchmark_matrix.sbatch` | Submit the grid benchmark matrix. |
| `continuum_cpu_reservation.sbatch` | Reserve CPU resources during GPU measurements. |
| `continuum_geant4_isolated.sbatch`, `continuum_geant4_queen.sbatch`, `continuum_geant4_spectra.sbatch` | Submit Geant4 benchmark workloads. |
| `continuum_precision.sbatch`, `continuum_pyrite_isolated.sbatch` | Submit precision workloads. |
| `escaped278` | Escaped-photon comparison and prepared-scorer benchmark harnesses; follow-up ownership in #821/#825. |

## Scheduled evidence coverage inventory

Static review of the 52 mapped producers (54 distinct claims), 2026-11-10.
The mapping in `pyrite.validation.check_records` identifies claim associations;
it does **not** certify a producer's exit code as a pass/fail gate. Unmeasured
runtime labels below are routing decisions; measured rows refer to the
admission samples documented below. Unmeasured transport,
solver and ensemble jobs must be profiled on the lab host before admission to
a bounded scheduled manifest. Never launch the entire inventory locally.

`Diagnostic` means the entry point prints/reduces results without establishing
a complete threshold verdict; exit 0 must not be promoted to scientific pass.
`Review` means internal guards exist but full claim gating needs inspection.
`Gate` means the producer has an explicit failure exit/exception; its required
mode, inputs and claim coverage still need review before scheduling. In
particular, multi-mode scripts must run their comparison/check mode, not just
capture or reduction. No row here promotes a physics claim.

The default CPU schedule includes `brem_angular_comparison.py` and
`transition_radiation_omission.py` (blocked by #1064 and #1065), plus
`longitudinal_coherence_regimes.py` (local smoke passed, covering two claims),
`shell_ionization_comparison.py` (remote gate passed in 51.69 s), and the
optional `dans_diffraction_oracle.py` / `xraylib_cascade_oracle.py` gates
(remote passes in 3.20 / 1.46 s). The full-default
`channeling_modulated_emission.py` gate also passed remotely in 21.58 s, and the
explicit-only multilayer analytic and slice-3 gates in 10.21 / 36.71 s.
The source table comparison has a reviewed
recipe but fails its existing gate (#1068); it remains explicit-only. CUDA
parity has its separate lab cron. Remaining ensembles require explicit remote
recipes. The refresh driver accepts only exact invocations in its `CPU_RECIPES`
registry, and validates every path before creating artifacts or starting any
producer, including direct `refresh()` calls and explicit `--check` selections.
The original three defaults, both oracle gates and channeling-emission gate
use no arguments; the shell
comparison and explicit-only source comparison use `--download` to obtain their
pinned references. Manifest command rows record exact arguments and elapsed
wall seconds.
The ledger index and all listed parts must be present before producers run;
remote sync omits documentation, so stage those tracked inputs and historical
`docs/validation/check-records/` separately in a dedicated profiling checkout.
Missing index/parts are rejected before execution rather than losing timing
metadata after an expensive probe.
`--all-mapped` is refused while any mapped entry lacks a reviewed recipe;
run diagnostic tools directly when investigating them. The raw
`pyrite-dev validation-records` collector still interprets process exit codes;
its callers must review the exact mode rather than treating mapping as proof
of a scientific gate. Full evidence coverage requires threshold adapters or
reviewed verdicts for the diagnostic rows, not merely rerunning their entry points.

| Producer (`checks/`) | Runtime / route | Prerequisites | Verdict contract |
| --- | --- | --- | --- |
| `abtem_channeling_oracle.py` | Optional CPU oracle; unmeasured | abTEM environment; explicit run and compare modes and captured artifacts | Gate (explicit mode where applicable) |
| `axial_depth_calibration.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_fiber_andersen.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_fiber_angular_map.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_fiber_fates.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_fiber_source.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_outgoing_convergence.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `axial_outgoing_maps.py` | Remote CPU ensemble; unmeasured | Explicit run/merge/compare recipe and retained arm/fate inputs; Andersen comparison needs paper fixture | Diagnostic |
| `brem_angular_comparison.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `brem_source_comparison.py` | Remote CPU table; 16.08 s (one warm sample) | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `channeling_handoff.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `channeling_modulated_emission.py` | Remote CPU; 21.58 s (one sample) | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `channeling_overflow.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `coherent_flat_omission.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Review |
| `coherent_physical_population_remote.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Review |
| `coherent_transverse_coherence.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `coherent_window_excluded_dispersion.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Review |
| `collision_statistics_refinement.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `cross_reflection_coherence.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `dans_diffraction_oracle.py` | Remote CPU oracle; 3.20 s (one sample) | Optional Dans_Diffraction installation; packaged crystal definitions | Gate (explicit mode where applicable) |
| `detector_solid_angle_check.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `elsepa_line_sensitivity.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `energy_loss_straggling_observables.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `energy_step_convergence_matrix.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `feranchuk_check_script.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Diagnostic |
| `feranchuk_vs_zhai_check.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `kephart_occupation.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `kinematic_validity_check.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Diagnostic |
| `line_setup_support_286.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Diagnostic |
| `line_window_backend_agreement.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Gate (explicit mode where applicable) |
| `longitudinal_coherence_regimes.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `mosaic_mc_check.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `multilayer_check.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `multilayer_slice3_check.py` | Remote CPU transport; 36.71 s (one sample) | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `multilayer_validation_check.py` | Remote CPU transport; 10.21 s (one sample) | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `optical_response_cuda.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Gate (explicit mode where applicable) |
| `optical_stack_cuda.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Gate (explicit mode where applicable) |
| `optical_transition_cuda.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Gate (explicit mode where applicable) |
| `quantum_decoherence_budget.py` | Remote CPU; 1.51 s (one sample) | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `radiation_error_estimator_calibration.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `sb85_eta_small_kappa_fit.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Diagnostic |
| `segment_escape_split_ladder.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `shell_ionization_comparison.py` | Remote CPU table; 51.69 s (one warm sample) | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Gate (explicit mode where applicable) |
| `shell_secondary_transport_observables.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `shell_soft_hard_transport_observables.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `sinc_bin_integration.py` | CUDA allocation; input/ladder-dependent | NVIDIA environment; some probes need captured trajectories or explicit CLI modes | Gate (explicit mode where applicable) |
| `soft_deflection_line_sensitivity.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `soft_inelastic_deflection.py` | CPU analytic/table; unmeasured | Packaged physics/catalog data; source and shell comparisons additionally need pinned reference/released tables | Diagnostic |
| `substep_invariance.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Diagnostic |
| `transition_radiation_omission.py` | Remote CPU transport/solver; unmeasured | Pinned tables/catalog and case inputs; convergence grids/ensembles may be expensive | Gate (explicit mode where applicable) |
| `transport_core_goldens.py` | Remote CPU capture/check; unmeasured | Existing reviewed golden directory; capture is not a validation verdict | Gate (explicit mode where applicable) |
| `xraylib_cascade_oracle.py` | Remote CPU oracle; 1.46 s (one sample) | Optional xraylib installation; packaged EADL cascade data | Gate (explicit mode where applicable) |

### First bounded table-recipe measurements

Job 10901, revision `80218ed8a`, 2026-11-10: one warm-reference sample per
recipe after job 10900 downloaded prerequisites. CPU backend, AMD Ryzen 9
9950X3D, two allocated CPUs, BLAS/OpenMP set to one thread, 8 GiB memory limit.
The two-recipe harness took 68.14 s wall and peaked at 1,739,092 KiB RSS;
per-recipe peak memory was not measured. These are admission measurements,
not an optimization comparison or timing guarantee. Both commands include
process startup and pinned-reference lookup/download handling.

| Exact producer invocation | Wall seconds | Exit / mapped claim verdict | Scheduling decision |
| --- | ---: | --- | --- |
| `checks/brem_source_comparison.py --download` | 16.08 | 1 / `brem-source-comparison`: fail | Explicit-only; 61 existing gate violations (46 H, 15 Li), no tolerance change. |
| `checks/shell_ionization_comparison.py --download` | 51.69 | 0 / `eedl-shell-ionization-comparison`: pass | Admit to bounded CPU defaults. |

Source prerequisites are pinned Seltzer–Berger reference data and released
BremsLib tables; shell prerequisites are pinned Bote–Salvat parameters and
packaged EEDL data. The remote manifest, exact arguments, logs, records and
ledger preview remain under `evidence/validation-344-profile/10901/` in the
dedicated profiling checkout; they were not promoted into tracked claim
records. Job 10900 evaluated both gates but could not render its preview
because remote sync omits ledger documentation. Stage the tracked ledger
inputs before remote refresh; the driver now checks that prerequisite before
starting a producer.

### Optional oracle recipe measurements

Job 10902, revision `6f8833d69`, 2026-11-10: one default-mode sample per
oracle, without an explicit warm-up, after `uv sync --all-groups --locked`.
That provisioning matches the existing scheduled workflow; no dependency or
workflow change was required. Python 3.14.6, Dans-Diffraction 3.4.0, xraylib
4.3.0, NumPy 2.5.4 and SciPy 1.18.1 ran on the same CPU/backend/resource limits
as the table sample above. The harness took 4.78 s wall and peaked at
251,964 KiB RSS; per-oracle peak memory and timing spread were not measured.
These single samples establish bounded admission, not performance guarantees.

| Exact producer invocation | Wall seconds | Exit / mapped claim verdict | Scheduling decision |
| --- | ---: | --- | --- |
| `checks/dans_diffraction_oracle.py` | 3.20 | 0 / `dans-diffraction-oracle`: pass | Admit; locked `oracle` group provisions Dans-Diffraction. |
| `checks/xraylib_cascade_oracle.py` | 1.46 | 0 / `characteristic-radiation`: pass | Admit; locked `oracle` group provisions xraylib. |

Both optional dependencies return exit 2 when unavailable; the refresh records
`skip`, never `pass`. Local use can provision them with `uv run --group oracle`.
The crystallography recipe evaluates its fixed material/reflection/energy set
with existing finite/tolerance gates. The cascade recipe preserves the existing
10% vacancy and 5% line bounds and fails if any prescribed case produces no
gated reference comparisons, closing the former empty-reference false pass.
No physical threshold or model was changed. Results, versions, logs, records,
manifest and preview remain under `evidence/validation-344-profile/10902/`
in the dedicated profiling checkout; claim records were not promoted.

### Quantum and channeling-emission gate measurements

Job 10903, revision `cb496c867`, 2026-11-10, ran each exact default invocation
once, without an explicit warm-up. Same CPU/backend/resource limits as above;
Python 3.14.6, NumPy 2.5.4 and SciPy 1.18.1 were provisioned with the existing
locked groups. Separate manifests and time reports preserve each result.

| Exact producer invocation | Producer wall seconds | Harness peak RSS [KiB] | Verdict / scheduling decision |
| --- | ---: | ---: | --- |
| `checks/quantum_decoherence_budget.py` | 1.51 | 280,368 | Fail / explicit-only: its old not-computable table rejects newly available MoS2 channeling data; material coverage needs review. |
| `checks/channeling_modulated_emission.py` | 21.58 | 266,128 | Pass / admit to CPU defaults; the conditional lossless-comb gate was exercised at 17 MeV. |

The quantum sample retained Si(110)/diamond(110) at all four prescribed
energies and audited all four HOPG/vdW entries. No rate bound was relaxed or
material assertion removed. Nonfinite rates and limit statistics now fail
rather than evade comparisons. Diagnostic channeling-radiation formation
lengths can remain unavailable when fewer than two bound states exist.

The emission sample retained all defaults: Si(110), 0.3/3/17 MeV, 16 cells,
maximum order 16, 2 micrometres, and all prescribed entry states/tilts. No
quick mode was substituted. The lossless-comb comparison ran at 17 MeV;
0.3 and 3 MeV had entry losses above its existing applicability condition.
Its existing entry/quadrature bounds are unchanged;
nonfinite overlaps, entry fluxes and yields now fail. If no prescribed energy
exercises the conditional comb gate, exit 2 records incomplete coverage rather
than a pass. The single admission measurements do not establish timing spread
or performance guarantees. Evidence remains under
`evidence/validation-344-profile/10903/`, with separate producer subdirectories
and peak-memory reports; no claim records were promoted.
