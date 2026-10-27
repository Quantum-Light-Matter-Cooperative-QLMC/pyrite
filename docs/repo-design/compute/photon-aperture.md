# Opt-in photon aperture observations

`Physics(photon_transport=PhotonTransport(...))` connects the bounded CPU or CUDA
ordinary-hit implementation to `simulate`, scalar Scene lowering, the case runner,
checkpoint components and full trajectory replay. This is experimental model
selection, not full-track validation. The interaction, weighted-source and
bounded charged-feedback claims retain their existing validation states; no
new collision equation or default numerical model is introduced here.

## Selection and units

```python
from pyrite import Physics, PhotonTransport, simulate

physics = Physics(
    bremsstrahlung_model="bremslib",
    radiative_model="bremslib-soft-hard",
    photon_transport=PhotonTransport(
        edges_eV=(1000, 3000, 10_000, 100_000),
        source="brem",
        n_samples=1000,
        seed=341,
    ),
)
result = simulate(beam, target, ideal_planar_detector, physics=physics, numerics=numerics)
counts = result.aperture_dataframe()
flux = result.aperture_dataframe(current_nA=1.0)
```

The detector must be physical, ideal and unpixelated, without a scorer or
acquisition. The selected source is either weighted continuous `brem`, physical
`characteristic`, or physical `annihilation`. A continuous source requires
positive `n_samples`; event sources require `n_samples=None`. Event sources
require equal positive line/continuum primary populations, shell-soft-hard
transport and a secondary threshold equal to `electron_cutoff_eV`. Annihilation
also requires the existing primary pair/positron selectors; the other two
sources refuse them. Characteristic sources always prepare Auger transport.
`charged_feedback=True` enables photon-born charged descendants for any source.
The existing [source and feedback contracts](../../research/physics/photon-secondary-feedback.md)
still apply. With finite filters, charged descendants use the scene feedback
stage across target/filter bodies and intervening vacuum, including backscatter
into the target. Each filter must name a catalog material with shell, stopping,
ELSEPA and BremsLib data; inline `MediumSpec` filters are refused for charged
feedback. The primary charged source still uses its contiguous target stack.
Photon-only continuum and annihilation retain their existing filter support.
The no-filter contiguous feedback path and its identities remain unchanged.

`Result.aperture` is a `PhotonContinuumTally` or `PhotonEventTally`. Its densities
are photons/(original incident electron eV), integrated over the physical
aperture. `aperture_dataframe` retains exact lower/upper histogram edges and
direct, redistributed and total densities and bin counts. Supplying mean
current in nA multiplies by electrons/s, giving photons/(s eV) and photons/s.
No solid angle, detector response, second filter attenuation, or proposal
normalization is applied. Out-of-bin detections remain separately available
on the tally. Event tallies retain sub-cutoff source energy and characteristic
relaxation budgets.

The ordinary directional spectra, their per-sr units, and `spectral_flux()`
remain separate. Their characteristic/continuum families are alternative
views of contributions also represented by an aperture observation. No method
sums those views. Primary transport runs once; both scoring routes reuse its
rows. The aperture sampler and feedback use their own seeded streams without
changing the primary loss ledger.

## Identity and execution

Mode off adds no Case key, output key, checkpoint component or identity marker.
Mode on appends a resolved `Case.photon_transport` record. It pins the original
source content key and exact Case snapshot, source table manifests, histogram,
source selection, seed, cutoffs, physical aperture/filter geometry, photon data
and enabled feedback tables. The complete enabled record enters the existing
case-content key; the underlying source and aperture observation digests stay
separate. The descriptors `observation.photon/cpu-reference` and
`observation.photon/cuda-reference` document their execution requirements without
certifying a validated physical domain. CUDA has a separate enabled-only model
and execution identity; CPU records and mode-off identities retain their
historical bytes.

Scene re-entry adds a separate `cpu-charged-scene-feedback-v1` or
`cuda-charged-scene-feedback-v1` feedback record. It pins ordered filter material
keys, all bodies' electron stopping/elastic/BremsLib tables, positron table
manifests, navigation/core selection and the visit/relaxation caps. These pins
belong to the enabled observation; they do not change the original primary
source identity. Replay resolves each body again and refuses changed table
pins or a material/composition mismatch before primary transport.

Preparing a selected case checks the source snapshot and installed pins before
primary transport or output capture. Changing the resolved Case after lowering
requires lowering the Scene again. Unsupported backends, grooves,
channeling, TR, adaptive/block transport, sweeps and pixel/acquisition requests
are refused. The scalar Python API is the selection boundary in this slice;
catalog/CLI configuration and remote scheduling are not added.

## CUDA execution

Select `PYRITE_MC_BACKEND=cuda` before importing PyRITE and use
`Numerics(transport_core="cuda")` for explicit GPU primary transport. The
production photon selector accepts CPU or CUDA; it refuses other backends.
Missing CUDA raises rather than falling back to CPU photon sampling.

`photon_histories.track_photon_histories(backend="cuda")` retains the CPU
queue's lineage keys, photon cutoffs and energy ledger. Bounded event waves run
analytic first flights, Rayleigh/Compton, pair final states, photoelectric
sampling and sampled EADL relaxation on CUDA. Conditional samplers consume
explicit uniform prefixes; insufficient prefixes or emission capacity cause
same-stream retries, with no dropped histories or truncated cascades. Reordering
and chunking preserve each lineage's stream. Resource admission bounds scratch
storage before device allocation.

The queue, Philox streams, EPDL rate lookup, weighted BremsLib source proposal,
charged emission harvesting and `fsum` ledger remain host controlled. This is
not a fully device-resident shower. Charged feedback explicitly requests the
native `"cuda"` transport core, including small descendant populations. The
same named-material and ideal-aperture restrictions apply on both backends.
Scene charged navigation clips binary64 rays on the selected device; local
phase-space transforms, visit scheduling and energy substitution stay on the
host. Empty CUDA batches still require CUDA, with no CPU fallback.
The CUDA execution identity records these boundaries. Binary64 CUDA math and
CPU math can round differently; cross-device bitwise equality is not promised.
The first-flight timings below do not measure whole-shower throughput.

## Checkpoints and replay

Raw case outputs and `store_result` carry a separate `photon_aperture` payload
with original reduced-artifact schema, array bytes, checksum and provenance.
The content-addressable cache preserves it. Component checkpoints write
optional `photon.h5` before the authoritative line index, alongside the existing
directional components. Mode-off stores keep their historical layout. Saves
validate the observation against its Case before replacing any component;
loads refuse missing, corrupt or mismatched selected companions. No current
photon tables or accelerator are needed to read a finished observation,
including one produced on CUDA.

```python
from pyrite.checkpoints.persistence import load_checkpoint
from pyrite.results.aperture import aperture_from_record, aperture_dataframe

records = load_checkpoint("silicon", checkpoint_dir=checkpoint_root)
tally = aperture_from_record(records[case_name][beam_energy_keV])
counts = aperture_dataframe(tally)
```

Full `spectrum_from_artifact` replay regenerates the selected aperture from
retained primary rows without rerunning primary transport, checking the
original Case/data pins and original execution backend. A different backend
requires lowering the Scene again. Bounded replay refuses photon mode because weighted
source selection requires the full segment population. Partial field/dataset
projections must retain `photon_aperture`; directional-only partial transfers
are refused. Ordinary full slim/archive paths preserve it, and directional
float32 slimming leaves the checked aperture payload at its original dtype.

Independent thick-target W/Si, alloy fluorescence and 511 keV full-track
comparisons, pixel observables and generalized charged navigation remain open.

## Standalone CUDA first flights

`montecarlo.transport.photon_cuda.scene_first_interactions_cuda` provides an
explicit CUDA first-flight boundary for stacks/prisms and disjoint oriented
boxes. It accepts the host arrays and two uniforms per ray used by
`photon_navigation.scene_first_interactions`, returning CuPy arrays with the
same distance [Å], region, element and channel conventions. Missing CUDA raises
an error, including for empty inputs. CUDA selection at this boundary is
independent of the module-global backend.

Input validation, EPDL rates and the optical-depth draw conversion remain on the
host. The default `implementation="fused"` uses two binary64 CUDA kernels for
clipping and stable interval/channel selection, disabling fused multiply-add
contraction. It evaluates each distinct ordered composition once per batch;
repeated slabs reuse those rates without changing region or element labels.
The original shared NumPy/CuPy implementation remains available through
`implementation="array"`. `threads=128` sets the fused block size; supported
sizes are 32, 64, 128, 256 and 512. These are execution controls, with no RNG or
physics parameter changes.

The fused interval walk compares entry distances in stable region order, with
quadratic work in region count per ray. It targets bounded analytic scenes;
this is not a large-mesh navigator. Storage grows with ray count times region
count and distinct material channels. Callers bound batches and retain each
ray's original draws when chunking or reordering. Binary64 reductions and box
projections can round differently across devices, so CPU/GPU bitwise equality
at ambiguous geometric/CDF boundaries is not guaranteed.

This low-level boundary has no random-number generator or history queue;
`photon_cuda_events` and `photon_histories` supply the production orchestration
described above. Device parity tests cover navigation, conditional samplers and
whole histories. `tests/montecarlo/test_photon_production_cuda.py` exercises real
public simulations, feedback, capture/replay, checkpoint/CAS portability and
mode-off stability. These hardware tests skip when CUDA is absent. An evidence
job must set `PYRITE_TEST_BACKEND=cuda` as well as `PYRITE_MC_BACKEND=cuda` and
require zero skips, so unavailable or incorrectly selected hardware cannot turn
a hardware check into a passing skip. Hardware job 2524 on RTX 5080
passed all 9 production and 39 kernel/history tests with zero skips; compact
source/data hashes and individual test results are retained in
`checks/photon_cuda/production-rtx5080.json`. This is implementation evidence,
not an independent full-track physics comparison.

`checks/photon_first_flight_cuda.py` requires real CUDA, compares a fixed seeded
EPDL workload against the CPU reference, then prints JSON with source/data
hashes, device/runtime, repeated warm end-to-end timings and memory context.
It includes host table evaluation and both transfer directions in the CUDA
timing. Run it from a synced checkout in a bounded remote GPU allocation with
`uv run --extra nvidia python checks/photon_first_flight_cuda.py`.

`uv run --extra nvidia --extra optimization python -m
checks.photon_cuda.benchmark scaling` compares warm CPU, array-CUDA and fused-CUDA
first flights across ray counts and region counts. The `tune` mode reuses
`OptunaOptimizer`'s ask/tell interface to vary batch chunks and block sizes,
gating every candidate on fixed-input CPU parity. Selection timings and fresh
alternating finalist measurements are recorded separately; no tuned settings
are installed as defaults. The `trace` mode brackets warmed calls with the
CUDA profiler API for Nsight Systems. CUDA event/trace elapsed time includes
idle dispatch gaps, so active kernel durations and host API costs must be
examined separately. These experiments exercise first flights, not complete
photon histories or production scoring.

RTX 5080 measurements (CuPy 14.2.0, CUDA 13.2, five warm end-to-end repeats,
including host lookup and both transfers) give these median times in ms:

| Rays | Regions | CPU | Array CUDA | Fused CUDA |
|---:|---:|---:|---:|---:|
| 1,024 | 3 | 0.899 | 3.475 | 1.099 |
| 16,384 | 3 | 7.282 | 7.424 | 5.130 |
| 262,144 | 3 | 126.626 | 82.786 | 86.369 |
| 1,024 | 17 | 4.317 | 10.402 | 1.143 |
| 16,384 | 17 | 38.723 | 40.416 | 4.174 |
| 262,144 | 17 | 807.319 | 661.339 | 88.210 |

The 17-region scene repeats one composition across 16 target slabs, so its
gain includes host rate reuse as well as kernel fusion. These are analytic
first-flight workloads, not full photon-track speedups. The small-scene
Nsight capture falls from 222 kernel instances per call to two, and aggregate
active kernel duration from 0.423 to 0.030 ms per call. CPU profiling of the
fused 16,384-ray/3-region call attributes most time to host rate preparation.
The largest 3-region case demonstrates that fusion alone need not improve
end-to-end time.

A seeded 24-trial Optuna search selected chunk size 65,536 and 256 threads.
Nine fresh alternating measurements gave 95.32 ms median (93.43–96.35 ms)
against the fused unchunked 128-thread path's 103.22 ms (101.75–125.13 ms).
This comparison is separate from the scaling matrix and retains timing noise;
it does not establish a hardware-wide optimum. Raw timings, source/data hashes,
parity results and compact Nsight reductions are retained in
`checks/photon_cuda/performance-rtx5080.json`.


## Charged scene re-entry evidence

The named-filter production path and its CUDA implementation are checked by
`tests/montecarlo/test_photon_reentry_production.py`,
`test_photon_reentry_cuda.py` and `test_photon_reentry_pair_cuda.py`.
RTX 5080 jobs **2528** and **2529** each pass **22 tests, zero skips** under
`PYRITE_FP64=1` and the default `PYRITE_FP64=0`, respectively. Coverage includes
Si/W filter admission and table pins, all three source components, real
Si electron/positron gap/filter parity against CPU, explicit device clipping,
capture/full replay, checkpoint/CAS, fresh CPU reading and mode-off bytes.
The public 40 keV annihilation calls retain the physical zero-source limit.
Job **2530** additionally passes the nonempty pair/annihilation full queue
inside a production-prepared Si filter under both precision settings. Photon
channel rates are replaced throughout that probe to force pair conversion
above threshold and make returned subthreshold photons transparent. Conditional
final states, charged flights, vacancy feedback and annihilation remain real;
the high-energy exit count is checked in aggregate. This establishes 23 distinct passing
checks per precision setting, not external full-track agreement.

Source checkpoint `70d37305` serves the 22-test matrix; `cbcb7302` adds only
evidence and the separate pair probe. Runtime source hashes match across
those checks. Compact results, exact source/data hashes, hardware versions
and reproduction commands: `checks/photon_cuda/reentry-rtx5080.json`.
[Independent verification](../../validation/beam-transport/photon-charged-reentry.md)
keeps the bounded `photon-charged-reentry` claim at **rederived**. This does
not advance thick-target/alloy/511 keV comparison claims or add primary charged
filter transport, time-dependent scoring, pixel/acquisition response or catalog/CLI
configuration. Cu filters still fail preflight when charged feedback is requested
because the existing shell model has no Cu conduction-band data.
