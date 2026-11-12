# Energy-resolved axial MCWF trajectories (issue #627)

**Experimental API** (sequential CPU reference plus a batched CUDA backend) building on [incident-energy ensembles (#626)](channeling-energy-ensemble.md). The energy of each electron is sampled **once at entrance** and remains fixed through every jump. Dynamic energy loss is *not* implemented; correlated energy–time records and arrival-time slices enter through [`axial_phase_space_mcwf` (#628)](channeling-energy-time-phase-space.md#correlated-axial-mcwf). This is distinct from running one averaged axial solve independently at each quadrature node.

## Contract

The `IncidentEnergyEnsemble` supplies zero-spread, bounded truncated-Gaussian or explicit weighted energies. `AxialEnergyMCWFOptions(sampling="binned")` selects an energy **node** with the #626 quadrature weights for each electron; `sampling="continuous"` uses a direct inverse-CDF draw from the same truncated Gaussian. Explicit weighted energies remain discrete in either mode. The `electron_energies_eV` override supplies measured individual energies (strictly positive, one per electron). The original Gaussian tail exclusion probability remains in provenance; the assigned electrons represent the **retained, renormalized** distribution.

Each electron uses independent Philox namespaces for entrance energy, transverse entry and MCWF trajectory. The individual trajectory seed is derived from the stable global electron ID; admission `chunk_size` **cannot** change its history. Outputs are accumulated in **increasing electron-ID order**; the `chunk_size` parameter only changes the admission loop, because the CPU reference solver deliberately executes one electron per underlying MCWF call. It is **not** optimized for bulk CPU performance or equivalent to the CUDA batch scheduler in the experimental #620 branch.

At each assigned kinetic energy, the solver reconstructs the corresponding relativistic kinematics, carrier tilt, split-step Hamiltonian and absorptive/electronic/TDS dissipator. An LRU cache retains at most `max_cached_solvers` such setups. The `entry_factory(E_eV, electron_id, rng)` callback returns a physically correlated `(EntryState, tilt_rad)` pair; a factory can therefore maintain correlations between energy and transverse entrance phase space without imposing energy-time correlations. With no factory, the supplied common `entry` and `config.tilt_rad` are used.

No wavefunctions or incompatible energy-dependent density matrices are averaged. Every reported density integrates to the fraction surviving per incident electron. Physical pixel-pair coherence includes the energy-specific carrier phase; OAM modes use one geometric disk. An `observable_factory(E)` can supply **physically matched linear observables** (e.g. fixed-space populations). Populations in different energy-dependent eigenbases are not automatically interchangeable; interpret energy-bin quantities within each bin, or project onto a common physical basis first. Sinked particles contribute zero.

## Example

```python
import numpy as np
from pyrite.montecarlo.channeling import (
    IncidentEnergyEnsemble,
    AxialEnergyMCWFOptions,
    axial_energy_mcwf,
)
from pyrite.montecarlo.channeling.axial_propagation import AxialPropagationConfig

spec = IncidentEnergyEnsemble(5e6, energy_spread_frac=0.002, order=9)
config = AxialPropagationConfig(
    kinetic_energy_eV=5e6,
    mode="mcwf",
    shape=(32, 32),
    depths_ang=(0.0, 10.0, 20.0),
    step_ang=2.0,
    oam_string=0,
)
options = AxialEnergyMCWFOptions(histories=200, seed=627, chunk_size=25)
results = axial_energy_mcwf(
    spec,
    config,
    options=options,
    coherence_pairs=((0, 0), (0, 1)),
    photon_energy_edges_eV=np.linspace(100, 10_000, 301),
)
# Per incident electron: surviving, density_per_ang2, coherence_per_ang2
# Energy-conditioned arrays: surviving_by_bin, density_by_bin_per_ang2,
# oam_populations_by_bin, jumps_by_bin, photon_density_by_bin_per_eV
assert results.bin_counts.sum() == options.histories
```

`result.energies_eV[i]` is the actual assigned kinetic energy of electron `i`; the energy histogram uses `energy_bin_edges_eV`, `energy_bin_centers_eV` and `bin_counts`. Empty quadrature bins have zero conditioned reductions. `surviving_by_bin[b]` is a fraction of the electrons **assigned to that bin**. The unconditional average is `sum_b (bin_counts[b] / histories) * conditioned[b]`. `jumps_by_bin` stores **integer event counts**, not probabilities or wavefunction phases. `coherence_per_ang2` is the lab-frame fixed-pixel coherence (the diagonal is a density).

For the optional radiation histogram, the energy-specific on-column bound-state catalog and polarization lines are independently built at every assigned energy. Each trajectory's **polarization intensity** is integrated across depth and deposited into fixed **photon-energy bin edges**; distinct photon energies/lines are never averaged by line index. `photon_density_per_eV` and the energy-bin-conditioned version are photons per incident electron per eV. `helicity_plus_per_eV` and `helicity_minus_per_eV` are separately reduced. `detector_sigma_eV` represents **instrument** smearing only; it does not apply an additional incident-energy spread. Radiation is zero where no supported on-column line catalog exists. This radiation model inherits the approximations/validity limits of the experimental axial line diagnostic.

## Checkpoint/restart and resource limits

```python
checkpoints = []
opts = AxialEnergyMCWFOptions(histories=200, seed=627, checkpoint_every=50)
r = axial_energy_mcwf(spec, config, options=opts, checkpoint_callback=checkpoints.append)
# Resume deterministically from a previously persisted checkpoint object:
r2 = axial_energy_mcwf(
    spec,
    config,
    options=AxialEnergyMCWFOptions(histories=200, seed=627, chunk_size=7),
    resume_from=checkpoints[-1],
)
```

`AxialEnergyMCWFCheckpoint` is a bounded, read-only snapshot of raw running bin sums, event counts and the last processed global electron ID. The callback receives the full object; **the API does not itself write files**. The caller may persist it (for example as a trusted Python pickle or a custom NPZ serializer), then supply `resume_from`. Its digest validates assignments, simulation geometry, observed pixel pairs, energy binning and detector energy grid. The caller must ensure that the same entry/observable/radiation callbacks and physics version are restored; arbitrary callback source cannot be hashed. The callback is emitted every `checkpoint_every` electrons, excluding the final total; no snapshots are retained by default. Snapshots require approximately twice the reduction memory and count against the bound.

Limits are explicit: at most 100,000 histories; at most 256 energy bins and 128 pixel coherence pairs; at most 4,096 photon-energy bins; at most 2,048 distinct solver energies by default (`max_distinct_energies`); maximum 8 concurrently cached energy solvers; `work_limit` bounds `electrons * pixels * split steps`; `storage_limit_bytes` preflights returned reductions and per-electron level-observable histories. **Solver-internal dissipator cache and FFT scratch allocations are not hard-capped by this number** and may consume additional RAM. If continuous sampling needs more distinct energies, increase that cap and plan for repeated solver construction. It is much more expensive than reusing a small quadrature grid.

## Validation status

The isolated deterministic tests cover zero spread, explicit weights, Gaussian inverse-CDF statistics, exact chunk-size independence, restart equivalence, correlated entrance tilts, energy-conditioned probability accounting, energy-shifted photon binning, helicity and resource validation. A separate integration test with actual PyRITE TDS solver is supplied and is skipped if the `xraydb` dependency is unavailable.

## CUDA backend and envelope

`AxialEnergyMCWFOptions(backend="cuda")` runs the trajectories on the complex128 axial CUDA MCWF solver (NVIDIA GPU with the `nvidia` extra). Electrons are admitted in global-ID chunks of `chunk_size`, split at every `checkpoint_every` boundary. Inside a chunk they are grouped by their fixed entrance energy, and each group runs as device batches on its energy-specific Hamiltonian and dissipator.

**RNG addressing.** Entrance energy and entrance phase space use the same host NumPy Philox streams as the CPU reference, so `energies_eV`, bin assignments, `bin_counts` and `entry_factory` calls (in electron-ID order) are identical across backends. Each trajectory uses the axial CUDA Philox4x32-10 stream keyed by `mcwf_stream_key(seed ^ 0x6272, electron_id)`, with the global electron ID as the key. Batch membership, energy grouping, admission and deaths do not move any electron's draws. CPU trajectories use NumPy Philox, so CPU and CUDA ensembles are **independent samples**. Their agreement is statistical, not paired.

**Reductions.** Density, fixed-pixel coherence (with each electron's own carrier phase), custom observables and the level history are summed on the device per energy group, in the CPU packed layout. Radiation is integrated from the group's summed level history. This equals the CPU per-electron sum because the depth trapezoid, line polarization and dipole yield (proportional to line strength) are linear. Group sums are added to their energy bin in chunk order and in first-electron-ID order within a chunk. Float rounding therefore differs from the CPU per-electron order and can differ between admission layouts (observed within `rtol=1e-10`). Jump counts, survival and bin counts did not change with batching or chunk size in any test or benchmark row.

**Memory admission.** Each batch is admitted by the axial CUDA rule (device budget: half of free memory, the resource policy cap, and the optional `memory_budget_bytes`). The admission first reserves the device dissipators of the other cached energies (`max_cached_solvers`, least-recently-used eviction). If those reserved bytes crowd out the current energy, they are released and admission is retried. Admission fails before upload if one electron cannot fit.

**Supported envelope.** Covered: binned, continuous or explicit energies; `entry_factory`; `coherence_pairs`; `observable_factory` outputs declaring `cuda_compatible = True`; radiation histograms through the built-in level observer; checkpoint/restart within the CUDA backend.

**Refused before solver construction:** `oam_string` (the OAM projection is host-only), `solver_factory`, `observable_factory` outputs without `cuda_compatible`, a missing CUDA device, and `memory_budget_bytes` on CPU.

CUDA checkpoints carry a backend tag in their signature, so they cannot resume a CPU run or the reverse. CPU checkpoint digests are unchanged.

AMD/ROCm GPUs (the lab `gpu-amd` partition) and complex64 are **outside** the envelope: neither has been run for this API, and production axial MCWF stays complex128. Dynamic energy loss is still not modelled.

### Evidence

Revision `b68995200`, SLURM job 4112, NVIDIA RTX 5080, CuPy 14.2.0, CUDA runtime 13020. Raw record: `docs/validation/check-records/axial-energy-mcwf-cuda.json`. All 25 selected tests passed, including the slow tests, with no skips:

- **Global-ID addressing.** A monoenergetic run reproduces a direct `AxialMCWF(backend="cuda")` run with `seed ^ 0x6272`. Jump counts and survival match exactly; density matches within `rtol=1e-10`.
- **Admission invariance.** For two energies with energy-correlated random tilts, chunk sizes 40 and 3 and a restart from a CUDA checkpoint give identical jumps, survival and bin counts. Density and coherence match within `rtol=1e-10`.
- **Deterministic reductions.** With no dissipation, two 10 MeV energies and per-electron tilts, CUDA matches the CPU reference within `rtol=1e-9` for density, coherence, a custom observable, energy-conditioned photon density and helicities.
- **Statistical parity.** For energy-conditioned survival, quadrant populations and per-electron jump counts, CPU and CUDA means agree within four combined standard errors plus a two-electron floor, over 8 seeds × 64 electrons with two energies.

Batching benchmark (`checks/axial_energy_mcwf_cuda.py`): Si ⟨001⟩ at 1 MeV, 32² grid, 100 steps of 10 Å. The CPU reference is capped at 64 electrons. Throughput is in electron-steps per second, from the warm median.

| Beam | CUDA electrons | CPU | CUDA, `chunk_size` 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|
| monoenergetic | 256 | 11 800 | 25 900 | 34 400 | 35 800 | — |
| 3 nodes | 256 | 11 200 | 22 200 | 25 100 | 29 900 | — |
| 7 nodes | 1024 | 7 330 | 18 200 | 27 700 | 33 100 | 36 200 |

Over that scan, CUDA runs 2.0–4.9 times faster than the CPU reference. The gain grows with `chunk_size`, because larger energy groups give larger device batches. With several energies it also needs `max_cached_solvers` at least the number of distinct energies; otherwise each chunk rebuilds dissipators. The first call at a new batch shape pays CUDA graph and FFT-plan setup: cold runs of 10–12 s appeared at chunk sizes 256 and 1024. CuPy pool reserve was 39–144 MB. These figures cover this device and workload only. They do not establish acceleration for small grids or few electrons, where the CPU can be faster.
