# Energy-resolved axial MCWF trajectories (issue #627)

**Experimental, CPU reference API** building on [incident-energy ensembles (#626)](channeling-energy-ensemble.md). The energy of each electron is sampled **once at entrance** and remains fixed through every jump. Dynamic energy loss, energy-time correlations and a validated CUDA per-electron backend are *not* implemented. This is distinct from running one averaged axial solve independently at each quadrature node.

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
from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig

spec = IncidentEnergyEnsemble(5e6, energy_spread_frac=0.002, order=9)
config = AxialDiagnosticConfig(
    kinetic_energy_eV=5e6,
    mode="mcwf",
    shape=(32, 32),
    depths_ang=(0.0, 10.0, 20.0),
    step_ang=2.0,
    oam_string=0,
)
options = AxialEnergyMCWFOptions(n_electrons=200, seed=627, chunk_size=25)
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
assert results.bin_counts.sum() == options.n_electrons
```

`result.energies_eV[i]` is the actual assigned kinetic energy of electron `i`; the energy histogram uses `energy_bin_edges_eV`, `energy_bin_centers_eV` and `bin_counts`. Empty quadrature bins have zero conditioned reductions. `surviving_by_bin[b]` is a fraction of the electrons **assigned to that bin**. The unconditional average is `sum_b (bin_counts[b] / n_electrons) * conditioned[b]`. `jumps_by_bin` stores **integer event counts**, not probabilities or wavefunction phases. `coherence_per_ang2` is the lab-frame fixed-pixel coherence (the diagonal is a density).

For the optional radiation histogram, the energy-specific on-column bound-state catalog and polarization lines are independently built at every assigned energy. Each trajectory's **polarization intensity** is integrated across depth and deposited into fixed **photon-energy bin edges**; distinct photon energies/lines are never averaged by line index. `photon_density_per_eV` and the energy-bin-conditioned version are photons per incident electron per eV. `helicity_plus_per_eV` and `helicity_minus_per_eV` are separately reduced. `detector_sigma_eV` represents **instrument** smearing only; it does not apply an additional incident-energy spread. Radiation is zero where no supported on-column line catalog exists. This radiation model inherits the approximations/validity limits of the experimental axial line diagnostic.

## Checkpoint/restart and resource limits

```python
checkpoints = []
opts = AxialEnergyMCWFOptions(n_electrons=200, seed=627, checkpoint_every=50)
r = axial_energy_mcwf(spec, config, options=opts, checkpoint_callback=checkpoints.append)
# Resume deterministically from a previously persisted checkpoint object:
r2 = axial_energy_mcwf(
    spec,
    config,
    options=AxialEnergyMCWFOptions(n_electrons=200, seed=627, chunk_size=7),
    resume_from=checkpoints[-1],
)
```

`AxialEnergyMCWFCheckpoint` is a bounded, read-only snapshot of raw running bin sums, event counts and the last processed global electron ID. The callback receives the full object; **the API does not itself write files**. The caller may persist it (for example as a trusted Python pickle or a custom NPZ serializer), then supply `resume_from`. Its digest validates assignments, simulation geometry, observed pixel pairs, energy binning and detector energy grid. The caller must ensure that the same entry/observable/radiation callbacks and physics version are restored; arbitrary callback source cannot be hashed. The callback is emitted every `checkpoint_every` electrons, excluding the final total; no snapshots are retained by default. Snapshots require approximately twice the reduction memory and count against the bound.

Limits are explicit: at most 100,000 electrons; at most 256 energy bins and 128 pixel coherence pairs; at most 4,096 photon-energy bins; at most 2,048 distinct solver energies by default (`max_distinct_energies`); maximum 8 concurrently cached energy solvers; `work_limit` bounds `electrons * pixels * split steps`; `storage_limit_bytes` preflights returned reductions and per-electron level-observable histories. **Solver-internal dissipator cache and FFT scratch allocations are not hard-capped by this number** and may consume additional RAM. If continuous sampling needs more distinct energies, increase that cap and plan for repeated solver construction. It is much more expensive than reusing a small quadrature grid.

## Validation status and CUDA

The isolated deterministic tests cover zero spread, explicit weights, Gaussian inverse-CDF statistics, exact chunk-size independence, restart equivalence, correlated entrance tilts, energy-conditioned probability accounting, energy-shifted photon binning, helicity and resource validation. A separate integration test with actual PyRITE TDS solver is supplied and is skipped if the `xraydb` dependency is unavailable.

**CUDA is intentionally rejected** (`backend="cuda"` raises `NotImplementedError`): current `main` contains only the CPU axial MCWF solver. The active `issue-620-quantum-cuda` branch includes an experimental CUDA runner with batch admission and CUDA-compatible callback requirements, but this wrapper's CPU `numpy` observer is *not* CUDA-compatible. Proper GPU support must implement global per-electron energy/RNG addressing, matching GPU observable reductions and an energy-aware memory admission policy, followed by CPU/CUDA parity and batching benchmarks. No parity or GPU speedup is asserted by this patch.
