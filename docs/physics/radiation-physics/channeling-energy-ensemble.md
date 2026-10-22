# Incoherent incident-energy ensembles for quantum channeling

**Experimental Python API** — [issue #626](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/626). Per-electron energy-resolved MCWF is provided by the separate [issue #627](channeling-energy-resolved-mcwf.md) CPU reference API. Correlated energy/time phase space remains outside scope (#628). These ensemble wrappers do not change PyRITE's production electron transport.

The optional diagnostic API lives at `pyrite.montecarlo.channeling.energy_ensemble`, and its symbols are also exported by `pyrite.montecarlo.channeling`. A kinetic energy is always in **eV**. `energy_spread_frac` follows the established `BeamSpec` convention: a fractional **RMS** spread in the kinetic energy, not in total energy or in Lorentz factor.

## Numerical model

For mutually incoherent incident energies, a measured physical observable is

\[
  \langle O \rangle = \int_0^\infty p(T)\,O(T)\,\mathrm{d}T
           \simeq \sum_i w_i O(T_i),\qquad \sum_i w_i=1.
\]

For a Gaussian input, the interval is bounded to $[\max(0,T_0-L\sigma_T),T_0+L\sigma_T]$ with $L=4$ by default (zero is treated as an open physical boundary), using `order=17` Gauss–Legendre nodes weighted by the Gaussian PDF. The finite-interval weights are renormalized to one. `quadrature().excluded_probability` records the parent-Gaussian mass outside the retained interval; `mean_eV`, `rms_eV`, integration bounds, rule and node count are also available. The number of solves is bounded by `max_nodes=65` by default. Increase **both** `order` and `cutoff_sigma` separately for convergence. A zero spread produces *one exact node* at $T_0$ with weight one, bypassing Gaussian quadrature entirely.

Explicit nodes permit measured or asymmetric distributions. Nodes must have positive finite kinetic energies and finite nonnegative weights; zero-weight nodes are ignored and remaining weights renormalized. This also supports a **monoenergetic** explicit node. No phase relation between incident-energy nodes is assumed.

## Planar examples

```python
import numpy as np
from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import (
    IncidentEnergyEnsemble,
    planar_solver_factory,
    planar_energy_diagnostic,
    planar_energy_spectrum,
)

profile = planar_profile("silicon", (1, 1, 0), 293.15)
beam = IncidentEnergyEnsemble(5e6, energy_spread_frac=0.002, order=17)
builder = planar_solver_factory(profile, n_cells=4, max_order=8)

sample_x = np.linspace(0, profile.period_ang, 129)
diagnostic = planar_energy_diagnostic(
    beam,
    builder,
    1e4,
    100,
    spatial_x_ang=sample_x,
    record_every=5,
)
# These remain incident-electron-normalized, including dechanneling sinks.
assert np.max(np.abs(diagnostic.balance_defect)) < 0.1

photon_eV = np.linspace(100, 20_000, 300)
spectrum = planar_energy_spectrum(
    beam,
    builder,
    1e4,
    photon_eV,
    theta_rad=0.0,
    dz_ang=100.0,
    sample_ang=500.0,
)
# spectrum.spectrum is the averaged d^2N / (dE dOmega).
# spectrum.stokes was averaged through intensity-weighted Q,U, not by
# averaging normalized Stokes values at each energy.
```

`planar_solver_factory` **recomputes** the Hamiltonian, Bloch eigenstates, and TDS/electronic scattering at every node. A custom `solver_factory(kinetic_energy_eV)` may be provided, but it must return a solver built at the requested energy, with a common lattice geometry and plane-wave basis. Passing a prebuilt fixed-energy solver defeats this requirement and is rejected when its nominal energy differs. A consistent spatial grid and depth grid are enforced.

The final planar `plane_wave_density` converts each energy-specific *eigenbasis* density matrix to the common, physically identified plane-wave basis **before** averaging; this retains physically meaningful final-state coherence. `band_populations` denotes the energy-ordered band index at each energy, **not** population of a single fixed eigenvector. If crossings or branch permutations matter, inspect individual node results (`retain_node_results=True`) or use real-space observables. Eigenbasis density matrices and line catalogs are never averaged.

Planar `energy_spread_eV` analytical line broadening is **incompatible** with this explicit ensemble integration and is rejected when nonzero. Per-node `channeling_spectrum` is called with zero analytic energy-spread broadening. Doppler shifts, energy-dependent transition strengths, dissipators and absorption are evaluated inside each individual solve.

## Axial examples

```python
from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig
from pyrite.montecarlo.channeling import axial_energy_diagnostic, axial_energy_spectrum

config = AxialDiagnosticConfig(
    kinetic_energy_eV=5e6,
    depths_ang=(0.0, 50.0, 100.0),
    step_ang=5.0,
    shape=(32, 32),
    mode="coherent",
)
ensemble = IncidentEnergyEnsemble(5e6, energy_spread_frac=0.001, order=9)
result = axial_energy_diagnostic(
    ensemble,
    config,
    coherence_pairs=[(0, 0), (0, 16)],
)
# result.density_per_ang2: weighted incident-electron density by depth
# result.coherence_per_ang2: weighted rho(x_i, x_j) in one physical pixel basis
# result.wavefunctions_per_ang is None for a multi-energy ensemble

edges = np.linspace(100.0, 8000.0, 301)
radiation = axial_energy_spectrum(ensemble, config, edges)
# radiation.photon_yield_per_bin: photons per incident electron per bin
# photon_density_per_eV: photons per incident electron per eV
```

`axial_energy_diagnostic` constructs the energy-specific axial Hamiltonian, dissipator, real-space propagation and optional OAM projection on every node via `axial_diagnostic`. `coherence_pairs` is bounded to 128 pairs of flattened physical-grid pixels, with the carrier tilt phase included. Off-diagonal coherence is averaged as $\langle\psi(x_i)\psi^*(x_j)\rangle$ **per energy**, never by averaging amplitudes. Results at more than one node have no unique wavefunction or phase. For a monoenergetic pure coherent state, the original wavefunction is preserved exactly. `observable_factory(E)` can construct an energy-dependent, density-linear additional observable; its labels must have a common physical meaning across energies.

`axial_energy_spectrum` independently constructs the energy-dependent axial line catalog and forward photon energies at every node. Its spectrum is an incident-normalized intensity histogram across **photon-energy bin edges** (not an average of line *indices*). `detector_sigma_eV` optionally integrates a Gaussian *instrument* response over each bin; it is not a second incident-energy spread. Yield outside the requested photon-energy range is not included, and no renormalization of the emitted spectrum is applied. Helicity channels are accumulated separately.

## Convergence, checks and costs

- Quadrature work is linear in the number of nodes: a 17-node ensemble reruns the quantum solver 17 times. The wrapper streams reductions and drops individual histories by default; opt in to `retain_node_results=True` when comparing eigenstructure and transition shifts, accounting for memory.
- Use `check_quadrature_convergence(ensemble, evaluate, orders=(9, 17, 33), rtol=...)` for a fixed physically labeled array (e.g. density or spectrum). It fails rather than quietly returning a nonconverged last order. Repeat with larger `cutoff_sigma` to bound truncated tails.
- Probability conservation is assessed by `planar_result.balance_defect`; axial physical-space density integrates to `surviving` (up to numerical integration and MCWF shot noise). Observe population and OAM *per incident primary*, not conditional on surviving particles unless explicitly requested.
- For independent numerical comparisons: use single-node results to check the mono limit exactly; compare an increasing Gaussian quadrature with high-statistics direct energy sampling at the **same truncation**; inspect bound-state energies, decoherence rates and Doppler-shifted lines at each node against direct `transverse_states`, kernel and radiation calls.
- Coherence averaged in a fixed **plane-wave or physical-position basis** is legitimate. Do not average eigenbasis density matrices with energy-varying eigenvectors, wavefunction amplitudes, MCWF jump counts, or misaligned line catalogs.

The API provides energy-ensemble diagnostic and radiation calculations. It does **not** modify the marimo channeling app's controls, native transport energy sampling, the independent #627 trajectory API or temporal-radiation phase space.
