# Experimental axial diagnostics

The channeling app and `axial_diagnostic` assemble the existing experimental
axial solvers. They add no physics, transport handoff, or campaign integration;
`Physics(channeling_model="axial")` remains refused. The core diagnostic needs
neither marimo nor plotting extras.

The proposed [standard-run contracts](../research/physics/axial-run-integration.md)
separate physical escape, numerical sinks and source scoring; they do not enable
Scene/profile axial selection.

## One Python call

```python
from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    axial_diagnostic,
)

result = axial_diagnostic(
    AxialDiagnosticConfig(
        crystal="silicon",
        axis=(0, 0, 1),
        kinetic_energy_eV=5e6,
        cells=(1, 1),
        shape=(32, 32),
        depths_ang=tuple(range(0, 10_001, 250)),
        step_ang=5,
        mode="coherent",
        seed=0,
    )
)
```

In `pyrite app channeling launch`, choose **Axial (experimental)** and press
**Run axial diagnostic** with the default settings to reproduce this solve.
Depth and conditioning selectors redraw stored arrays; **Play depth** steps
the recorded depth at the chosen frame interval (wrapping at the end) and
redraws the maps and radiation panels each frame. The last four distinct
completed solves are cached. The axial branch waits for Run, so ordinary static
export still executes the default planar demo. Depths run from 10 nm to 1 mm;
the app records 41 evenly spaced depths (`axial_demo_depths`). App bounds are
48² samples, 3² cells, steps of at least 1 Å, 32 MCWF electrons, and grid
samples × steps × trajectories ≤ 1.1e9 (`APP_WORK_LIMIT`, at most about a minute
on one core). At 32² and 10 Å steps this reaches 1 mm for a coherent pure state
and about 300 µm for 32 MCWF electrons. Maps are drawn with the projected origin,
and so string 0 and the default impact, at the panel center. These are
illustrative statistics: run expensive studies through the remote workflow,
not the interactive app or local heavy sweeps.

The API returns the resolved profile/grid, configuration (including seed),
entry state, sampled input norms, potential in eV, depths and coordinates in Å,
and density in Å⁻² with shape `(depth, x, y)`. Its cell integral is the
surviving incident-electron weight. `result.density(conditional=True)` divides
by that weight, returning zero when none survives. Each periodized entry
component is normalized on the grid before evolution, matching `AxialMCWF`;
`sampled_norms` records the norms before this normalization. This convention
describes a normalized periodic input, not a prediction of entrance losses.

## Entry states and scattering

The app offers a Gaussian, radial-order-zero vortices with ℓ = ±1, and an
equal **incoherent** ±1 mixture. Waist `w` follows `exp(-r²/w²)` for the
Gaussian; impact `(x, y)` is in Å relative to the projected cell origin.
The selected carrier tilt is in mrad in the app and radians in the API.
To use the existing entry-state interface directly:

```python
from dataclasses import replace
from pyrite.montecarlo.channeling.states import laguerre_gauss, mixture

entry = mixture(
    [
        (0.5, laguerre_gauss(1, waist_ang=0.5, center_ang=(0.2, 0.0))),
        (0.5, laguerre_gauss(-1, waist_ang=0.5, center_ang=(0.2, 0.0))),
    ]
)
scattering = axial_diagnostic(
    replace(result.config, mode="mcwf", n_electrons=16, seed=547),
    entry=entry,
)
```

MCWF uses the existing thermal-diffuse and electronic dissipator, including
out-of-band sinks, with per-electron Philox streams keyed by seed and electron
index. Requested depths must be increasing, nonnegative multiples of the
step; the wrapper rejects depths that the backend would round. Seed is
retained in coherent configuration too, but coherent evolution uses no RNG.
Mixture probabilities are averaged, never complex amplitudes. MCWF returns
ensemble densities and grid survival, with no unique phase. Coherent pure
states additionally return complex envelope histories in Å⁻¹ with the carrier
tilt factored out; their unitary norm is **not scattering survival**.

Set `oam_string=0` (or another profile string index) to request the existing
cylindrical projection. `oam_origin_ang` identifies the actual string position;
`oam_radius_ang` defaults to 0.5 Å. `oam_populations` reports ℓ = −8…8 weight
inside that disk **per incident electron**, without renormalizing omitted
radial or angular-mode weight. This is a local disk diagnostic, not total
cell OAM or a probability conditioned on survival.

## Channeling radiation

When the case has a flat-level line catalog (one string per projected cell,
as on silicon or diamond [001], at energies with at least two flat bound
levels), the app records each level's density on string 0's Wigner–Seitz
cell at every stored depth (`LevelDensity` through the `observable` argument
of `axial_diagnostic`). `axial_radiation` turns these into each line's
polarization matrix `T_ab` and its depth integral. The app then draws:

- photons per incident electron emitted before the selected depth, per line,
  split by the line's `J_z = ±1` content (the on-axis helicity);
- each line's emission rate versus depth, which falls as MCWF scattering
  empties the emitting level (coherent evolution keeps it constant);
- the selected line's far field versus `γθ`: intensity and phase of the
  components `E_±` on a fixed lab circular basis (`dipole_far_field`). A phase
  winding `ℓ` around the axis is orbital angular momentum `ℓ = J_z − σ`, so
  a pure `J_z = +1` line leaves `E₊` flat and winds `E₋` twice. `σ` is the
  photon spin projection a fixed circular polarizer selects; `J_z = σ + ℓ` is
  the total set by the electron transition. Each component has its own
  intensity scale; titles give its share of the mapped photons and the
  winding on the `γθ = 2` ring. Contours show the Doppler energy relative to
  forward. Dipole order allows only `J_z = ±1` (transverse `d`, no `J_z = 0`);
  `|ΔJ_z| ≥ 2` needs quadrupole or higher order, suppressed by
  `(state size / rest-frame wavelength)²`.

The planar mode draws the same maps for its real dipole along the plane
normal, an equal coherent mix of `J_z = ±1` with no net OAM. These are
dipole-order display diagnostics; the field claim is
[`cr-dipole-far-field-amplitude`](../validation/ledger-channeling.md#cr-dipole-far-field-amplitude)
and the levels and lines follow
[`cr-axial-helicity-emission`](../validation/ledger-channeling.md#cr-axial-helicity-emission).
On [110] (several strings per cell) no line catalog is built.

## Supported data and limits

Catalog thermal data currently cover silicon and diamond. The app offers
[001] and [110]; the API accepts axes supported by `axial_profile`, which
refuses oblique projected cells. Coordinates follow the returned profile's
transverse axes; boundaries are periodic in both transverse directions.

A narrow probe can require more momentum samples; a broad or drifting probe
can wrap around a small supercell. Independently vary grid shape, cell count,
and depth step. MCWF supercell images move together within one cell;
use at least 2–3 cells for convergence studies. A 1-cell demo is deliberately
small and is not a converged production prediction. Large tilt needs smaller
depth steps. Phase is undefined at density zeros and depends on the chosen
entry/global phase.

The existing claims and statuses remain unchanged:
[coherent propagation](../validation/channeling/cr-axial-realspace-propagation.md)
and [real-space dissipator](../validation/channeling/cr-axial-realspace-dissipator.md).
They document finite-grid sink/clamp effects, isotropic displacement limits,
and the remaining inhomogeneous electronic-ensemble validation gap. The
[ledger](../validation/ledger-channeling.md#cr-axial-realspace-dissipator)
remains the validation owner; this diagnostic surface does not promote it.

## Stored quantum evolution metrics

Both app modes show instantaneous state-population bars and depth heatmaps.
Planar bars include all retained Bloch bands, summed over blocks. Axial bars
sum each degenerate string-0 local level. These are local squared projection
weights: sampled orbitals need not be orthogonal. The Gram-matrix spectral
defect is displayed; survival minus their sum is a signed coverage diagnostic,
which can be negative and must not be interpreted as loss or unrepresented
probability. Without a supported local catalog,
the app labels and displays the complete finite position-grid basis instead.
Incident normalization is the default. Survivor conditioning divides each depth
by its surviving weight, giving zero populations for all-lost ensembles.
Controls only redraw stored arrays.

Purity uses the **ensemble** density, conditioned on its retained trace. Planar
purity is exact for the retained block-diagonal ensemble. Axial purity uses
32 fixed independent complex random probes of the full physical-position-grid
density. Error bars show one probe standard error, conditional on the empirical
finite-electron ensemble; they exclude trajectory sampling uncertainty and
finite-ensemble purity bias. The result records the number of electrons. Small
ensembles need independent seed/convergence studies. Raw estimates can exceed
one. Squared off-diagonal Frobenius coherence subtracts the exact squared
diagonal and depends on basis: instantaneous Bloch eigenstates for planar,
physical position grid for axial. Raw noisy coherence estimates can be negative.
These two basis-dependent quantities should not be compared as an invariant
coherence measure.

Transverse-energy mean and standard deviation are conditional Hamiltonian
moments in eV. Axial moments include both kinetic directions, carrier tilt and
the real potential. All conditional metrics are NaN when no weight survives.
Axial OAM curves remain truncated disk-local populations per incident electron,
not whole-cell angular momentum. Numerical grid sinks are not physical escape.

Use `record_metrics=True` with `axial_diagnostic` or
`DensityMatrixSolver.propagate` to request reductions. The app enables them.
Density-sketch output exceeding 128 MiB is refused before computation. The app
also limits recording to 41 depths. Metric recording retains no electron
wavefunction histories. See the
[metric derivation](../validation/channeling/cr-quantum-ensemble-metrics.md)
for normalization, uncertainty and validation limits.
