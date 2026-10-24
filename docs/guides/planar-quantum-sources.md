# Portable quantum sources in planar diagnostics (#680)

The validated `TransverseStateSpec` can now be projected into the existing
planar channeling Bloch density matrix and evolved **diagnostically**, without
constructing a transport `Case` or pretending a source has been injected into
production electron trajectories and channeling-radiation scoring.

```python
import pyrite as pr
from pyrite.planar_source import propagate_planar_source

state = pr.TransverseStateSpec.recipe("gaussian", waist_ang=1.0, dimension=1)
setup = pr.QuantumSetup(
    region=pr.QuantumRegion(kind="planar", plane_hkl=(1, 1, 0)),
    dynamics=pr.QuantumDynamics(kind="channeling"),
    numerics=pr.QuantumNumerics(basis_size=8, bloch_cells=4, step_ang=50.0),
)

entry = setup.project_state(
    state, crystal="silicon", kinetic_energy_eV=17e6, incidence_rad=0.0, impact="fixed"
)
assert abs(entry.total_population - 1) < 2e-3
result = propagate_planar_source(entry, depth_ang=500.0)
print(entry.retained_population, entry.entry_population, result.trace[-1])
```

The source is specified in the **beam's transverse frame**, with x along the
plane normal. `incidence_rad` is the **effective refracted incidence to the
planes**, not a target tilt in the laboratory frame; for a physical Scene
compute it using `materials.channeling.entry.plane_incidence` after applying
the scene's crystal orientation. `impact="fixed"` retains the source's
position modulo the lattice plane spacing and off-diagonal plane-wave
coherence. `impact="averaged"` loses the absolute impact phase; 2-D states
are reduced over their in-plane y momentum, so planar evolution does **not**
track electron orbital angular momentum. Source mixtures use incoherent
weights; superpositions retain complex phase.

The population bookkeeping is

`retained_population + sum(free_population) + outside_population = 1`.

Here `free_population` is the initial in-band weight removed by the channeling
cutoff and `outside_population` is momentum **outside** the plane-wave basis.
Neither is a dechanneling event. Propagation uses the solver's existing
`initial=(subspace_density, entry_flux)` contract, and returns its usual
`DensityMatrixResult`. An energy-calibrated source (aperture probe or sampled
state with `kinetic_energy_eV`) refuses inconsistent beam energies.

## Incident-normalized channeling-radiation diagnostics

The CR estimator can now reuse the **same projected initial density** as the
propagator, avoiding a second wavefunction projection or an inconsistent
incidence/divergence setting. This is useful for studying how controlled source
coherence and impact influence planar channeling lines.

```python
import numpy as np
from pyrite.planar_source import score_planar_source

energies = np.linspace(1_000.0, 50_000.0, 500)  # eV
spectrum = score_planar_source(
    entry,
    depth_ang=500.0,
    photon_energy_eV=energies,
    theta_rad=0.0,
    absorption=False,
)
print(spectrum.spectrum, spectrum.bound_population)
```

The result is the existing `ChannelingSpectrum`, in photons/eV/sr/incident
electron; it is not a persisted detector response. `theta_rad` and `phi_rad`
are in the beam/plane frame, not slab-lab coordinates. The projected state
already contains the normal momentum spread; specifying that spread again
as a classical divergence would double-count it. Only in-plane broadening
(`divergence_y_rad`) and independently specified energy spread remain optional
for 1-D sources. For **2-D sources**, supply `divergence_y_rad` explicitly:
tracing out the in-plane momentum removes its actual distribution, and a
single Gaussian width is a modelling approximation rather than a predicted
spread. Planar CR remains insensitive to the full photon OAM structure.

`sample_ang` controls line-rate sampling independently of the density solver
step (`setup.numerics.step_ang`). The optional `path_limits_ang` represents
geometric/cutoff path censoring only; never pass dechanneling survival there.
Absorption assumptions and photon escape remain those of the diagnostic CR
scorer. This does **not** validate electron handoffs, physical source-frame
matching, or replay identities for an end-to-end Monte Carlo run.

**Restricted production envelope (experimental):** A centered 1D Gaussian
with uniform lattice-impact averaging is supported through `Scene` and
`Beam(state=...)`. Direct `Beam.to_spec()` remains intentionally unable to
lower a quantum source without a compatible planar scene. The production
handoff is conditioned on sampled Bloch momentum, and CR live/replay scoring
shares the same projected source (see [conditional source validation](../validation/channeling/cr-planar-conditional-source.md)).
All other source types (including fixed-impact coherent states, 2D OAM
wavefunctions and custom sampled arrays) remain diagnostic-only. The outgoing
handoff still uses #506's approximate depth/angle/sign sampling and requires
independent convergence/physics validation. Axial production is still refused.

## Interaction with axial integration (#624)

Axial configuration and diagnostic propagation are independent from this planar
projection adapter. The continuation work on `issue-624-axial-consumer-refusals`
centralizes the **physical admission refusal**; this bridge uses the same
`campaign.axial_spec.refuse_unverified_axial_run` entrypoint, which delegates to
the shared refusal in that branch. A `QuantumSetup` with an axial region raises
before constructing planar kernels, reading sample arrays, or writing outputs.

The existing versioned `channeling_axial` candidate resolver is not converted
into a planar state and must not be advertised as a production quantum runner.
The #623 matching surface, source normalization, and case-specific depth budget
remain prerequisites for any supported axial transport and photon emission.
