# Material composition

Away from the crystallography, PyRITE describes matter with exactly one currency: a list of `(element, number density)` pairs, densities in atoms per $\AA^3$. Attenuation, stopping power, bremsstrahlung, and elastic scattering all consume that list and nothing else. This page defines it, states where each composition comes from, and records the additivity assumption every consumer relies on.

## The composition record

```{math}
:label: eq-material-composition-record

\{(\text{el}_i,\; n_i)\},
\qquad n_i~\text{in atoms\,Å}^{-3}.
```

Number density, rather than mass density plus atomic weight, is the primitive. That choice removes a whole class of inconsistency: there is no mass density to keep in agreement with a lattice, and the standard atomic weight $A$ cancels out of the stopping-power coefficient exactly (see [Elemental transport data](../atomic-physics/elemental-transport-data.md)).

Compositions arise two ways.

**Crystalline layers** derive theirs from the structure, by counting basis sites per unit cell and dividing by the cell volume — see [Crystal structure representation](crystal-structure.md). The density is therefore the CIF's, at the CIF's refinement conditions, and cannot drift from the lattice used for the coherent sum.

**Amorphous media** carry theirs as configured data, because there is no cell to count. Fused silica is entered directly as $n_{\rm Si} = 0.02205$ $\AA^{-3}$ and $n_{\rm O} = 0.04410$ $\AA^{-3}$ — the 1:2 stoichiometry at 2.2 g cm$^{-3}$. A medium is a pure absorber and scatterer: it has no lattice, no reflections, and radiates no coherent lines.

## Additivity

Every consumer treats a compound as independent atoms at their respective number densities, with no molecular or condensed-phase correction.

**Attenuation** adds inverse absorption lengths:

```{math}
:label: eq-material-composition-mu

\mu(E) = \sum_i \frac{1}{L_{{\rm abs},i}(E)}
       = 2 r_e \lambda \sum_i n_i f''_i(E).
```

**Stopping power** adds per-element terms (Bragg's rule), each retaining its own mean excitation energy $J_i$ and its own low-energy correction $k_i$; there is no effective $Z$ or effective $J$ for the mixture.

**Bremsstrahlung** adds per-element cross sections scaling as $Z_i^2 n_i$.

**Elastic scattering** takes per-element free paths and cross sections from the same list.

The physical content of {eq}`eq-material-composition-mu` and its siblings is the independent-atom approximation: chemical bonding does not perturb the response. For X-ray attenuation this is accurate at the percent level away from edges, and degrades near them, where the fine structure that bonding produces — XANES and EXAFS oscillations — is entirely absent from the free-atom tables. A model attenuation edge is a step, not a step with structure on it.

## Which materials can be used as a homogeneous medium

Not every catalog entry denotes one uniform substance, and the attenuation entry point enforces the distinction. A catalog **crystal** or **medium** key resolves to a single composition and is accepted as a filter or absorber. A runnable **target material** is rejected, because its film-plus-stack structure is not one homogeneous filter medium: it has layers at different depths with different compositions, and collapsing them to a single composition would silently discard the layer geometry that the layered optical depth exists to represent. Layered absorbers are handled by walking the stack; see [Multilayer film-on-substrate materials](multilayer-materials.md).

The attenuation entry point also validates its inputs strictly rather than coercing them: the energy grid must be one-dimensional, nonempty, finite, and strictly positive, and every composition entry must have a positive finite number density. The returned coefficient array is read-only, so a shared attenuation curve cannot be mutated by one consumer under another.

## Composition gates transport support

A material's composition is the union over its film crystal and every substrate or stack layer. That union must be a subset of the supported transport-element table, and catalog validation rejects the material by name if it is not — at load time, with the unsupported symbols listed, rather than partway into a sweep.

This is the practical meaning of the coverage tiers described in [Elemental transport data](../atomic-physics/elemental-transport-data.md): X-ray response is available for any element, transport constants for 24, and Mott elastic cross sections only for elements whose NIST SRD 64 tables the user supplies.

## Assumptions and limits

* Independent atoms: no molecular, bonding, or condensed-phase corrections; no near-edge fine structure.
* Homogeneous within a layer. Density gradients, porosity, roughness, and interfacial mixing are not represented; a layer is uniform to its boundaries.
* Static densities. No thermal expansion, no compaction, no amorphous-density variation — a configured medium density is whatever was entered, and a crystal density is whatever the CIF implies.
* Attenuation is passive primary-beam attenuation only: photons removed from the beam are gone. No scattering back into the beam, no fluorescence re-emission, no diffraction, no secondary production.
* Elemental, not isotopic. Standard atomic weights are terrestrial averages.

## Validation

`absorption-length` (`anchored`) covers the per-element $\mu = 2 r_e \lambda n f_2$ used in {eq}`eq-material-composition-mu`, including the mixture limit. `self-absorption` (`rederived`) covers the layered path-integral built on top of it, and `positioned-filter-attenuation` (**`unverified`**) covers the finite-plate filter geometry. Crystal compositions inherit the `crystals-cif-adapter` (`anchored`) row through the site counting they derive from; configured medium densities are input data and carry no derivation row of their own.

See the [validation ledger](../../validation/physics-validation-ledger.md). No row here is human `signed-off`.

Implementation owners: `pyrite.materials.attenuation` — `linear_attenuation_inv_mm`, `_mu_total_inv_ang`, `_stack_tau`; `pyrite.materials.catalog` (`MediumSpec`, `CrystalInfo.composition`, and the transport-element gate).
