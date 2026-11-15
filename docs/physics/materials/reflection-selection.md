# Reflection selection

Line emission is a sum over reciprocal-lattice vectors, and that sum has to be truncated. Which reflections enter affects both yield and scoring cost. PyRITE uses measured line-yield selection for supported incoherent runs, a family count for other routes or explicit requests, and catalog pins for textured crystals.

## Default measured selection

With neither `n_families`, `max_reflections` nor `line_yield_tolerance` set,
the spectrum phase scores reciprocal shells until the estimated omitted yield
is at most 1e-3 of the kept yield on the requested line axis. Shell spectra
are retained and summed. The automatic axis bounds the candidate cap before
selection, so it may be substantially wider than the four-family axis.
The estimator is qualified on the Si and MoS₂/Si scenes in
[the selector evidence](../../validation/radiation-physics/reflection-yield-tolerance-selector.md);
it is not a rigorous error bound for arbitrary scenes.

An explicit `CxrReflections(n_families=4)` or `--n-families 4` restores the
four-family policy. Explicit counts take precedence over automatic selection.
An explicit `line_yield_tolerance` requests selection and retains its
incompatibility errors. Automatic selection falls back to four families for
coherent emission, line roulette, temporal profiles, channeling, two-beam
transition radiation and streamed scoring. Catalog-pinned lists remain pinned.
Diffracted bremsstrahlung and Kossel terms keep the build-time reflection set.

The accuracy gain has a measured cost. On qlmc-ace (ROCm), the 64-history,
0.5–10 keV gate at 0.09375 eV bin spacing took 26.3 s for Si and
176–208 s for layered MoS₂/Si with incremental selection. The four-family
arms took about 1–2 s. On two CPU threads (qlmc-queen, same gate, node
quadrature, 101,335 nodes), Si took 24 s at four families and 2,185 s
(about 36 min) for the measured selection, which keeps G30 / 13,688
reflections. Layered MoS₂/Si took 156 s at four families, and its selected arm
had not finished when the 90 min job limit cancelled it. Treat the CPU path as
impractical for these wide-band, high-energy scenes; use a GPU or narrow the
declared line axis. These measurements exclude transport. Selection, ledger
status and the GPU measurements are in
[#1173](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1173);
#1200 sizes each selection call by its coupling-table memory and admits it
before scoring. With the sets unchanged, GPU selection got faster on three
axes:

- 1.6–2.2× faster on the 0.5–10 keV band.
- 2.3× faster on the layered automatic axis.
- No faster on the Si automatic axis.

See the
[reflection-selection benchmarks](../../repo-design/compute/reflection-selection-benchmarks.md).

## Automatic ranking

The fixed-family path enumerates candidate reflections and keeps the strongest *families*.

**Enumeration.** All integer triples with $|\mathbf g| \le g_{\max}$ are generated, with $g_{\max} = 8$ $\AA^{-1}$ by default. The per-axis index bound is exact rather than a guess:

```{math}
:label: eq-reflection-selection-index-bound

|h_i| \le \frac{g_{\max}\,|\mathbf a_i|}{2\pi},
```

which follows from $\mathbf b_i \cdot \mathbf a_i = 2\pi$ and bounds the search box tightly for any crystal system, including strongly anisotropic layered cells where a cubic guess would be badly wrong on one axis.

**Ranking.** Candidates are ordered by

```{math}
:label: eq-reflection-selection-metric

M(\mathbf g) = \frac{|S(\mathbf g)|}{g^2},
```

with the Debye–Waller factor already inside $S$. The $1/g^2$ is not a heuristic weight — it makes {eq}`eq-reflection-selection-metric` proportional to $|\chi_{\mathbf g}|$ evaluated at *each reflection's own* line energy. The argument: $\chi_{\mathbf g} \propto \lambda^2 S \propto S/\omega^2$, and the resonance energy of a reflection scales as $\omega_{\rm res} \propto g$ at fixed geometry, so $\chi_{\mathbf g} \propto S/g^2$. Ranking by {eq}`eq-reflection-selection-metric` therefore ranks reflections by the coupling strength they will actually radiate with, not by their structure factor at some fixed reference energy.

**Family grouping.** Symmetry-equivalent reflections are grouped by identical $(|\mathbf g|, M)$ to rounding. This recovers symmetry mates without any space-group machinery: two reflections related by a symmetry operation of the crystal necessarily share both quantities exactly, up to floating-point noise. All members of the top $n$ families are returned, Friedel mates $\pm(hkl)$ included, since $+\mathbf g$ and $-\mathbf g$ radiate into different directions and both must be summed. A representatives-only mode returns one deterministic member per family, for visualizations that should show each family once.

**Cutoff.** Families whose metric falls below $10^{-9}$ of the top family are dropped as forbidden or negligible, which removes extinct reflections without a special case for them.

Four families follow the published practice this project benchmarks against.
That truncation does not bound omitted yield: the #1143 Si and layered
MoS₂/Si scenes lose about 61 % and 78–79 % of their 0.5–10 keV yield.

## Explicit pinning

A catalog crystal may instead pin its reflection families outright. When it does, a free-text **reason is required** — the schema rejects a pin without one — so that a hand override always carries its justification next to it. Pinned representatives are given as positive-representative triples and are expanded automatically to both reciprocal directions.

The motivating case is fiber texture. HOPG is a fiber-textured aggregate: its $c$ axes are aligned but its in-plane orientations are random across grains, so in-plane reflections are incoherent across the illuminated volume and must not be summed as if they were a single crystal. Only the basal $(00l)$ reflections survive, and HOPG pins $(002)$ and $(004)$ with exactly that reason recorded.

This is the general caveat about the automatic ranking, stated as a rule: {eq}`eq-reflection-selection-metric` ranks the **crystal structure only**. Texture, preferred orientation, and the sample's actual grain statistics are not inputs to it. A material whose grains are not equivalent to a single crystal needs its families pinned by hand.

## What the ranking does not account for

* **Texture and preferred orientation** — as above.
* **Detector acceptance.** A reflection is ranked by coupling strength, not by whether its resonance lands in the observed energy window or its emission cone reaches the detector. A strongly coupled reflection outside the configured line grid still costs kernel time.
* **Geometry.** Ranking is done before the crystal is oriented, so the electron direction, tilt, and azimuth do not enter. Which pinned or ranked reflections actually radiate is decided later by the resonance condition.
* **Energy dependence.** The ranking is evaluated at a single reference photon energy (1 keV by default). This is exact for the non-resonant term, which is energy-independent; it matters only through the anomalous terms of edge-prone elements, and only enough to reorder near-degenerate families.
* **Absorption.** A reflection whose line sits below an absorption edge of the material may be strongly self-absorbed on the way out. That is applied downstream, in [photon escape](../radiation-physics/photon-escape-and-dispersion.md), not in the ranking.

## Assumptions and limits

* Kinematic single-crystal coupling for every enumerated reflection, inheriting the [structure-factor](structure-factor.md) assumptions.
* Family grouping infers symmetry from numerically identical metrics rather than from the space group; accidental degeneracy between genuinely inequivalent reflections would merge them into one family. Since both members would then be returned together and both would be summed, the consequence is a family-count accounting artifact, not a dropped reflection.
* The $g_{\max} = 8$ $\AA^{-1}$ cap corresponds to $s = 0.64$ $\AA^{-1}$, well inside the fitted range of the form-factor parameterization, but it is a hard truncation: reflections above it are never considered.
* Cost scales with the number of returned reflections, so raising the family count is a real runtime decision, not a free accuracy improvement.

## Validation

The ranking metric is a **selection policy**, not a physical claim, and carries no validation-ledger row of its own. What is pinned is its *output*: the catalog golden snapshot serializes, for **every** catalog crystal, the top two ranked families at a 1 keV reference energy together with $|\mathbf g|$ and the complex structure factor of the leading reflection (`tests/materials/test_material_catalog.py::test_catalog_matches_serialized_physics_for_every_crystal`). A change in selection therefore fails the golden comparison rather than passing silently. The snapshot is regenerated deliberately, never repaired in place — see the `regen-golden` workflow.

The physics the metric is built from is separately covered — `structure-factor` (`anchored`) for $S$ and its Debye–Waller factor, and the per-crystal rows in the [crystal structure provenance ledger](../../validation/ledger-crystal-structure-provenance.md) for pinned family choices such as the HOPG basal restriction.

See the [validation ledger](../../validation/physics-validation-ledger.md).

Implementation owners: `pyrite.materials.crystal.dominant_reflections`; pinned families in `pyrite.materials.catalog` (`CrystalSpec.hkl_families`, `CrystalSpec.hkl_list`); resolution at `pyrite.campaign.geometry.crystal_params`.
